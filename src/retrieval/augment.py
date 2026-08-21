import networkx as nx
import hashlib

from pathlib import Path
from collections import Counter
from src.chunking.ast_parser import CodeChunk
from src.chunking.dependency_graph import build_dependency_graph, get_related_files, save_graph, load_graph
from src.logging_config import get_logger
from src.logging_config import setup_logging
from src.chunking.ast_parser import chunk_repo
from src.retrieval.keyword_index import BM25KeywordIndex
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.hybrid import hybrid_retrieve

logger = get_logger(__name__)

MAX_AUGMENTED_CHUNKS = 10  # piece 2's configurable cap, referenced here for now


def build_combined_dependency_graph(repo_roots: list[Path]) -> nx.DiGraph:
    """
    Build one merged dependency graph across multiple repos, by
    building each repo's graph independently (build_dependency_graph
    only understands a single repo root) and composing them into one
    nx.DiGraph. Safe because every node is a repo-prefixed file path
    string, so there's no collision risk between repos — flask's
    app.py and click's app.py (if it existed) would be distinct nodes.
    """
    per_repo_graphs = [build_dependency_graph(root) for root in repo_roots]
    combined = nx.compose_all(per_repo_graphs)
    logger.info(
        f"Combined dependency graph across {len(repo_roots)} repos: "
        f"{combined.number_of_nodes()} nodes, {combined.number_of_edges()} edges"
    )
    return combined


def _build_file_to_chunks_index(all_chunks: list[CodeChunk]) -> dict[str, list[CodeChunk]]:
    """
    Group all chunks by file_path, so augment_with_related_files can
    look up 'what chunks exist in this related file' in O(1) instead of
    scanning the full corpus per related file
    """
    index: dict[str, list[CodeChunk]] = {}
    for chunk in all_chunks:
        index.setdefault(chunk.file_path, []).append(chunk)
    return index


def _rotation_offset(question: str, n: int) -> int:
    """
    Compute a deterministic rotation offset from the query string, so
    round-robin allocation doesn't always start at fused_chunks[0] (which
    would systematically favor whatever RRF happened to rank highest,
    every single query). Same question always produces the same offset
    (reproducible/testable); different questions produce different
    offsets (no single rank position is always privileged in practice).

    Uses a hash rather than random.shuffle specifically so this stays
    deterministic — the project's existing tests assert exact expected
    chunk sets, which requires stable, repeatable behavior.
    """
    if n == 0:
        return 0
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
    return int(digest, 16) % n


MAX_AUGMENTED_CHUNKS = 10  # configurable cap on related-file chunks appended per query


def augment_with_related_files(
    question: str,
    fused_chunks: list[CodeChunk],
    graph: nx.DiGraph,
    all_chunks: list[CodeChunk],
    depth: int = 2,
    max_augmented: int = MAX_AUGMENTED_CHUNKS,
) -> list[CodeChunk]:
    """
    For each of the fused (Day 8) chunks, look up related files via
    the dependency graph (files it imports, and files that import it,
    within `depth` hops) and append one representative chunk from each
    related file not already present in the result set — preferring a
    module_docstring chunk (usually the best single summary of what a
    file is for) and falling back to that file's first chunk otherwise.

    Additions are allocated ROUND-ROBIN across fused chunks, starting
    from a rotation offset deterministically derived from `question`
    (via _rotation_offset) rather than always starting at fused_chunks[0].
    Without this, whichever chunk RRF happened to rank highest would
    always get first pick in every query's round-robin allocation —
    this rotation spreads that advantage across different rank
    positions depending on the query, while staying fully reproducible
    for a given question (needed for deterministic testing).

    Capped at `max_augmented` appended chunks total, to guard against
    context explosion from depth=2 traversal.

    Requires all_chunks (the full corpus, not just the fused 15) because
    a newly-discovered related file's chunks are, by definition, not
    already among the fused results — that's the entire point of
    augmentation.
    """
    file_to_chunks = _build_file_to_chunks_index(all_chunks)

    result = list(fused_chunks)
    already_present_files = {chunk.file_path for chunk in fused_chunks}
    added_files: set[str] = set()

    n = len(fused_chunks)
    if n == 0:
        logger.info("Augmentation received zero fused chunks — nothing to augment")
        return result

    offset = _rotation_offset(question, n)
    rotated_indices = [(offset + i) % n for i in range(n)]

    related_queues: dict[int, list[str]] = {}
    for i, chunk in enumerate(fused_chunks):
        related_queues[i] = get_related_files(graph, chunk.file_path, depth=depth)

    made_progress = True
    while len(added_files) < max_augmented and made_progress:
        made_progress = False
        for i in rotated_indices:
            if len(added_files) >= max_augmented:
                break

            queue = related_queues[i]
            while queue:
                candidate_file = queue.pop(0)
                if candidate_file in already_present_files or candidate_file in added_files:
                    continue

                candidates = file_to_chunks.get(candidate_file)
                if not candidates:
                    continue

                representative = next(
                    (c for c in candidates if c.chunk_type == "module_docstring"),
                    candidates[0],
                )
                result.append(representative)
                added_files.add(candidate_file)
                made_progress = True
                break

    logger.info(
        f"Augmentation added {len(added_files)} related-file chunks "
        f"(round-robin, offset={offset}, cap={max_augmented}) on top of "
        f"{len(fused_chunks)} fused chunks"
    )
    return result



if __name__ == '__main__':
    # testing augment_with_related_files()
    setup_logging()
    repo_names = ["flask", "click", "jinja", "itsdangerous", "markupsafe", "werkzeug", "requests", "httpx"]
    repo_roots = [Path(f"data/repos/{name}") for name in repo_names]

    all_chunks = []
    for root, name in zip(repo_roots, repo_names):
        all_chunks.extend(chunk_repo(root, repo_name=name))
    print(f"Total chunks across all repos: {len(all_chunks)}")

    graph_path = Path("data/combined_dependency_graph.json")
    if graph_path.exists():
        graph = load_graph(graph_path)
        print(f"Loaded existing graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    else:
        graph = build_combined_dependency_graph(repo_roots)
        save_graph(graph, graph_path)

    bm25_index = BM25KeywordIndex(all_chunks)
    vector_store = VectorStoreManager()
    vector_store.upsert_chunks(all_chunks)  # idempotent, safe to re-run

    question = "Where is authentication middleware defined and what does it check?"
    fused = hybrid_retrieve(question, bm25_index, vector_store, top_k_per_source=15, top_k_final=15)

    print(f"\n--- BEFORE augmentation ({len(fused)} chunks) ---")
    for chunk in fused:
        print(f"  {chunk.name}  ({chunk.file_path}:{chunk.start_line})")

    augmented = augment_with_related_files(question, fused, graph, all_chunks, depth=2)

    print(f"\n--- AFTER augmentation ({len(augmented)} chunks) ---")
    for chunk in augmented:
        marker = "  [NEW]" if chunk not in fused else ""
        print(f"  {chunk.name}  ({chunk.file_path}:{chunk.start_line}){marker}")

    # testing augement_with_related_files()
    # in_degrees = [d for _, d in graph.in_degree()]
    # out_degrees = [d for _, d in graph.out_degree()]
    # total_degrees = [graph.in_degree(n) + graph.out_degree(n) for n in graph.nodes()]

    # total_degrees.sort(reverse=True)
    # print(f"Node count: {graph.number_of_nodes()}")
    # print(f"Top 15 nodes by total degree (in+out):")
    # degree_by_node = sorted(graph.nodes(), key=lambda n: graph.in_degree(n) + graph.out_degree(n), reverse=True)
    # for node in degree_by_node[:15]:
    #     print(f"  {graph.in_degree(node) + graph.out_degree(node):3d}  {node}")

    # print(f"\nMedian total degree: {total_degrees[len(total_degrees)//2]}")
    # print(f"Mean total degree: {sum(total_degrees)/len(total_degrees):.2f}")
    # print(f"Max total degree: {total_degrees[0]}")

    # for threshold in [5, 8, 10, 12, 15, 18, 20]:
    #     count = sum(1 for n in graph.nodes() if graph.in_degree(n) + graph.out_degree(n) > threshold)
    #     print(f"Nodes with degree > {threshold}: {count}")