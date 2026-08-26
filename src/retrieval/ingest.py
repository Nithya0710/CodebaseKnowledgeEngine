import json
import networkx as nx

from pathlib import Path
from src.chunking.ast_parser import chunk_repo, CodeChunk
from src.retrieval.keyword_index import BM25KeywordIndex
from src.retrieval.vector_store import VectorStoreManager
from src.chunking.dependency_graph import save_graph, load_graph
from src.retrieval.augment import build_combined_dependency_graph
from src.logging_config import get_logger

logger = get_logger(__name__)

MANIFEST_PATH = Path("data/ingestion_manifest.json")
BM25_CHUNKS_PATH = Path("data/bm25_chunks.json")
GRAPH_PATH = Path("data/combined_dependency_graph.json")


def _read_manifest() -> dict | None:
    if not MANIFEST_PATH.exists():
        return None
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read ingestion manifest: {e}")
        return None


def _write_manifest(repo_names: list[str], chunk_counts: dict[str, int]) -> None:
    manifest = {"repo_names": sorted(repo_names), "chunk_counts": chunk_counts,
                "total_chunks": sum(chunk_counts.values())}
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote ingestion manifest: {manifest['total_chunks']} chunks across {len(repo_names)} repos")


def load_or_ingest_corpus(
    repo_names: list[str],
    repo_roots: list[Path],
    force: bool = False,
) -> tuple[list[CodeChunk], BM25KeywordIndex, VectorStoreManager, "nx.DiGraph"]:
    """
    Ensure the given repos are chunked, embedded, indexed, and graphed
    exactly ONCE, then reused across every subsequent call — rather than
    re-chunking and re-embedding the entire corpus on every process run.

    This matters for two real reasons, not just speed: (1) re-embedding
    the same text via sentence-transformers in separate forward passes
    is NOT perfectly numerically reproducible (floating-point operation
    ordering in batched neural net inference can vary run to run), which
    was empirically confirmed to shift which near-tied chunks land in a
    query's top-k — a retrieval-quality bug masquerading as instability.
    (2) unconditional re-ingestion cannot scale to a deployed product
    where users query a corpus repeatedly without expecting a multi-
    minute re-embed on every request.

    Ingestion state is tracked via a manifest file recording which repos
    and how many chunks were ingested; if the manifest matches the
    requested repos AND the vector store's actual stored count agrees
    with it, everything is loaded from disk (BM25 chunks from Day 7's
    persistence, Chroma reopened without upserting, graph reloaded).
    Any mismatch — new repos requested, missing manifest, or a vector
    store count that disagrees with what the manifest claims — triggers
    a full re-ingestion, since a stale/partial disk state is worse than
    a slower but correct one.
    """
    manifest = _read_manifest()
    vector_store = VectorStoreManager()

    can_reuse = (
        not force
        and manifest is not None
        and manifest["repo_names"] == sorted(repo_names)
        and BM25_CHUNKS_PATH.exists()
        and GRAPH_PATH.exists()
        and vector_store.code_collection.count() == manifest["total_chunks"]
    )

    if can_reuse:
        logger.info("Ingestion manifest matches — loading existing corpus from disk, skipping re-embed")
        bm25_index = BM25KeywordIndex.load(str(BM25_CHUNKS_PATH))
        all_chunks = bm25_index.chunks
        graph = load_graph(GRAPH_PATH)
        return all_chunks, bm25_index, vector_store, graph

    logger.info("No valid cached ingestion found — running full ingestion (chunk + embed + index + graph)")

    all_chunks: list[CodeChunk] = []
    chunk_counts: dict[str, int] = {}
    for root, name in zip(repo_roots, repo_names):
        repo_chunks = chunk_repo(root, repo_name=name)
        all_chunks.extend(repo_chunks)
        chunk_counts[name] = len(repo_chunks)

    vector_store.upsert_chunks(all_chunks)

    bm25_index = BM25KeywordIndex(all_chunks)
    bm25_index.save_chunks(str(BM25_CHUNKS_PATH))

    graph = build_combined_dependency_graph(repo_roots)
    save_graph(graph, GRAPH_PATH)

    _write_manifest(repo_names, chunk_counts)

    return all_chunks, bm25_index, vector_store, graph