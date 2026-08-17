from pathlib import Path
from collections import Counter
from src.logging_config import get_logger
from src.retrieval.keyword_index import BM25KeywordIndex
from src.retrieval.vector_store import VectorStoreManager
from src.chunking.ast_parser import CodeChunk
from src.chunking.ast_parser import chunk_repo
from src.retrieval.naive_baseline import NaiveBaselineStore

logger = get_logger(__name__)


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """
    Fuse multiple ranked lists of document IDs into a single fused
    score per document, using Reciprocal Rank Fusion (RRF).

    Formula: score(doc) = sum over all lists containing doc of 1/(k + rank),
    where rank is the doc's 1-indexed position within that specific list.
    A doc missing from a given list simply contributes nothing from that
    list — there's no explicit penalty term, absence just means fewer
    summands.

    RRF is deliberately RANK-based, not SCORE-based. Each input list here
    will come from a different retrieval method (BM25 keyword search,
    dense cosine similarity over code, dense cosine similarity over
    docstrings), and their raw scores live on completely incompatible
    numeric scales: BM25 scores are unbounded TF-IDF-derived numbers
    (e.g. 12.50), while cosine similarities/distances are roughly bounded
    0-1 (or 0-2 for distance). Averaging or summing those raw scores
    directly would let whichever retriever happens to produce larger
    numbers dominate the fused ranking, regardless of actual relevance.
    By using only rank POSITION (1st place, 2nd place, ...) instead of
    the raw score that produced that position, RRF sidesteps the
    incompatible-scales problem entirely — "ranked 1st" means the same
    thing no matter which retrieval method produced that ranking.

    The constant k (standard default: 60, from the original RRF paper)
    is a smoothing term: it dampens the score gap between adjacent ranks
    (e.g. 1/(60+1) vs 1/(60+2) is a much smaller relative gap than
    1/1 vs 1/2 would be), which prevents any single list's #1 pick from
    completely dominating the fused ranking on its own.
    """
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    logger.info(
        f"RRF fused {len(ranked_lists)} ranked lists into {len(scores)} unique documents"
    )
    return scores


def hybrid_retrieve(
    question: str,
    bm25_index: BM25KeywordIndex,
    vector_store: VectorStoreManager,
    top_k_per_source: int = 15,
    top_k_final: int = 15,
) -> list[CodeChunk]:
    """
    Run three independent retrieval channels — BM25 keyword search,
    dense code embedding search, and dense doc embedding search — and
    fuse their ranked results into a single list via Reciprocal Rank
    Fusion.

    Each channel returns its own top-`top_k_per_source` results
    independently; RRF then combines all three ranked-ID lists into
    one fused score per unique chunk ID, so a chunk that ranks well in
    ANY channel is rewarded, and one that ranks well across MULTIPLE
    channels is rewarded more (RRF scores sum across lists). Returns
    the top-`top_k_final` chunks by fused score, already de-duplicated
    since RRF's output is naturally one score per unique ID.
    """
    bm25_results = bm25_index.query(question, top_k=top_k_per_source)
    bm25_ids = [VectorStoreManager._chunk_id(chunk) for chunk, _ in bm25_results]

    code_results = vector_store.query_code(question, top_k=top_k_per_source)
    code_ids = [r["id"] for r in code_results]

    doc_results = vector_store.query_docs(question, top_k=top_k_per_source)
    doc_ids = [r["id"].removesuffix("_doc") for r in doc_results]

    fused_scores = reciprocal_rank_fusion([bm25_ids, code_ids, doc_ids])

    # Build an ID -> CodeChunk lookup so we can return full objects, not just IDs.
    # BM25's chunks are already full CodeChunk objects — prefer that source.
    id_to_chunk: dict[str, CodeChunk] = {
        chunk_id: chunk for chunk_id, (chunk, _) in zip(bm25_ids, bm25_results)
    }
    # Fallback: reconstruct from Chroma metadata for any ID BM25 didn't have
    # (shouldn't normally happen if both indexes are built from the same
    # corpus, but defensive against partial/stale indexes).
    for r in code_results + doc_results:
        chunk_id = r["id"].removesuffix("_doc")
        if chunk_id not in id_to_chunk:
            meta = r["metadata"]
            id_to_chunk[chunk_id] = CodeChunk(
                repo_name=meta["repo_name"],
                file_path=meta["file_path"],
                chunk_type=meta["chunk_type"],
                name=meta["name"],
                start_line=meta["start_line"],
                end_line=meta["end_line"],
                code_text=r["document"],
                docstring=None,
            )

    ranked_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)
    top_ids = ranked_ids[:top_k_final]

    results = []
    for cid in top_ids:
        chunk = id_to_chunk.get(cid)
        if chunk is None:
            logger.warning(f"Fused ID {cid} has no resolvable CodeChunk — skipping")
            continue
        results.append(chunk)

    logger.info(
        f"hybrid_retrieve fused {len(bm25_ids)} BM25 + {len(code_ids)} code + "
        f"{len(doc_ids)} doc results into top-{len(results)} chunks for: {question!r}"
    )
    return results


def compare_naive_vs_hybrid(
    naive_store: "NaiveBaselineStore",
    bm25_index: BM25KeywordIndex,
    vector_store: VectorStoreManager,
) -> None:
    """
    Run Day 5's 5 baseline questions through both the naive
    single-collection baseline and today's fused hybrid retrieval,
    and print a side-by-side comparison of which chunks each approach
    surfaced. This is retrieval-only — no generation happens here,
    since hybrid_retrieve doesn't call an LLM. The point is to see
    concretely how the candidate set changes with fusion, before
    Day 9 (graph augmentation), Day 10 (reranking), and Day 11
    (generation) build on top of it.
    """
    questions = [
        "Where is authentication middleware defined and what does it check?",
        "How does the rate limiter work and which files use it?",
        "What happens when a payment fails? Trace the full code path.",
        "Find all places add_url_rule is called and explain the contexts.",
        "How does Flask handle routing for a URL?",
    ]

    for question in questions:
        print(f"\n{'='*80}\nQUESTION: {question}\n{'='*80}")

        naive_results = naive_store.collection.query(query_texts=[question], n_results=5)
        print("\n--- NAIVE (cosine-only, top-5) ---")
        for i in range(len(naive_results["ids"][0])):
            meta = naive_results["metadatas"][0][i]
            print(f"  {meta['name']}  ({meta['file_path']}:{meta['start_line']})")

        hybrid_results = hybrid_retrieve(
            question, bm25_index, vector_store, top_k_per_source=15, top_k_final=15
        )
        print(f"\n--- HYBRID (BM25 + dense code + dense doc, fused, top-{len(hybrid_results)}) ---")
        for chunk in hybrid_results:
            print(f"  {chunk.name}  ({chunk.file_path}:{chunk.start_line})")


if __name__=='__main__':
    # testing reciprocal_rank_fusion()
    # list_a = ["chunk1", "chunk2", "chunk3"]
    # list_b = ["chunk2", "chunk1", "chunk4"]

    # result = reciprocal_rank_fusion([list_a, list_b], k=60)
    # for doc_id, score in sorted(result.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{doc_id}: {score:.6f}")

    # testing hybrid_retrieve()
    # chunks = chunk_repo(Path("data/repos/flask"), repo_name="flask")

    # bm25_index = BM25KeywordIndex(chunks)
    vector_store = VectorStoreManager()
    # vector_store.upsert_chunks(chunks)  # skip if flask is already upserted from Day 6

    # results = hybrid_retrieve("add_url_rule", bm25_index, vector_store, top_k_per_source=15, top_k_final=15)
    # for chunk in results:
    #     print(f"{chunk.name}  ({chunk.file_path}:{chunk.start_line})")

    # comparing naive vs hybrid retrieval for the 5 Day 5 questions
    repo_names = ["flask", "click", "jinja", "itsdangerous", "markupsafe", "werkzeug", "requests", "httpx"]
    all_chunks = []
    for repo_name in repo_names:
        all_chunks.extend(chunk_repo(Path(f"data/repos/{repo_name}"), repo_name=repo_name))
    vector_store.upsert_chunks(all_chunks)
    bm25_index = BM25KeywordIndex(all_chunks)
    naive_store = NaiveBaselineStore()
    compare_naive_vs_hybrid(naive_store, bm25_index, vector_store)

    # testing whether the corpus is balanced or not
    # all_code = vector_store.code_collection.get()
    # repo_counts = Counter(m["repo_name"] for m in all_code["metadatas"])
    # print("code_chunks by repo:")
    # for repo, count in repo_counts.most_common():
    #     print(f"  {repo}: {count}")

