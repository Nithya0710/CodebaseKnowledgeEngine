from sentence_transformers import CrossEncoder
from src.chunking.ast_parser import CodeChunk
from src.logging_config import get_logger
from src.logging_config import setup_logging

logger = get_logger(__name__)

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Loaded once at module import time, not per-query. Loading a transformer
# model from disk is expensive (multiple seconds), so this cost is paid
# exactly once per process, and every call to rerank() reuses this same
# in-memory model instance.
_cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
logger.info(f"Loaded cross-encoder: {CROSS_ENCODER_MODEL}")


def rerank(
    question: str,
    candidates: list[CodeChunk],
    keep_top: int = 5,
    alpha: float = 0.65,
) -> list[tuple[CodeChunk, float]]:
    """
    Score every candidate chunk jointly against the question using the
    cross-encoder, then BLEND that score with the chunk's incoming rank
    (its position after hybrid retrieval + graph augmentation) rather
    than sorting purely by cross-encoder score.

    RATIONALE FOR BLENDING (not a pure cross-encoder override):
    Days 6-9 built substantial multi-signal agreement — a chunk ranking
    highly after RRF fusion (Day 8) reflects agreement across THREE
    independent retrieval channels (BM25, dense code, dense doc), and
    graph augmentation (Day 9) adds structurally-verified relevance on
    top of that. That's strong, hard-won evidence. A single
    general-purpose cross-encoder (trained on web search relevance, not
    code) can have real blind spots — e.g. over-weighting a literal
    keyword match ("middleware") over deeper semantic relevance. Fully
    overriding strong multi-signal consensus with one model's opinion
    risks discarding that evidence based on a surface-level quirk. This
    blend requires the cross-encoder to show a MEANINGFUL preference,
    not just any preference, to overturn a chunk's prior-stage standing.

    alpha controls how much weight the cross-encoder gets vs. prior rank
    (default 0.65 — cross-encoder is still the majority signal, since
    it IS a more accurate model in principle, but prior consensus
    remains a real counterweight rather than being discarded entirely).

    Prior rank is normalized via 1/(1+rank) (0-indexed), giving a smooth
    score in (0, 1] that preserves spacing near the top of the list.
    Cross-encoder scores are min-max normalized to [0, 1] WITHIN this
    batch, since raw logits are unbounded and their range shifts
    per-query — there's no fixed, universal scale to normalize against.
    """
    if not candidates:
        logger.warning("rerank called with zero candidates")
        return []

    pairs = [(question, chunk.code_text) for chunk in candidates]

    try:
        raw_scores = _cross_encoder.predict(pairs)
    except Exception as e:
        logger.error(f"Cross-encoder scoring failed: {e}")
        return []

    min_score, max_score = min(raw_scores), max(raw_scores)
    score_range = max_score - min_score

    def normalize_ce_score(score: float) -> float:
        if score_range == 0:
            return 1.0  # all candidates scored identically — no info to differentiate
        return (score - min_score) / score_range

    def normalize_prior_rank(rank: int) -> float:
        return 1.0 / (1 + rank)

    blended = []
    for prior_rank, (chunk, raw_score) in enumerate(zip(candidates, raw_scores)):
        ce_norm = normalize_ce_score(raw_score)
        prior_norm = normalize_prior_rank(prior_rank)
        final_score = alpha * ce_norm + (1 - alpha) * prior_norm
        blended.append((chunk, final_score, raw_score, prior_rank))

    blended.sort(key=lambda item: item[1], reverse=True)
    top = blended[:keep_top]

    for new_rank, (chunk, final_score, raw_score, prior_rank) in enumerate(top):
        delta = prior_rank - new_rank
        logger.info(
            f"  rank delta: {chunk.name} — was #{prior_rank}, now #{new_rank} "
            f"(delta={delta:+d}, blended={final_score:.4f}, raw_ce={raw_score:.4f})"
        )

    logger.info(
        f"Reranked {len(candidates)} candidates -> top-{len(top)} for: {question!r} "
        f"(alpha={alpha})"
    )
    return [(chunk, final_score) for chunk, final_score, _, _ in top]


if __name__ == '__main__':
    setup_logging()
    # testing that the cross-encoder loads correctly
    # print(f"Cross-encoder loaded: {_cross_encoder}")

    # # Quick manual scoring sanity check, ahead of piece 2's real rerank() function
    # pairs = [
    #     ("How do I hash a password?", "def hash_password(pw): return hashlib.sha256(pw.encode()).hexdigest()"),
    #     ("How do I hash a password?", "def render_template(name): return template_env.get_template(name)"),
    # ]
    # scores = _cross_encoder.predict(pairs)
    # for (query, doc), score in zip(pairs, scores):
    #     print(f"score={score:.4f}  doc={doc[:50]}...")

    # testing rerank() with a few candidates
    # candidates = [
    #     CodeChunk(
    #         repo_name="test", file_path="auth.py", chunk_type="function",
    #         name="hash_password", start_line=1, end_line=2,
    #         code_text="def hash_password(pw): return hashlib.sha256(pw.encode()).hexdigest()",
    #         docstring=None,
    #     ),
    #     CodeChunk(
    #         repo_name="test", file_path="templating.py", chunk_type="function",
    #         name="render_template", start_line=1, end_line=2,
    #         code_text="def render_template(name): return template_env.get_template(name)",
    #         docstring=None,
    #     ),
    #     CodeChunk(
    #         repo_name="test", file_path="auth.py", chunk_type="function",
    #         name="verify_password", start_line=4, end_line=5,
    #         code_text="def verify_password(pw, hash): return hash_password(pw) == hash",
    #         docstring=None,
    #     ),
    # ]

    # results = rerank("How do I hash a password?", candidates, keep_top=3)
    # for chunk, score in results:
    #     print(f"score={score:.4f}  {chunk.name}")

    # testing rerank() with the 5 baseline questions
    from pathlib import Path
    from src.retrieval.ingest import load_or_ingest_corpus
    from src.retrieval.hybrid import hybrid_retrieve
    from src.retrieval.augment import augment_with_related_files
    # from src.chunking.dependency_graph import save_graph, load_graph

    repo_names = ["flask", "click", "jinja", "itsdangerous", "markupsafe", "werkzeug", "requests", "httpx"]
    repo_roots = [Path(f"data/repos/{name}") for name in repo_names]

    all_chunks, bm25_index, vector_store, graph = load_or_ingest_corpus(repo_names, repo_roots)
    print(f"Corpus ready: {len(all_chunks)} chunks")

    questions = [
        "Where is authentication middleware defined and what does it check?",
        "How does the rate limiter work and which files use it?",
        "What happens when a payment fails? Trace the full code path.",
        "Find all places add_url_rule is called and explain the contexts.",
        "How does Flask handle routing for a URL?",
    ]

    final_results = {}

    final_results = {}
    for question in questions:
        print(f"\n{'='*80}\nQUESTION: {question}\n{'='*80}")
        fused = hybrid_retrieve(question, bm25_index, vector_store, top_k_per_source=15, top_k_final=15)
        augmented = augment_with_related_files(question, fused, graph, all_chunks, depth=2)
        top5 = rerank(question, augmented, keep_top=5)
        final_results[question] = top5
        print("FINAL TOP-5 (after hybrid -> augment -> rerank):")
        for chunk, score in top5:
            print(f"  score={score:.4f}  {chunk.name}  ({chunk.file_path}:{chunk.start_line})")

    import json
    output_path = Path("eval/day10_final_top5_per_question.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        q: [
            {"name": c.name, "file_path": c.file_path, "start_line": c.start_line,
             "end_line": c.end_line, "code_text": c.code_text, "score": float(s)}
            for c, s in results
        ]
        for q, results in final_results.items()
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved final top-5 per question to {output_path}")