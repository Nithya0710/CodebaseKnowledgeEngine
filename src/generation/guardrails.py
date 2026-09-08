import re
from pydantic import BaseModel
from src.retrieval.vector_store import VectorStoreManager
from src.chunking.ast_parser import CodeChunk
from src.retrieval.rerank import RAW_SCORE_GUARDRAIL_THRESHOLD
from src.logging_config import get_logger
from src.logging_config import setup_logging

logger = get_logger(__name__)


class InsufficientContextResponse(BaseModel):
    """
    Returned instead of a generated answer when no retrieved chunk
    is confidently relevant enough to the question. Prevents the LLM
    from generating a fluent, confident-sounding answer on
    out-of-corpus questions — this is a code-level guarantee, not a
    request to the model to 'not make things up.'
    """
    question: str
    message: str
    best_score_achieved: float
    threshold: float


def check_context_sufficiency(
    question: str,
    ranked_chunks: list[tuple[CodeChunk, float, float]],
    threshold: float = RAW_SCORE_GUARDRAIL_THRESHOLD,
) -> InsufficientContextResponse | None:
    """Pre-generation guardrail: if the BEST (maximum) RAW cross-encoder
    score across all final retrieved chunks is below `threshold`, return
    an InsufficientContextResponse instead of allowing generation to
    proceed. Returns None if context is sufficient.

    Uses the RAW cross-encoder score (not the blended rerank score)
    because the blended score is normalized RELATIVE TO ITS OWN BATCH —
    something always scores near its own max regardless of whether any
    candidate in that batch is actually relevant, which made an earlier
    version of this guardrail fail to block an out-of-scope question
    (a weather query scored 0.72 on the blended scale despite every
    candidate being irrelevant). The raw score is comparable ACROSS
    queries, which is what an absolute "is there real signal here"
    check actually requires."""
    if not ranked_chunks:
        logger.warning(f"No chunks retrieved at all for question: {question!r}")
        return InsufficientContextResponse(
            question=question,
            message="No relevant context was found in the corpus for this question.",
            best_score_achieved=0.0,
            threshold=threshold,
        )

    best_raw_score = max(raw_score for _, _, raw_score in ranked_chunks)

    if best_raw_score < threshold:
        logger.warning(
            f"Context insufficient for question: {question!r} "
            f"(best_raw_score={best_raw_score:.4f}, threshold={threshold})"
        )
        return InsufficientContextResponse(
            question=question,
            message="No sufficiently relevant context was found in the corpus to answer this question confidently.",
            best_score_achieved=best_raw_score,
            threshold=threshold,
        )

    logger.info(
        f"Context sufficient for question: {question!r} (best_raw_score={best_raw_score:.4f})"
    )
    return None


CITATION_PATTERN = re.compile(r"\[(?:cite|chunk_id):\s*([^\]]+)\]")


def validate_citations(
    answer: str,
    ranked_chunks: list[tuple[CodeChunk, float, float]],
) -> str:
    """
    Post-generation guardrail: parse every citation tag out of the
    generated answer — accepting both [cite: chunk_id] (the instructed
    format) and [chunk_id: chunk_id] (a format-drift variant observed
    in real generations, where the model echoed the label used in the
    prompt's context blocks instead of the instructed citation wrapper)
    — verify each cited chunk_id actually exists in the retrieved
    context set, and strip any citation that doesn't (a hallucinated
    citation), logging each incident.

    This catches a DIFFERENT failure mode than the pre-generation
    guardrail: even when there IS enough good context overall (the
    pre-generation check passed), the model can still fabricate a
    citation to a chunk_id that sounds plausible but was never actually
    retrieved. Only the citation tag itself is removed, not the
    surrounding sentence, since the underlying factual claim may still
    be accurate even if the model got the specific citation wrong.

    After stripping, collapses any resulting double-space and removes a
    stray space left before trailing punctuation (e.g. "internally ."
    becomes "internally."), so a stripped citation doesn't leave visible
    formatting artifacts in the final answer.
    """
    valid_ids = {VectorStoreManager._chunk_id(chunk) for chunk, _, _ in ranked_chunks}

    def replace_if_invalid(match: re.Match) -> str:
        cited_id = match.group(1).strip()
        if cited_id in valid_ids:
            return match.group(0)  # leave valid citations untouched

        logger.warning(
            f"Hallucinated citation detected and stripped: {cited_id!r} "
            f"was not in the retrieved context set"
        )
        return ""

    cleaned = CITATION_PATTERN.sub(replace_if_invalid, answer)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)          # collapse double+ spaces
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)  # remove space before punctuation
    return cleaned


if __name__ == '__main__':
    setup_logging()
    from src.chunking.ast_parser import CodeChunk

    # testing context sufficiency guardrail
    # strong_chunks = [
    #     (CodeChunk(repo_name="flask", file_path="a.py", chunk_type="function",
    #                 name="foo", start_line=1, end_line=2, code_text="def foo(): pass", docstring=None), 0.91),
    # ]
    # weak_chunks = [
    #     (CodeChunk(repo_name="flask", file_path="a.py", chunk_type="function",
    #                 name="foo", start_line=1, end_line=2, code_text="def foo(): pass", docstring=None), 0.42),
    # ]

    # print("Strong context:", check_context_sufficiency("real question", strong_chunks))
    # print("Weak context:", check_context_sufficiency("what's the weather today", weak_chunks))
    # print("No chunks:", check_context_sufficiency("totally unrelated", []))

    # testing citation validation guardrail
    # ranked_chunks = [
    #     (CodeChunk(repo_name="flask", file_path="flask/app.py", chunk_type="function",
    #                 name="add_url_rule", start_line=10, end_line=20,
    #                 code_text="def add_url_rule(...): ...", docstring=None), 0.91),
    # ]

    # fake_answer = (
    #     "The add_url_rule function registers a route "
    #     "[cite: flask:flask/app.py:add_url_rule:10]. "
    #     "It also handles authentication internally "
    #     "[cite: flask:flask/security.py:check_auth:5]."
    # )

    # cleaned = validate_citations(fake_answer, ranked_chunks)
    # print("BEFORE:", fake_answer)
    # print("AFTER: ", cleaned)

    # testing guardrails with the 5 baseline questions and one extra question which is out of context
    from pathlib import Path
    import json
    from src.retrieval.ingest import load_or_ingest_corpus
    from src.retrieval.hybrid import hybrid_retrieve
    from src.retrieval.augment import augment_with_related_files
    from src.retrieval.rerank import rerank
    from src.generation.generator import generate
    from src.generation.guardrails import check_context_sufficiency, validate_citations

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
        "What's the weather like in Bangalore today?",  # deliberately out-of-scope
    ]

    transcripts = []

    for question in questions:
        print(f"\n{'='*80}\nQUESTION: {question}\n{'='*80}")

        fused = hybrid_retrieve(question, bm25_index, vector_store, top_k_per_source=15, top_k_final=15)
        augmented = augment_with_related_files(question, fused, graph, all_chunks, depth=2)
        top5 = rerank(question, augmented, keep_top=5)

        guardrail_result = check_context_sufficiency(question, top5)

        if guardrail_result is not None:
            print(f"BLOCKED by pre-generation guardrail: {guardrail_result.message}")
            print(f"  best_score_achieved={guardrail_result.best_score_achieved:.4f}")
            transcripts.append({
                "question": question,
                "blocked": True,
                "guardrail_response": guardrail_result.model_dump(),
                "retrieved_context": [],
                "answer": None,
            })
            continue

        raw_answer = generate(question, top5)
        validated_answer = validate_citations(raw_answer, top5)

        print(f"ANSWER:\n{validated_answer}")

        transcripts.append({
            "question": question,
            "blocked": False,
            "guardrail_response": None,
            "retrieved_context": [
                {"chunk_id": VectorStoreManager._chunk_id(chunk), "name": chunk.name,
                 "file_path": chunk.file_path, "start_line": chunk.start_line,
                 "end_line": chunk.end_line, "code_text": chunk.code_text,
                 "blended_score": float(blended_score), "raw_ce_score": float(raw_score)}
                for chunk, blended_score, raw_score in top5
            ],
            "answer": validated_answer,
        })

    output_path = Path("docs/day11_transcripts.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(transcripts, f, indent=2)
    print(f"\nSaved {len(transcripts)} transcripts to {output_path}")