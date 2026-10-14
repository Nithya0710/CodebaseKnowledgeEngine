import argparse
import json
from pathlib import Path

from src.config import settings
from src.eval.gold_set import DEFAULT_TEST_SET_PATH, SUBSETS, load_gold_set
from src.generation.generator import generate
from src.generation.guardrails import check_context_sufficiency, validate_citations_with_counts
from src.logging_config import get_logger, setup_logging
from src.retrieval.augment import augment_with_related_files
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.ingest import load_or_ingest_corpus
from src.retrieval.rerank import rerank
from src.retrieval.vector_store import VectorStoreManager

logger = get_logger(__name__)

REPO_NAMES = ["flask", "click", "jinja", "itsdangerous", "markupsafe", "werkzeug", "requests", "httpx"]


def run_pipeline_on_test_set(
    test_set_path: Path,
    repo_names: list[str],
    repo_roots: list[Path],
    subset: str | None = None,
) -> list[dict]:
    """Run the full pipeline (hybrid retrieve -> augment -> rerank ->
    generate, guardrails included) against every question in the gold
    standard test set (or a named subset), producing RAGAS-ready records:
    question, answer, contexts (retrieved chunk texts), and ground_truth.

    Guardrail-blocked questions still produce a record (blocked=True, the
    guardrail's message as the answer, empty contexts) so evaluation can
    report guardrail behavior; run_ragas excludes them from the metrics.

    Each answered record also keeps the raw pre-validation answer and the
    counts of valid and stripped citations, so hallucinated citations are
    measurable from the saved file. best_raw_ce_score (the value the
    pre-generation guardrail checks) is saved for threshold analysis."""
    entries = load_gold_set(test_set_path, subset)

    all_chunks, bm25_index, vector_store, graph = load_or_ingest_corpus(repo_names, repo_roots)
    logger.info(f"Corpus ready: {len(all_chunks)} chunks; running {len(entries)} questions")

    records = []
    for entry in entries:
        question = entry.question
        logger.info(f"Running pipeline for: {question!r}")

        fused = hybrid_retrieve(question, bm25_index, vector_store, top_k_per_source=15, top_k_final=15)
        augmented = augment_with_related_files(question, fused, graph, all_chunks, depth=2)
        top5 = rerank(question, augmented, keep_top=5)
        best_raw = max((float(raw) for _, _, raw in top5), default=None)

        base = {
            "question": question,
            "ground_truth": entry.ground_truth_answer,
            "ground_truth_context_chunk_ids": entry.ground_truth_context_chunk_ids,
            "generation_model": settings.ollama_model,
            "best_raw_ce_score": best_raw,
        }

        guardrail_result = check_context_sufficiency(question, top5)
        if guardrail_result is not None:
            records.append({
                **base,
                "answer": guardrail_result.message,
                "contexts": [],
                "retrieved_chunk_ids": [],
                "blocked": True,
                "raw_answer": None,
                "n_valid_citations": 0,
                "n_stripped_citations": 0,
            })
            continue

        raw_answer = generate(question, top5)
        validation = validate_citations_with_counts(raw_answer, top5)

        records.append({
            **base,
            "answer": validation.answer,
            "contexts": [chunk.code_text for chunk, _, _ in top5],
            "retrieved_chunk_ids": [VectorStoreManager._chunk_id(chunk) for chunk, _, _ in top5],
            "blocked": False,
            "raw_answer": raw_answer,
            "n_valid_citations": validation.n_valid_citations,
            "n_stripped_citations": validation.n_stripped_citations,
        })

    return records


def main() -> None:
    """CLI entrypoint: run the pipeline and save records to JSON."""
    parser = argparse.ArgumentParser(description="Run the RAG pipeline over the gold test set.")
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET_PATH)
    parser.add_argument("--subset", choices=sorted(SUBSETS), default=None)
    parser.add_argument("--out", type=Path, default=Path("eval/pipeline_results.json"))
    args = parser.parse_args()

    setup_logging()
    repo_roots = [Path(f"data/repos/{name}") for name in REPO_NAMES]
    records = run_pipeline_on_test_set(args.test_set, REPO_NAMES, repo_roots, args.subset)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nRan pipeline on {len(records)} questions with {settings.ollama_model}, saved to {args.out}")
    for r in records:
        status = "BLOCKED" if r["blocked"] else "ANSWERED"
        print(f"  [{status}] raw_ce={r['best_raw_ce_score']} valid={r['n_valid_citations']} "
              f"stripped={r['n_stripped_citations']} {r['question'][:60]}")
    answered = [r for r in records if not r["blocked"]]
    print(f"Hallucinated citations stripped: {sum(r['n_stripped_citations'] for r in records)} total; "
          f"answers with zero valid citations: {sum(1 for r in answered if r['n_valid_citations'] == 0)}/{len(answered)}")


if __name__ == "__main__":
    main()
