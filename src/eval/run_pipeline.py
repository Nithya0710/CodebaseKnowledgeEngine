from pathlib import Path
import json
from src.retrieval.ingest import load_or_ingest_corpus
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.augment import augment_with_related_files
from src.retrieval.rerank import rerank
from src.generation.generator import generate
from src.generation.guardrails import check_context_sufficiency, validate_citations_with_counts
from src.retrieval.vector_store import VectorStoreManager
from src.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def run_pipeline_on_test_set(test_set_path: str, repo_names: list[str], repo_roots: list[Path]) -> list[dict]:
    """Run the full pipeline (hybrid retrieve -> augment -> rerank ->
    generate, guardrails included) against every question in the gold
    standard test set, producing RAGAS-ready records: question, answer,
    contexts (retrieved chunk texts), and ground_truth.

    Guardrail-blocked questions still produce a record — the guardrail's
    own message becomes the 'answer', with an empty contexts list, since
    RAGAS evaluation should be able to see how the pipeline behaves on
    out-of-scope questions, not just skip them."""
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    all_chunks, bm25_index, vector_store, graph = load_or_ingest_corpus(repo_names, repo_roots)
    logger.info(f"Corpus ready: {len(all_chunks)} chunks")

    records = []
    for entry in test_set:
        question = entry["question"]
        logger.info(f"Running pipeline for: {question!r}")

        fused = hybrid_retrieve(question, bm25_index, vector_store, top_k_per_source=15, top_k_final=15)
        augmented = augment_with_related_files(question, fused, graph, all_chunks, depth=2)
        top5 = rerank(question, augmented, keep_top=5)

        guardrail_result = check_context_sufficiency(question, top5)

        if guardrail_result is not None:
            records.append({
                "question": question,
                "answer": guardrail_result.message,
                "contexts": [],
                "ground_truth": entry["ground_truth_answer"],
                "ground_truth_context_chunk_ids": entry["ground_truth_context_chunk_ids"],
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
            "question": question,
            "answer": validation.answer,
            "contexts": [chunk.code_text for chunk, _, _ in top5],
            "ground_truth": entry["ground_truth_answer"],
            "ground_truth_context_chunk_ids": entry["ground_truth_context_chunk_ids"],
            "retrieved_chunk_ids": [VectorStoreManager._chunk_id(chunk) for chunk, _, _ in top5],
            "blocked": False,
            "raw_answer": raw_answer,
            "n_valid_citations": validation.n_valid_citations,
            "n_stripped_citations": validation.n_stripped_citations,
        })

    return records


if __name__ == '__main__':
    setup_logging()

    repo_names = ["flask", "click", "jinja", "itsdangerous", "markupsafe", "werkzeug", "requests", "httpx"]
    repo_roots = [Path(f"data/repos/{name}") for name in repo_names]

    records = run_pipeline_on_test_set("eval/test_set.json", repo_names, repo_roots)

    output_path = Path("eval/pipeline_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nRan pipeline on {len(records)} questions, saved to {output_path}")
    for r in records:
        status = "BLOCKED" if r["blocked"] else "ANSWERED"
        print(f"  [{status}] valid={r['n_valid_citations']} stripped={r['n_stripped_citations']} {r['question'][:60]}...")
    answered = [r for r in records if not r["blocked"]]
    print(f"Hallucinated citations stripped: {sum(r['n_stripped_citations'] for r in records)} total; "
          f"answers with zero valid citations: {sum(1 for r in answered if r['n_valid_citations'] == 0)}/{len(answered)}")