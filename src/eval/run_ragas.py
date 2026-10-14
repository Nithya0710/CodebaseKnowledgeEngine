"""RAGAS evaluation of pipeline records with an injectable, local-first judge.

Usage (from the repo root):
    python -m src.eval.run_ragas --smoke
    python -m src.eval.run_ragas --records eval/pipeline_results_baseline5.json \
        --out eval/ragas_baseline5.csv --decision-rule
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from src.config import settings
from src.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

METRIC_COLUMNS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
# ragas column name -> our column name
_RAGAS_RENAMES = {"llm_context_precision_with_reference": "context_precision"}
# Metrics whose score depends on the answer text; the judge smoke test and the
# decision rule compare these. Context metrics depend only on retrieval.
ANSWER_METRICS = ["faithfulness", "answer_relevancy"]

BM25_CHUNKS_PATH = Path("data/bm25_chunks.json")

# One model on one laptop: parallel judge calls just queue inside Ollama and
# then hit the timeout, so calls run one at a time with a generous timeout.
JUDGE_RUN_CONFIG = dict(max_workers=1, timeout=600, max_retries=3, seed=42)

NAN_RATE_LIMIT = 0.25
DECISION_FAITHFULNESS_MIN = 0.70
DECISION_RELEVANCY_MIN = 0.70
DECISION_MAX_STRIPPED = 1  # >= 2 stripped citations across the run triggers a switch


class PipelineRecord(BaseModel):
    """One record written by src.eval.run_pipeline."""
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    ground_truth_context_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    blocked: bool
    n_valid_citations: int = 0
    n_stripped_citations: int = 0
    best_raw_ce_score: float | None = None
    generation_model: str | None = None


def id_retrieval_metrics(retrieved_ids: list[str], gold_ids: list[str]) -> tuple[float | None, float | None]:
    """Exact, judge-free retrieval scores from chunk IDs.

    precision = share of retrieved chunks that are gold (None if nothing retrieved);
    recall = share of gold chunks that were retrieved (None if the gold list is empty,
    e.g. questions with no answer in the corpus, where recall is undefined).
    """
    retrieved = list(dict.fromkeys(retrieved_ids))
    gold = set(gold_ids)
    hits = sum(1 for cid in retrieved if cid in gold)
    precision = hits / len(retrieved) if retrieved else None
    recall = hits / len(gold) if gold else None
    return precision, recall


def records_to_dataset(records: list[PipelineRecord]) -> tuple[EvaluationDataset, list[int]]:
    """Convert answered records into a RAGAS dataset.

    Guardrail-blocked records are excluded: their 'answer' is a refusal
    message, and scoring it for faithfulness or relevancy as if it were an
    answer would be meaningless. Returns the dataset and the indexes of the
    records it contains, so scores can be mapped back.
    """
    samples, included = [], []
    for i, r in enumerate(records):
        if r.blocked:
            continue
        samples.append(SingleTurnSample(
            user_input=r.question,
            retrieved_contexts=r.contexts,
            response=r.answer,
            reference=r.ground_truth,
        ))
        included.append(i)
    return EvaluationDataset(samples=samples), included


def build_judge() -> tuple[Any, Any, str]:
    """Build the judge LLM and embeddings from settings.

    Returns (ragas_llm, ragas_embeddings, judge_label). The judge is injected
    into ragas.evaluate explicitly; RAGAS would otherwise default to OpenAI.
    Heavy imports are local so unit tests never load a model.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    if settings.ragas_judge_backend == "ollama":
        from langchain_ollama import ChatOllama
        chat = ChatOllama(
            model=settings.ragas_judge_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_ctx=settings.ollama_num_ctx,
            seed=42,
        )
        label = f"ollama:{settings.ragas_judge_model}"
    elif settings.ragas_judge_backend == "groq":
        if not settings.groq_api_key or not settings.groq_judge_model:
            raise RuntimeError("Groq judge selected but GROQ_API_KEY or GROQ_JUDGE_MODEL is not set in .env")
        try:
            from langchain_groq import ChatGroq
        except ImportError as e:
            raise RuntimeError("Groq judge selected but langchain-groq is not installed") from e
        chat = ChatGroq(model=settings.groq_judge_model, api_key=settings.groq_api_key, temperature=0)
        label = f"groq:{settings.groq_judge_model}"
    else:
        raise RuntimeError(f"Unknown RAGAS_JUDGE_BACKEND {settings.ragas_judge_backend!r}")

    embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{settings.embedding_model}")
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(embeddings), label


def score_records(records: list[PipelineRecord], llm: Any, embeddings: Any, judge_label: str) -> list[dict]:
    """Run the four RAGAS metrics on answered records and return one row per record.

    Blocked records get empty metric cells. Exact ID precision/recall are
    computed for every record that retrieved anything.
    """
    rows = []
    for r in records:
        id_p, id_r = id_retrieval_metrics(r.retrieved_chunk_ids, r.ground_truth_context_chunk_ids)
        rows.append({
            "question": r.question,
            "blocked": r.blocked,
            **{m: None for m in METRIC_COLUMNS},
            "id_precision": id_p,
            "id_recall": id_r,
            "n_gold_chunks": len(r.ground_truth_context_chunk_ids),
            "n_valid_citations": r.n_valid_citations,
            "n_stripped_citations": r.n_stripped_citations,
            "best_raw_ce_score": r.best_raw_ce_score,
            "generation_model": r.generation_model,
            "judge": judge_label,
        })

    dataset, included = records_to_dataset(records)
    if not included:
        logger.warning("Every record was blocked; no RAGAS metrics to compute")
        return rows

    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(**JUDGE_RUN_CONFIG),
        show_progress=True,
    )
    scored = result.to_pandas().rename(columns=_RAGAS_RENAMES)
    for pos, idx in enumerate(included):
        for m in METRIC_COLUMNS:
            value = scored.iloc[pos][m]
            rows[idx][m] = None if value is None or math.isnan(value) else float(value)
    return rows


def nan_count(rows: list[dict], metrics: list[str] = METRIC_COLUMNS) -> int:
    """Number of metric cells the judge failed to score on answered rows."""
    return sum(1 for r in rows if not r["blocked"] for m in metrics if r[m] is None)


def metric_means(rows: list[dict], require_gold: bool = False) -> dict[str, float | None]:
    """Mean of each metric over answered rows, ignoring unscored cells.

    With require_gold, rows whose gold chunk list is empty (questions with
    no answer in the corpus) are left out.
    """
    means: dict[str, float | None] = {}
    selected = [r for r in rows if not r["blocked"] and (r["n_gold_chunks"] > 0 or not require_gold)]
    for m in METRIC_COLUMNS + ["id_precision", "id_recall"]:
        values = [r[m] for r in selected if r[m] is not None]
        means[m] = sum(values) / len(values) if values else None
    return means


def judge_usability(good: dict, wrong: dict) -> list[str]:
    """Apply the 'judge is unusable' criteria to the smoke-test rows.

    (a) more than 25% of metric calls on the good and wrong samples unscored;
    (b) identical scores for good and wrong on an answer-dependent metric;
    (c) the wrong answer scores at or above the good one.
    An answer metric that cannot be compared at all (unscored on either
    sample) also fires, since the decision rule depends on exactly those
    metrics. Context metrics are excluded from (b)/(c): both samples share
    the same contexts, so equal context scores are correct, not a failure.
    Returns the list of fired criteria (empty means usable).
    """
    fired = []
    cells = [row[m] for row in (good, wrong) for m in METRIC_COLUMNS]
    nan_rate = sum(1 for v in cells if v is None) / len(cells)
    if nan_rate > NAN_RATE_LIMIT:
        fired.append(f"(a) {nan_rate:.0%} of metric calls unscored (limit {NAN_RATE_LIMIT:.0%})")
    for m in ANSWER_METRICS:
        g, w = good[m], wrong[m]
        if g is None or w is None:
            fired.append(f"(a) {m} not comparable: good={g}, wrong={w}")
        elif g == w:
            fired.append(f"(b) {m} identical for good and wrong ({g:.3f})")
        elif w > g:
            fired.append(f"(c) {m} wrong ({w:.3f}) > good ({g:.3f})")
    return fired


def decision_rule(rows: list[dict]) -> list[str]:
    """Apply the automated part of the locked Ollama-vs-Gemini rule (criteria 1-3).

    Criteria 1-2 average only answered questions that have gold chunks.
    RAGAS multiplies answer_relevancy by 0 for 'noncommittal' answers, so an
    honest "the corpus has no payment logic" would score 0 and could trigger
    a switch for being honest; such questions are covered by criterion 4.

    Criterion 4 (payment-failure answer fabricates a trace) is judged by
    reading the answer and is recorded by hand. Returns the fired criteria;
    any fired criterion means switch generation to Gemini.
    """
    fired = []
    means = metric_means(rows, require_gold=True)
    if means["faithfulness"] is None or means["faithfulness"] < DECISION_FAITHFULNESS_MIN:
        fired.append(f"(1) mean faithfulness {means['faithfulness']} < {DECISION_FAITHFULNESS_MIN}")
    if means["answer_relevancy"] is None or means["answer_relevancy"] < DECISION_RELEVANCY_MIN:
        fired.append(f"(2) mean answer_relevancy {means['answer_relevancy']} < {DECISION_RELEVANCY_MIN}")
    stripped = sum(r["n_stripped_citations"] for r in rows)
    if stripped > DECISION_MAX_STRIPPED:
        fired.append(f"(3) {stripped} hallucinated citations stripped (switch at >= {DECISION_MAX_STRIPPED + 1})")
    return fired


def _chunk_texts(chunk_ids: list[str]) -> list[str]:
    """Look up chunk code text by ID in the committed BM25 chunk cache."""
    with open(BM25_CHUNKS_PATH, "r", encoding="utf-8") as f:
        by_id = {f"{c['repo_name']}:{c['file_path']}:{c['name']}:{c['start_line']}": c["code_text"] for c in json.load(f)}
    return [by_id[cid] for cid in chunk_ids]


def smoke_records() -> list[PipelineRecord]:
    """Three hand-built records: clearly good, clearly wrong, out-of-scope.

    Good and wrong share the same question and real contexts, so only the
    answer differs. The good answer is fully supported by the three
    LimitedStream chunks; the wrong one is unfaithful and off-topic.
    """
    ids = [
        "werkzeug:data/repos/werkzeug/src/werkzeug/wsgi.py:LimitedStream.on_exhausted:482",
        "werkzeug:data/repos/werkzeug/src/werkzeug/wsgi.py:LimitedStream.readinto:534",
        "werkzeug:data/repos/werkzeug/src/werkzeug/wsgi.py:LimitedStream.readall:582",
    ]
    contexts = _chunk_texts(ids)
    question = "What does werkzeug's LimitedStream.on_exhausted method do, and when is it called?"
    reference = (
        "LimitedStream.on_exhausted is called when attempting to read from the stream after its byte "
        "limit has been reached. By default it does nothing, except when the limit is a maximum, in which "
        "case it raises RequestEntityTooLarge. It is invoked from readinto() and readall() when the "
        "remaining byte budget is exhausted."
    )
    common = dict(ground_truth=reference, ground_truth_context_chunk_ids=ids, retrieved_chunk_ids=ids, blocked=False)
    return [
        PipelineRecord(question=question, contexts=contexts, answer=(
            "on_exhausted is called when a read is attempted after the byte limit has been reached [S1]. "
            "It does nothing by default, but if the limit is a maximum it raises RequestEntityTooLarge [S1]. "
            "readinto() calls it when the remaining budget (limit minus position) is zero or less [S2], and "
            "readall() calls it when the stream is already exhausted [S3]."
        ), **common),
        PipelineRecord(question=question, contexts=contexts, answer=(
            "on_exhausted configures the Jinja2 template cache size and recompiles every template when the "
            "Flask application starts [S1]. It is called by the CLI's run command [S2]."
        ), **common),
        PipelineRecord(
            question="What's the weather like in Bangalore today?",
            contexts=contexts,
            answer="The provided code context does not contain any information about the weather in Bangalore.",
            ground_truth="This question is outside the scope of the codebase.",
            ground_truth_context_chunk_ids=[],
            retrieved_chunk_ids=ids,
            blocked=False,
        ),
    ]


def write_csv(rows: list[dict], path: Path) -> None:
    """Write score rows to CSV (empty cells for unscored metrics)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_records(path: Path) -> list[PipelineRecord]:
    """Load and validate records written by src.eval.run_pipeline."""
    with open(path, "r", encoding="utf-8") as f:
        return [PipelineRecord.model_validate(r) for r in json.load(f)]


def main() -> None:
    """CLI entrypoint: judge smoke test, or score a pipeline results file."""
    parser = argparse.ArgumentParser(description="Score pipeline records with RAGAS.")
    parser.add_argument("--smoke", action="store_true", help="run the 3-record judge smoke test only")
    parser.add_argument("--records", type=Path, default=Path("eval/pipeline_results.json"))
    parser.add_argument("--out", type=Path, default=Path("eval/ragas_scores.csv"))
    parser.add_argument("--decision-rule", action="store_true", help="apply the Ollama-vs-Gemini rule")
    args = parser.parse_args()

    setup_logging()
    llm, embeddings, judge_label = build_judge()
    logger.info(f"Judge: {judge_label}; embeddings: {settings.embedding_model}")

    if args.smoke:
        rows = score_records(smoke_records(), llm, embeddings, judge_label)
        out = Path("eval/judge_smoke_test.csv")
        write_csv(rows, out)
        for label, row in zip(["good", "wrong", "out_of_scope"], rows):
            logger.info(f"{label:13s} " + " ".join(f"{m}={row[m]}" for m in METRIC_COLUMNS))
        fired = judge_usability(rows[0], rows[1])
        if fired:
            logger.warning(f"Judge UNUSABLE ({judge_label}): " + "; ".join(fired))
        else:
            logger.info(f"Judge usable ({judge_label}): no unusable criterion fired")
        logger.info(f"Saved {out}")
        return

    records = load_records(args.records)
    rows = score_records(records, llm, embeddings, judge_label)
    write_csv(rows, args.out)

    answered = [r for r in rows if not r["blocked"]]
    logger.info(f"Scored {len(rows)} records ({len(answered)} answered, {len(rows) - len(answered)} blocked) "
                f"with judge {judge_label}; unscored metric cells: {nan_count(rows)}")
    for m, v in metric_means(rows).items():
        logger.info(f"mean {m}: {'n/a' if v is None else f'{v:.3f}'}")
    logger.info(f"Saved {args.out}")

    if args.decision_rule:
        fired = decision_rule(rows)
        if fired:
            logger.warning("Decision rule: SWITCH to Gemini. Fired: " + "; ".join(fired))
        else:
            logger.info("Decision rule criteria 1-3: none fired (stay on Ollama unless criterion 4 fires)")
        logger.info("Criterion 4 (payment-failure fabrication) must be checked by reading the answer.")


if __name__ == "__main__":
    main()
