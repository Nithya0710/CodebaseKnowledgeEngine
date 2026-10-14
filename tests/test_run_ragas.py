import json
import math

import pandas as pd
import pytest

from src.eval import run_ragas
from src.eval.gold_set import BASELINE5_QUESTIONS, load_gold_set
from src.eval.run_ragas import (
    METRIC_COLUMNS,
    PipelineRecord,
    decision_rule,
    id_retrieval_metrics,
    judge_usability,
    metric_means,
    records_to_dataset,
    score_records,
    smoke_records,
)


def make_record(question="q", blocked=False, gold=("a",), retrieved=("a", "b"), stripped=0):
    return PipelineRecord(
        question=question,
        answer="refused" if blocked else f"answer to {question} [S1]",
        contexts=[] if blocked else ["ctx a", "ctx b"],
        ground_truth=f"reference for {question}",
        ground_truth_context_chunk_ids=list(gold),
        retrieved_chunk_ids=[] if blocked else list(retrieved),
        blocked=blocked,
        n_valid_citations=0 if blocked else 1,
        n_stripped_citations=stripped,
    )


def make_row(blocked=False, n_gold=1, stripped=0, **metrics):
    row = {"blocked": blocked, "n_gold_chunks": n_gold, "n_stripped_citations": stripped,
           "id_precision": None, "id_recall": None}
    row.update({m: metrics.get(m) for m in METRIC_COLUMNS})
    return row


class TestIdRetrievalMetrics:
    def test_precision_and_recall(self):
        assert id_retrieval_metrics(["a", "b", "c", "d"], ["a", "c", "z"]) == (0.5, 2 / 3)

    def test_nothing_retrieved_gives_undefined_precision(self):
        assert id_retrieval_metrics([], ["a"]) == (None, 0.0)

    def test_empty_gold_gives_undefined_recall(self):
        """Questions with no answer in the corpus have no gold chunks, so
        recall is undefined rather than 0 or 1."""
        assert id_retrieval_metrics(["a", "b"], []) == (0.0, None)

    def test_duplicate_retrieved_ids_count_once(self):
        assert id_retrieval_metrics(["a", "a"], ["a"]) == (1.0, 1.0)


class TestRecordsToDataset:
    def test_blocked_records_are_excluded(self):
        records = [make_record("q1"), make_record("q2", blocked=True), make_record("q3")]
        dataset, included = records_to_dataset(records)
        assert included == [0, 2]
        assert [s.user_input for s in dataset.samples] == ["q1", "q3"]

    def test_fields_map_to_ragas_columns(self):
        dataset, _ = records_to_dataset([make_record("q1")])
        sample = dataset.samples[0]
        assert sample.response == "answer to q1 [S1]"
        assert sample.retrieved_contexts == ["ctx a", "ctx b"]
        assert sample.reference == "reference for q1"

    def test_all_blocked_gives_empty_dataset(self):
        dataset, included = records_to_dataset([make_record(blocked=True)])
        assert included == []
        assert len(dataset.samples) == 0


class TestScoreRecords:
    def test_all_blocked_never_calls_the_judge(self, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("evaluate must not run when every record is blocked")
        monkeypatch.setattr(run_ragas, "evaluate", fail)
        rows = score_records([make_record(blocked=True)], llm=None, embeddings=None, judge_label="fake")
        assert rows[0]["blocked"] is True
        assert all(rows[0][m] is None for m in METRIC_COLUMNS)

    def test_scores_map_back_to_answered_rows_and_nan_becomes_none(self, monkeypatch):
        class FakeResult:
            def to_pandas(self):
                return pd.DataFrame([
                    {"faithfulness": 0.9, "answer_relevancy": 0.8,
                     "llm_context_precision_with_reference": 0.7, "context_recall": 0.6},
                    {"faithfulness": math.nan, "answer_relevancy": 0.5,
                     "llm_context_precision_with_reference": 0.4, "context_recall": 0.3},
                ])
        monkeypatch.setattr(run_ragas, "evaluate", lambda *a, **k: FakeResult())
        records = [make_record("q1"), make_record("q2", blocked=True), make_record("q3")]
        rows = score_records(records, llm=None, embeddings=None, judge_label="fake")

        assert rows[0]["faithfulness"] == 0.9 and rows[0]["context_precision"] == 0.7
        assert all(rows[1][m] is None for m in METRIC_COLUMNS)
        assert rows[2]["faithfulness"] is None
        assert rows[2]["answer_relevancy"] == 0.5
        assert rows[0]["judge"] == "fake"
        assert rows[0]["id_precision"] == 0.5 and rows[0]["id_recall"] == 1.0


class TestJudgeUsability:
    def test_discriminating_judge_is_usable(self):
        good = make_row(faithfulness=1.0, answer_relevancy=0.9, context_precision=1.0, context_recall=1.0)
        wrong = make_row(faithfulness=0.0, answer_relevancy=0.2, context_precision=1.0, context_recall=1.0)
        assert judge_usability(good, wrong) == []

    def test_identical_answer_scores_fire_b(self):
        good = make_row(faithfulness=1.0, answer_relevancy=0.9, context_precision=1.0, context_recall=1.0)
        wrong = make_row(faithfulness=1.0, answer_relevancy=0.2, context_precision=1.0, context_recall=1.0)
        fired = judge_usability(good, wrong)
        assert len(fired) == 1 and fired[0].startswith("(b) faithfulness")

    def test_wrong_scoring_higher_fires_c(self):
        good = make_row(faithfulness=0.4, answer_relevancy=0.9, context_precision=1.0, context_recall=1.0)
        wrong = make_row(faithfulness=0.8, answer_relevancy=0.2, context_precision=1.0, context_recall=1.0)
        assert any(f.startswith("(c) faithfulness") for f in judge_usability(good, wrong))

    def test_many_unscored_cells_fire_a(self):
        good = make_row(faithfulness=1.0, answer_relevancy=0.9)
        wrong = make_row(faithfulness=0.0, answer_relevancy=0.1)
        fired = judge_usability(good, wrong)
        assert any(f.startswith("(a)") and "50%" in f for f in fired)

    def test_unscored_answer_metric_is_not_comparable(self):
        good = make_row(faithfulness=None, answer_relevancy=0.9, context_precision=1.0, context_recall=1.0)
        wrong = make_row(faithfulness=0.0, answer_relevancy=0.1, context_precision=1.0, context_recall=1.0)
        assert any("faithfulness not comparable" in f for f in judge_usability(good, wrong))


class TestDecisionRule:
    def _good_rows(self):
        return [make_row(faithfulness=0.9, answer_relevancy=0.85) for _ in range(4)]

    def test_nothing_fires_on_good_results(self):
        assert decision_rule(self._good_rows()) == []

    def test_low_faithfulness_fires_1(self):
        rows = self._good_rows() + [make_row(faithfulness=0.0, answer_relevancy=0.9)] * 3
        assert any(f.startswith("(1)") for f in decision_rule(rows))

    def test_two_stripped_citations_fire_3(self):
        rows = self._good_rows()
        rows[0]["n_stripped_citations"] = 1
        rows[1]["n_stripped_citations"] = 1
        assert any(f.startswith("(3)") for f in decision_rule(rows))

    def test_one_stripped_citation_does_not_fire_3(self):
        rows = self._good_rows()
        rows[0]["n_stripped_citations"] = 1
        assert decision_rule(rows) == []

    def test_no_gold_question_is_excluded_from_means(self):
        """An honest 'not in the corpus' answer gets relevancy 0 from RAGAS
        (noncommittal); it must not drag criteria 1-2 down."""
        rows = self._good_rows() + [make_row(n_gold=0, faithfulness=None, answer_relevancy=0.0)]
        assert decision_rule(rows) == []
        assert metric_means(rows)["answer_relevancy"] < 0.7

    def test_blocked_rows_are_ignored(self):
        rows = self._good_rows() + [make_row(blocked=True)]
        assert decision_rule(rows) == []

    def test_all_unscored_fires_rather_than_passing(self):
        rows = [make_row(faithfulness=None, answer_relevancy=None)]
        fired = decision_rule(rows)
        assert any(f.startswith("(1)") for f in fired) and any(f.startswith("(2)") for f in fired)


class TestSmokeRecords:
    def test_three_records_with_real_contexts(self):
        records = smoke_records()
        assert len(records) == 3
        good, wrong, oos = records
        assert good.question == wrong.question
        assert good.contexts == wrong.contexts and len(good.contexts) == 3
        assert "def on_exhausted" in good.contexts[0]
        assert oos.ground_truth_context_chunk_ids == []


class TestGoldSet:
    def test_baseline5_loads_from_real_test_set_in_order(self):
        entries = load_gold_set(subset="baseline5")
        assert [e.question for e in entries] == list(BASELINE5_QUESTIONS)

    def test_unknown_subset_raises(self):
        with pytest.raises(ValueError, match="Unknown subset"):
            load_gold_set(subset="nope")

    def test_subset_question_missing_from_file_raises(self, tmp_path):
        path = tmp_path / "test_set.json"
        path.write_text(json.dumps([{
            "question": BASELINE5_QUESTIONS[0], "ground_truth_answer": "x",
            "ground_truth_context_chunk_ids": [],
        }]))
        with pytest.raises(ValueError, match="missing"):
            load_gold_set(path, subset="baseline5")

    def test_full_set_loads_without_subset(self, tmp_path):
        path = tmp_path / "test_set.json"
        path.write_text(json.dumps([{"question": "q", "ground_truth_answer": "a",
                                     "ground_truth_context_chunk_ids": ["id"]}]))
        assert [e.question for e in load_gold_set(path)] == ["q"]

    def test_every_gold_chunk_id_exists_in_committed_corpus(self):
        with open("data/bm25_chunks.json", encoding="utf-8") as f:
            corpus = {f"{c['repo_name']}:{c['file_path']}:{c['name']}:{c['start_line']}" for c in json.load(f)}
        missing = [cid for e in load_gold_set() for cid in e.ground_truth_context_chunk_ids if cid not in corpus]
        assert missing == []
