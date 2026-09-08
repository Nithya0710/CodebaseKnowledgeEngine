import pytest
from src.generation.guardrails import (
    check_context_sufficiency,
    validate_citations,
    InsufficientContextResponse,
)
from src.retrieval.rerank import RAW_SCORE_GUARDRAIL_THRESHOLD
from src.chunking.ast_parser import CodeChunk


def make_chunk(name="foo", file_path="a.py", repo_name="test", start_line=1):
    return CodeChunk(
        repo_name=repo_name, file_path=file_path, chunk_type="function",
        name=name, start_line=start_line, end_line=start_line + 1,
        code_text=f"def {name}(): pass", docstring=None,
    )


class TestPreGenerationGuardrail:
    def test_out_of_scope_question_blocks_generation(self):
        """
        An obviously out-of-scope question should produce a low RAW
        cross-encoder score (below RAW_SCORE_GUARDRAIL_THRESHOLD), which
        must block generation — this directly regression-tests the bug
        found on real data, where a weather question's BLENDED score
        (0.72) incorrectly cleared the old 0.65 threshold despite every
        candidate's raw score being clearly irrelevant (-10.70).
        """
        weak_chunks = [(make_chunk(), 0.72, -10.70)]  # (chunk, blended, raw)

        result = check_context_sufficiency("what's the weather today", weak_chunks)

        assert result is not None
        assert isinstance(result, InsufficientContextResponse)
        assert result.best_score_achieved == -10.70
        assert result.threshold == RAW_SCORE_GUARDRAIL_THRESHOLD

    def test_strong_context_allows_generation(self):
        """
        A question with a confidently high RAW cross-encoder score
        should return None, allowing generation to proceed — even if
        its blended score happens to be modest, since blended score is
        no longer what the guardrail checks.
        """
        strong_chunks = [(make_chunk(), 0.55, -2.6167)]  # real Q4 example from Day 10/11
        result = check_context_sufficiency("how does add_url_rule work?", strong_chunks)
        assert result is None

    def test_no_chunks_retrieved_blocks_generation(self):
        """
        Empty retrieval (nothing at all came back) must also block
        generation, with best_score_achieved reported as 0.0.
        """
        result = check_context_sufficiency("totally unrelated question", [])
        assert result is not None
        assert result.best_score_achieved == 0.0

    def test_best_of_multiple_chunks_is_used_not_first(self):
        """
        The guardrail must check the MAXIMUM raw score across all
        retrieved chunks, not just the first one in the list — proves
        it doesn't accidentally check only ranked_chunks[0].
        """
        chunks = [
            (make_chunk(name="a"), 0.5, -12.0),
            (make_chunk(name="b"), 0.4, -3.0),   # this one should pass the gate
            (make_chunk(name="c"), 0.3, -15.0),
        ]
        result = check_context_sufficiency("some question", chunks)
        assert result is None  # -3.0 clears the -10.0 threshold


class TestPostGenerationCitationValidator:
    def test_fake_chunk_id_is_detected_and_stripped(self, caplog):
        """
        A deliberately injected fake chunk_id (one that does NOT
        correspond to anything in the retrieved context) must be
        detected and stripped from the visible answer, with the
        incident logged.
        """
        real_chunk = make_chunk(name="add_url_rule", file_path="flask/app.py", repo_name="flask", start_line=10)
        ranked_chunks = [(real_chunk, 0.91, -2.5)]

        fake_id = "flask:flask/security.py:check_auth:5"
        answer = (
            f"Route registration happens here [cite: flask:flask/app.py:add_url_rule:10]. "
            f"Auth is also checked [cite: {fake_id}]."
        )

        cleaned = validate_citations(answer, ranked_chunks)

        assert fake_id not in cleaned
        assert "[cite: flask:flask/app.py:add_url_rule:10]" in cleaned
        assert "Hallucinated citation detected" in caplog.text

    def test_answer_with_no_citations_is_unchanged(self):
        """
        An answer with no citation tags at all should pass through
        validate_citations completely unchanged — no false positives
        on plain text.
        """
        ranked_chunks = [(make_chunk(), 0.91, -2.5)]
        answer = "This is a plain answer with no citations at all."
        assert validate_citations(answer, ranked_chunks) == answer

    def test_chunk_id_format_variant_is_also_validated(self):
        """
        Real generations were observed mixing [cite: ...] and
        [chunk_id: ...] labels for the same underlying citation concept
        (the model echoing the context block's label instead of the
        instructed wrapper). Both must be recognized and validated —
        a real citation in [chunk_id: ...] form should survive, and a
        fake one in that form should be caught and stripped.
        """
        real_chunk = make_chunk(name="add_url_rule", file_path="flask/app.py", repo_name="flask", start_line=10)
        ranked_chunks = [(real_chunk, 0.91, -2.5)]

        answer = (
            "Registers a route [chunk_id: flask:flask/app.py:add_url_rule:10]. "
            "Also checks auth [chunk_id: flask:flask/security.py:fake_fn:99]."
        )

        cleaned = validate_citations(answer, ranked_chunks)

        assert "[chunk_id: flask:flask/app.py:add_url_rule:10]" in cleaned
        assert "flask:flask/security.py:fake_fn:99" not in cleaned