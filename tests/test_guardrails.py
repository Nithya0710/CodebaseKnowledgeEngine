from src.generation.guardrails import (
    check_context_sufficiency,
    validate_citations,
    InsufficientContextResponse,
    resolve_citations,
    validate_citations_with_counts,
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
    def _two_chunks(self):
        return [
            (make_chunk(name="alpha", file_path="a.py", repo_name="r", start_line=1), 0.9, -2.0),
            (make_chunk(name="beta", file_path="b.py", repo_name="r", start_line=5), 0.8, -3.0),
        ]

    def test_out_of_range_label_is_stripped_and_logged(self, caplog):
        answer = "It registers a route [S1]. It also checks auth [S7]."
        cleaned = validate_citations(answer, self._two_chunks())

        assert "[S1]" in cleaned
        assert "[S7]" not in cleaned
        assert "Hallucinated citation detected" in caplog.text
        assert cleaned == "It registers a route [S1]. It also checks auth."

    def test_s0_is_invalid(self):
        """Labels are 1-indexed, so S0 must be treated as fake."""
        cleaned = validate_citations("Claim [S0].", self._two_chunks())
        assert "[S0]" not in cleaned

    def test_mixed_group_keeps_only_valid_labels(self):
        cleaned = validate_citations("Claim [S1, S9].", self._two_chunks())
        assert "[S1]" in cleaned
        assert "S9" not in cleaned

    def test_fully_invalid_group_is_removed(self):
        cleaned = validate_citations("Claim [S8, S9].", self._two_chunks())
        assert "[" not in cleaned

    def test_ordinary_code_indexing_is_not_touched(self):
        """The reason the label is [S1], not [1]: code like args[0] in an
        answer must never be mistaken for a citation."""
        answer = "It reads `args[0]` and `parts[1]` then returns [S1]."
        cleaned = validate_citations(answer, self._two_chunks())
        assert "args[0]" in cleaned
        assert "parts[1]" in cleaned
        assert "[S1]" in cleaned

    def test_answer_with_no_citations_is_unchanged(self):
        answer = "This is a plain answer with no citations at all."
        assert validate_citations(answer, self._two_chunks()) == answer

    def test_valid_comma_group_is_kept_intact(self):
        answer = "Both functions register routes [S1, S2]."
        assert validate_citations(answer, self._two_chunks()) == answer

    def test_stripped_tag_mid_sentence_leaves_single_space(self):
        answer = "It checks auth [S9] before dispatch [S2]."
        cleaned = validate_citations(answer, self._two_chunks())
        assert cleaned == "It checks auth before dispatch [S2]."

    def test_stripped_tag_at_line_start_leaves_no_leading_space(self):
        answer = "[S9] Flask routes requests. More detail [S1]."
        cleaned = validate_citations(answer, self._two_chunks())
        assert cleaned == "Flask routes requests. More detail [S1]."

    def test_layout_and_code_indentation_are_preserved(self):
        """Regression: an earlier version collapsed all whitespace in the
        answer, flattening paragraphs and code-block indentation (which
        would make every multi-line snippet fail a syntax check)."""
        answer = (
            "Paragraph one [S9].\n\n"
            "```python\n"
            "def f(x):\n"
            "    if x :\n"
            "        return 1\n"
            "```\n"
            "Paragraph two [S1]."
        )
        expected = (
            "Paragraph one.\n\n"
            "```python\n"
            "def f(x):\n"
            "    if x :\n"
            "        return 1\n"
            "```\n"
            "Paragraph two [S1]."
        )
        assert validate_citations(answer, self._two_chunks()) == expected

    def test_counts_valid_and_stripped_labels(self):
        answer = "A [S1]. B [S1, S9]. C [S7]. D [S2]."
        result = validate_citations_with_counts(answer, self._two_chunks())
        assert result.n_valid_citations == 3
        assert result.n_stripped_citations == 2
        assert result.answer == "A [S1]. B [S1]. C. D [S2]."

    def test_counts_are_zero_without_citations(self):
        result = validate_citations_with_counts("No citations here.", self._two_chunks())
        assert (result.n_valid_citations, result.n_stripped_citations) == (0, 0)


class TestResolveCitations:
    def _two_chunks(self):
        return [
            (make_chunk(name="alpha", file_path="a.py", repo_name="r", start_line=1), 0.9, -2.0),
            (make_chunk(name="beta", file_path="b.py", repo_name="r", start_line=5), 0.8, -3.0),
        ]

    def test_labels_map_to_real_chunk_ids_in_order_without_duplicates(self):
        answer = "First [S2], then [S1], and again [S2]."
        assert resolve_citations(answer, self._two_chunks()) == [
            "r:b.py:beta:5",
            "r:a.py:alpha:1",
        ]

    def test_invalid_labels_are_ignored(self):
        assert resolve_citations("Claim [S5].", self._two_chunks()) == []

    def test_grouped_labels_are_resolved(self):
        assert resolve_citations("Claim [S1, S2].", self._two_chunks()) == [
            "r:a.py:alpha:1",
            "r:b.py:beta:5",
        ]