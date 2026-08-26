import pytest
from src.retrieval.rerank import rerank
from src.chunking.ast_parser import CodeChunk


def make_chunk(name, code_text="def foo(): pass", file_path="mod.py"):
    return CodeChunk(
        repo_name="test", file_path=file_path, chunk_type="function",
        name=name, start_line=1, end_line=2, code_text=code_text, docstring=None,
    )


class TestRerankEdgeCases:
    def test_empty_candidates_returns_empty_list(self):
        assert rerank("any question", [], keep_top=5) == []

    def test_single_candidate_returns_that_candidate(self):
        chunk = make_chunk("solo_fn", code_text="def solo_fn(): return 1")
        results = rerank("what does solo_fn do?", [chunk], keep_top=5)
        assert len(results) == 1
        assert results[0][0].name == "solo_fn"

    def test_keep_top_greater_than_candidates_returns_all(self):
        chunks = [make_chunk("a"), make_chunk("b"), make_chunk("c")]
        results = rerank("test query", chunks, keep_top=100)
        assert len(results) == 3

    def test_alpha_zero_uses_pure_prior_rank(self, monkeypatch):
        """
        alpha=0 should completely ignore the cross-encoder score and
        preserve the INPUT order exactly, since prior_rank normalization
        (1/(1+rank)) is monotonically decreasing with rank — proves the
        blend formula's alpha=0 edge behaves as a pure pass-through.
        """
        chunks = [make_chunk("first"), make_chunk("second"), make_chunk("third")]

        def fake_predict(pairs):
            # Deliberately give the LAST candidate the highest raw score,
            # to prove alpha=0 ignores this entirely.
            return [0.1, 0.5, 9.9]

        monkeypatch.setattr("src.retrieval.rerank._cross_encoder.predict", fake_predict)

        results = rerank("test query", chunks, keep_top=3, alpha=0.0)
        result_names = [c.name for c, _ in results]
        assert result_names == ["first", "second", "third"]

    def test_alpha_one_uses_pure_cross_encoder_score(self, monkeypatch):
        """
        alpha=1 should rank purely by cross-encoder score, ignoring
        input order entirely.
        """
        chunks = [make_chunk("first"), make_chunk("second"), make_chunk("third")]

        def fake_predict(pairs):
            # third has the highest raw score despite being last in input
            return [0.1, 0.5, 9.9]

        monkeypatch.setattr("src.retrieval.rerank._cross_encoder.predict", fake_predict)

        results = rerank("test query", chunks, keep_top=3, alpha=1.0)
        result_names = [c.name for c, _ in results]
        assert result_names == ["third", "second", "first"]

    def test_identical_cross_encoder_scores_does_not_crash(self, monkeypatch):
        """
        If every candidate scores identically (score_range == 0), the
        min-max normalization would divide by zero unless guarded —
        confirms the guard in normalize_ce_score handles this.
        """
        chunks = [make_chunk("a"), make_chunk("b"), make_chunk("c")]

        def fake_predict(pairs):
            return [3.0, 3.0, 3.0]

        monkeypatch.setattr("src.retrieval.rerank._cross_encoder.predict", fake_predict)

        results = rerank("test query", chunks, keep_top=3, alpha=0.65)
        assert len(results) == 3
        # With tied CE scores, prior rank should be the only differentiator
        result_names = [c.name for c, _ in results]
        assert result_names == ["a", "b", "c"]

    def test_cross_encoder_failure_returns_empty_list(self, monkeypatch):
        """
        If the cross-encoder itself raises (e.g. OOM, bad input), rerank
        should catch it and return [] rather than propagate the exception —
        matches the established pattern from every prior retrieval stage.
        """
        chunks = [make_chunk("a")]

        def fake_predict(pairs):
            raise RuntimeError("simulated model failure")

        monkeypatch.setattr("src.retrieval.rerank._cross_encoder.predict", fake_predict)

        results = rerank("test query", chunks, keep_top=5)
        assert results == []