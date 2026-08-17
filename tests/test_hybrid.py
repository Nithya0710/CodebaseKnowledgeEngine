import pytest
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.keyword_index import BM25KeywordIndex
from src.retrieval.vector_store import VectorStoreManager
from src.chunking.ast_parser import CodeChunk
from src.retrieval.hybrid import hybrid_retrieve


def make_chunk(
    repo_name="testrepo",
    file_path="mod.py",
    chunk_type="function",
    name="foo",
    start_line=1,
    end_line=5,
    code_text="def foo(): pass",
    docstring=None,
):
    return CodeChunk(
        repo_name=repo_name,
        file_path=file_path,
        chunk_type=chunk_type,
        name=name,
        start_line=start_line,
        end_line=end_line,
        code_text=code_text,
        docstring=docstring,
    )


class TestReciprocalRankFusion:
    def test_hand_computed_scores_two_lists(self):
        """
        Hand-computed RRF example with k=60, verified against the
        exact formula: score(doc) = sum over lists of 1/(k + rank),
        rank is 1-indexed, absence from a list contributes 0
        """
        list_a = ["chunk1", "chunk2", "chunk3"]
        list_b = ["chunk2", "chunk1", "chunk4"]

        result = reciprocal_rank_fusion([list_a, list_b], k=60)

        # chunk1: rank 1 in list_a, rank 2 in list_b
        expected_chunk1 = 1 / (60 + 1) + 1 / (60 + 2)
        # chunk2: rank 2 in list_a, rank 1 in list_b
        expected_chunk2 = 1 / (60 + 2) + 1 / (60 + 1)
        # chunk3: rank 3 in list_a only
        expected_chunk3 = 1 / (60 + 3)
        # chunk4: rank 3 in list_b only
        expected_chunk4 = 1 / (60 + 3)

        assert result["chunk1"] == pytest.approx(expected_chunk1)
        assert result["chunk2"] == pytest.approx(expected_chunk2)
        assert result["chunk3"] == pytest.approx(expected_chunk3)
        assert result["chunk4"] == pytest.approx(expected_chunk4)

        # chunk1 and chunk2 should tie, since they occupy the same rank
        # SET {1, 2} across the two lists, just swapped
        assert result["chunk1"] == pytest.approx(result["chunk2"])

    def test_hand_computed_scores_three_lists(self):
        """
        A doc appearing in all three lists should score higher than
        one appearing in only one — proves the 'sum across lists' part
        of the formula, not just per-list ranking
        """
        list_a = ["x", "y"]
        list_b = ["x", "z"]
        list_c = ["x", "y"]

        result = reciprocal_rank_fusion([list_a, list_b, list_c], k=60)

        # x: rank 1 in all three lists
        expected_x = 3 * (1 / (60 + 1))
        assert result["x"] == pytest.approx(expected_x)

        # y: rank 2 in list_a and list_c, absent from list_b
        expected_y = 2 * (1 / (60 + 2))
        assert result["y"] == pytest.approx(expected_y)

        # z: rank 2 in list_b only
        expected_z = 1 / (60 + 2)
        assert result["z"] == pytest.approx(expected_z)

        assert result["x"] > result["y"] > result["z"]

    def test_empty_lists_produce_empty_result(self):
        """
        Fusing zero ranked lists (or lists that are all empty) should
        return an empty dict, not raise
        """
        assert reciprocal_rank_fusion([]) == {}
        assert reciprocal_rank_fusion([[], []]) == {}

    def test_single_list_equals_pure_rank_score(self):
        """
        With only one input list, fusion should just reduce to that
        list's own 1/(k+rank) scores — no cross-list summing to obscure
        """
        single_list = ["a", "b", "c"]
        result = reciprocal_rank_fusion([single_list], k=60)

        assert result["a"] == pytest.approx(1 / 61)
        assert result["b"] == pytest.approx(1 / 62)
        assert result["c"] == pytest.approx(1 / 63)


@pytest.fixture
def vector_store(tmp_path):
    return VectorStoreManager(persist_dir=str(tmp_path))


class TestHybridRetrieveEdgeCases:
    def test_empty_corpus_returns_empty_list(self, vector_store):
        """
        If both BM25 and the vector store are empty (nothing upserted
        or indexed), hybrid_retrieve should return an empty list, not
        raise — mirrors the Day 6/7 empty-collection tests, now at the
        fusion layer
        """
        bm25_index = BM25KeywordIndex([])
        results = hybrid_retrieve("anything", bm25_index, vector_store)
        assert results == []

    def test_mismatched_corpora_does_not_crash(self, vector_store):
        """
        This is the real bug class from today's session: BM25 and the
        vector store were built from different corpora (BM25 = flask
        only, vector store = click only), so most of BM25's top results
        had no matching entry in the vector store's metadata and vice
        versa. hybrid_retrieve must handle this gracefully — via the
        metadata-reconstruction fallback — rather than crashing on a
        missing chunk.
        """
        bm25_only_chunk = make_chunk(name="bm25_exclusive", code_text="def bm25_exclusive(): pass")
        shared_chunk = make_chunk(name="shared_fn", code_text="def shared_fn(): pass", start_line=20)
        vector_only_chunk = make_chunk(name="vector_exclusive", code_text="def vector_exclusive(): pass", start_line=30)

        # BM25 only knows about bm25_only_chunk + shared_chunk
        bm25_index = BM25KeywordIndex([bm25_only_chunk, shared_chunk])

        # Vector store only knows about vector_exclusive_chunk + shared_chunk
        vector_store.upsert_chunks([vector_only_chunk, shared_chunk])

        results = hybrid_retrieve("function", bm25_index, vector_store, top_k_per_source=15, top_k_final=15)

        result_names = {chunk.name for chunk in results}
        # Should not crash, and should surface whatever each retriever
        # actually found — no silent total failure.
        assert len(results) > 0
        assert "shared_fn" in result_names

    def test_doc_and_code_match_same_chunk_deduplicates(self, vector_store):
        """
        A chunk with a docstring gets embedded into BOTH code_chunks
        and doc_chunks (Day 6), under two different Chroma IDs
        (id vs id_doc). hybrid_retrieve must recognize these as the
        SAME logical chunk and count it once in the fused results, not
        twice — this is the exact bug fixed earlier today
        """
        chunk = make_chunk(
            name="documented_fn",
            code_text="def documented_fn(): return 42",
            docstring="This function returns 42.",
        )
        vector_store.upsert_chunks([chunk])
        bm25_index = BM25KeywordIndex([chunk])

        results = hybrid_retrieve("documented_fn", bm25_index, vector_store, top_k_per_source=15, top_k_final=15)

        matching = [c for c in results if c.name == "documented_fn"]
        assert len(matching) == 1, f"Expected exactly one entry for documented_fn, got {len(matching)}"

    def test_top_k_final_greater_than_unique_results(self, vector_store):
        """
        If top_k_final exceeds the number of unique fused chunks
        available, hybrid_retrieve should return everything it has,
        not error or pad with junk — same principle as Day 6's
        top_k-greater-than-collection-size test, now at fusion level
        """
        chunks = [
            make_chunk(name="foo", start_line=1, code_text="def foo(): pass"),
            make_chunk(name="bar", start_line=10, code_text="def bar(): pass"),
        ]
        vector_store.upsert_chunks(chunks)
        bm25_index = BM25KeywordIndex(chunks)

        results = hybrid_retrieve("foo bar", bm25_index, vector_store, top_k_per_source=15, top_k_final=100)

        result_names = {chunk.name for chunk in results}
        assert result_names == {"foo", "bar"}
        assert len(results) == 2