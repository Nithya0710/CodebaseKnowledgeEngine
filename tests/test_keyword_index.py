import pytest

from pathlib import Path
from src.retrieval.keyword_index import split_identifiers, BM25KeywordIndex
from src.chunking.ast_parser import CodeChunk, chunk_repo


class TestBM25Persistence:
    """
    Proves save_chunks/load round-trips correctly: chunks written to
    disk and reloaded produce an index whose contents and query results
    match the original, unsaved index exactly
    """

    def test_save_and_load_round_trip(self, tmp_path):
        chunks = [
            make_chunk(name="foo", code_text="def foo(): pass"),
            make_chunk(name="get_user_by_id", code_text="def get_user_by_id(uid): return db.fetch(uid)"),
            make_chunk(name="bar", code_text="def bar(): pass"),
        ]
        index = BM25KeywordIndex(chunks)

        save_path = tmp_path / "bm25_chunks.json"
        index.save_chunks(str(save_path))

        assert save_path.exists()

        reloaded = BM25KeywordIndex.load(str(save_path))

        assert len(reloaded.chunks) == len(chunks)
        assert [c.name for c in reloaded.chunks] == [c.name for c in chunks]

        original_results = index.query("get_user_by_id", top_k=1)
        reloaded_results = reloaded.query("get_user_by_id", top_k=1)

        assert original_results[0][0].name == reloaded_results[0][0].name
        assert original_results[0][1] == pytest.approx(reloaded_results[0][1])

    def test_load_missing_file_raises(self, tmp_path):
        """
        Loading from a path that doesn't exist should raise, not
        silently return an empty or broken index — a missing index file
        is a real startup error the caller needs to know about
        """
        missing_path = tmp_path / "does_not_exist.json"
        with pytest.raises(Exception):
            BM25KeywordIndex.load(str(missing_path))


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


class TestSplitIdentifiers:
    """
    Proves split_identifiers correctly decomposes camelCase and
    snake_case identifiers into lowercased sub-word tokens
    """

    @pytest.mark.parametrize("identifier,expected", [
        ("getUserById", ["get", "user", "by", "id"]),
        ("parseJSONResponse", ["parse", "json", "response"]),
        ("HTTPSConnectionPool", ["https", "connection", "pool"]),
        ("toString", ["to", "string"]),
        ("isValidEmail", ["is", "valid", "email"]),
    ])
    def test_camel_case_splitting(self, identifier, expected):
        assert split_identifiers(identifier) == expected

    @pytest.mark.parametrize("identifier,expected", [
        ("process_payment_result", ["process", "payment", "result"]),
        ("get_user_by_id", ["get", "user", "by", "id"]),
        ("max_retry_count", ["max", "retry", "count"]),
        ("is_valid", ["is", "valid"]),
        ("_private_helper", ["private", "helper"]),
    ])
    def test_snake_case_splitting(self, identifier, expected):
        assert split_identifiers(identifier) == expected


class TestBM25KeywordIndex:
    def test_query_returns_ranked_chunks(self):
        """
        Basic sanity: querying for a token present in only one chunk
        should rank that chunk first
        """
        chunks = [
            make_chunk(name="foo", code_text="def foo(): pass"),
            make_chunk(name="get_user_by_id", code_text="def get_user_by_id(uid): return db.fetch(uid)"),
            make_chunk(name="bar", code_text="def bar(): pass"),
        ]
        index = BM25KeywordIndex(chunks)
        results = index.query("get_user_by_id", top_k=1)
        assert results[0][0].name == "get_user_by_id"

    def test_getuserbyid_retrieval_beats_naive_baseline(self):
        """
        Integration test proving BM25 + identifier-aware tokenization
        retrieves an exact function match on a real repo, where Day 5's
        naive embedding-only baseline demonstrably struggled.

        Day 5 evidence: querying for 'add_url_rule' via naive cosine
        similarity returned two correct add_url_rule hits ALONGSIDE an
        irrelevant httpx URL-parsing chunk (URL.is_absolute_url), which
        the LLM then incorrectly wove into its answer. BM25's exact
        sub-word matching should surface only genuinely related chunks,
        with no such contamination.
        """
        repo_path = Path("data/repos/flask")
        if not repo_path.exists():
            pytest.skip("flask repo not present at data/repos/flask")

        chunks = chunk_repo(repo_path, repo_name="flask")
        index = BM25KeywordIndex(chunks)
        results = index.query("add_url_rule", top_k=5)

        result_names = [chunk.name for chunk, _ in results]
        assert any("add_url_rule" in name.lower() for name in result_names), (
            f"Expected an add_url_rule chunk in top 5, got: {result_names}"
        )

        # Contamination check: no unrelated httpx URL-parsing chunks should
        # appear in the top results, unlike Day 5's naive baseline.
        result_files = [chunk.file_path for chunk, _ in results]
        assert not any("httpx" in f.lower() for f in result_files), (
            f"BM25 pulled in unrelated httpx chunks, same contamination as naive baseline: {result_files}"
        )