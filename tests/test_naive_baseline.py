import pytest
import requests

from unittest.mock import patch
from src.retrieval.naive_baseline import NaiveBaselineStore, naive_query
from src.chunking.ast_parser import CodeChunk


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


@pytest.fixture
def store(tmp_path):
    return NaiveBaselineStore(persist_dir=str(tmp_path))


def test_naive_query_empty_collection_returns_no_context_message(store):
    """
    Querying a collection with zero upserted chunks should return the
    'no relevant context' message rather than erroring or hitting Ollama
    with an empty context block
    """
    answer = naive_query(store, "anything at all", top_k=5)
    assert answer == "[No relevant context found in the corpus.]"


def test_naive_query_ollama_unreachable_returns_error_string(store):
    """
    If retrieval succeeds but Ollama can't be reached, naive_query
    should catch the exception and return an error string rather than
    crashing
    """
    chunk = make_chunk(code_text="def foo(): return 42")
    store.upsert_chunks([chunk])

    with patch("src.retrieval.naive_baseline.requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        answer = naive_query(store, "what does foo do?", top_k=5)

    assert answer.startswith("[ERROR: generation failed:")


def test_upsert_chunks_idempotent_on_rerun(store):
    """
    Re-running upsert_chunks with the same chunks (same deterministic
    IDs) should not duplicate entries in the naive_baseline collection —
    idempotency guarantee
    """
    chunks = [
        make_chunk(name="foo", start_line=1, code_text="def foo(): pass"),
        make_chunk(name="bar", start_line=10, code_text="def bar(): pass"),
    ]
    store.upsert_chunks(chunks)
    assert store.collection.count() == 2

    store.upsert_chunks(chunks)  # re-run, unchanged
    assert store.collection.count() == 2