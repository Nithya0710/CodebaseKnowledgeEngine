import pytest
from src.retrieval.vector_store import VectorStoreManager
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
    return VectorStoreManager(persist_dir=str(tmp_path))


def test_upsert_duplicate_id_updates_not_duplicates(store):
    """
    Upserting a chunk with the same identifying fields (repo/file/name/start_line)
    twice should update in place, not create a second entry — this is what makes
    re-ingesting an unchanged repo idempotent
    """
    chunk_v1 = make_chunk(code_text="def foo(): pass", docstring="v1 docstring")
    store.upsert_chunks([chunk_v1])

    assert store.code_collection.count() == 1
    assert store.doc_collection.count() == 1

    chunk_v2 = make_chunk(code_text="def foo(): return 42", docstring="v2 docstring")
    store.upsert_chunks([chunk_v2])

    # Count must NOT grow — same ID, same slot
    assert store.code_collection.count() == 1
    assert store.doc_collection.count() == 1

    # Content must reflect the update, not the original
    chunk_id = VectorStoreManager._chunk_id(chunk_v2)
    stored_code = store.code_collection.get(ids=[chunk_id])
    assert stored_code["documents"][0] == "def foo(): return 42"

    doc_id = f"{chunk_id}_doc"
    stored_doc = store.doc_collection.get(ids=[doc_id])
    assert stored_doc["documents"][0] == "v2 docstring"


def test_query_empty_collection(store):
    """
    Querying before anything has been upserted should return an empty list,
    not raise — downstream retrieval code shouldn't need a special empty-store
    code path.
    """
    results = store.query_code("anything", top_k=5)
    assert results == []

    results = store.query_docs("anything", top_k=5)
    assert results == []


def test_query_top_k_greater_than_collection_size(store):
    """
    If top_k exceeds the number of stored chunks, Chroma should just return
    everything it has — not error, not pad with junk
    """
    chunks = [
        make_chunk(name="foo", start_line=1, code_text="def foo(): pass"),
        make_chunk(name="bar", start_line=10, code_text="def bar(): pass"),
    ]
    store.upsert_chunks(chunks)

    results = store.query_code("foo", top_k=10)
    assert len(results) == 2

    returned_names = {r["metadata"]["name"] for r in results}
    assert returned_names == {"foo", "bar"}