import pytest
import json
from pathlib import Path
from src.chunking.ast_parser import CodeChunk
import src.retrieval.ingest as ingest_module
from src.retrieval.ingest import load_or_ingest_corpus


def make_chunk(name, repo_name="repoA", file_path="mod.py"):
    return CodeChunk(
        repo_name=repo_name, file_path=file_path, chunk_type="function",
        name=name, start_line=1, end_line=2, code_text=f"def {name}(): pass", docstring=None,
    )


class FakeBM25Index:
    """
    Stand-in for BM25KeywordIndex, avoiding a real BM25Okapi build in
    every test — we're testing ingest.py's caching LOGIC, not BM25 itself
    (already covered by test_keyword_index.py).
    """
    def __init__(self, chunks):
        self.chunks = chunks

    def save_chunks(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([c.model_dump() for c in self.chunks], f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls([CodeChunk(**item) for item in data])


class FakeVectorStore:
    """
    Stand-in for VectorStoreManager — avoids spinning up a real Chroma
    client + embedding model in every ingest.py test.
    """
    def __init__(self, expected_count=0):
        self._count = expected_count
        self.upsert_calls = 0

        class FakeCollection:
            def __init__(self, outer):
                self.outer = outer
            def count(self):
                return self.outer._count

        self.code_collection = FakeCollection(self)

    def upsert_chunks(self, chunks):
        self.upsert_calls += 1
        self._count = len(chunks)


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """
    Redirect ingest.py's module-level path constants into a temp dir,
    so tests never touch the real project's data/ files.
    """
    manifest_path = tmp_path / "manifest.json"
    bm25_path = tmp_path / "bm25_chunks.json"
    graph_path = tmp_path / "graph.json"

    monkeypatch.setattr(ingest_module, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(ingest_module, "BM25_CHUNKS_PATH", bm25_path)
    monkeypatch.setattr(ingest_module, "GRAPH_PATH", graph_path)

    return manifest_path, bm25_path, graph_path


@pytest.fixture
def patched_ingestion(monkeypatch, isolated_paths):
    """
    Replace the real chunking/embedding/graph-building calls with
    lightweight fakes, so tests exercise ingest.py's CACHING logic
    without depending on real repo files on disk or a real embedding
    model — those are already covered by their own component tests.
    """
    test_chunks = [make_chunk("fn_a"), make_chunk("fn_b")]

    monkeypatch.setattr(ingest_module, "chunk_repo", lambda root, repo_name: test_chunks)
    monkeypatch.setattr(ingest_module, "BM25KeywordIndex", FakeBM25Index)
    monkeypatch.setattr(ingest_module, "build_combined_dependency_graph", lambda roots: "FAKE_GRAPH")
    monkeypatch.setattr(ingest_module, "save_graph", lambda graph, path: Path(path).write_text("{}"))
    monkeypatch.setattr(ingest_module, "load_graph", lambda path: "FAKE_GRAPH")

    return test_chunks


class TestLoadOrIngestCorpus:
    def test_no_manifest_triggers_full_ingestion(self, monkeypatch, patched_ingestion, isolated_paths):
        manifest_path, _, _ = isolated_paths
        fake_store = FakeVectorStore(expected_count=0)
        monkeypatch.setattr(ingest_module, "VectorStoreManager", lambda: fake_store)

        chunks, bm25_index, vector_store, graph = load_or_ingest_corpus(
            ["repoA"], [Path("fake/repoA")]
        )

        assert fake_store.upsert_calls == 1
        assert manifest_path.exists()
        assert len(chunks) == 2

    def test_matching_manifest_skips_reingestion(self, monkeypatch, patched_ingestion, isolated_paths):
        manifest_path, bm25_path, graph_path = isolated_paths

        # First call: populates manifest + BM25 + graph on disk
        fake_store_1 = FakeVectorStore(expected_count=0)
        monkeypatch.setattr(ingest_module, "VectorStoreManager", lambda: fake_store_1)
        load_or_ingest_corpus(["repoA"], [Path("fake/repoA")])

        # Second call: fresh VectorStoreManager instance, but count matches
        # what the manifest recorded — should skip re-ingestion entirely.
        fake_store_2 = FakeVectorStore(expected_count=2)  # matches manifest's total_chunks
        monkeypatch.setattr(ingest_module, "VectorStoreManager", lambda: fake_store_2)

        chunks, bm25_index, vector_store, graph = load_or_ingest_corpus(
            ["repoA"], [Path("fake/repoA")]
        )

        assert fake_store_2.upsert_calls == 0  # never re-upserted
        assert len(chunks) == 2

    def test_different_repo_names_triggers_reingestion(self, monkeypatch, patched_ingestion, isolated_paths):
        fake_store_1 = FakeVectorStore(expected_count=0)
        monkeypatch.setattr(ingest_module, "VectorStoreManager", lambda: fake_store_1)
        load_or_ingest_corpus(["repoA"], [Path("fake/repoA")])

        # Different repo list this time — manifest won't match
        fake_store_2 = FakeVectorStore(expected_count=2)
        monkeypatch.setattr(ingest_module, "VectorStoreManager", lambda: fake_store_2)

        load_or_ingest_corpus(["repoA", "repoB"], [Path("fake/repoA"), Path("fake/repoB")])

        assert fake_store_2.upsert_calls == 1  # re-ingested, since repo set changed

    def test_vector_store_count_mismatch_triggers_reingestion(self, monkeypatch, patched_ingestion, isolated_paths):
        """
        Simulates someone manually deleting the Chroma directory but
        leaving the manifest behind — the count check should catch this
        and force re-ingestion rather than silently trusting a stale
        manifest.
        """
        fake_store_1 = FakeVectorStore(expected_count=0)
        monkeypatch.setattr(ingest_module, "VectorStoreManager", lambda: fake_store_1)
        load_or_ingest_corpus(["repoA"], [Path("fake/repoA")])

        # Vector store count doesn't match manifest's recorded total (2)
        fake_store_2 = FakeVectorStore(expected_count=0)
        monkeypatch.setattr(ingest_module, "VectorStoreManager", lambda: fake_store_2)

        load_or_ingest_corpus(["repoA"], [Path("fake/repoA")])

        assert fake_store_2.upsert_calls == 1  # forced re-ingestion due to count mismatch

    def test_force_true_always_reingests(self, monkeypatch, patched_ingestion, isolated_paths):
        fake_store_1 = FakeVectorStore(expected_count=0)
        monkeypatch.setattr(ingest_module, "VectorStoreManager", lambda: fake_store_1)
        load_or_ingest_corpus(["repoA"], [Path("fake/repoA")])

        # Everything matches perfectly, but force=True should re-ingest anyway
        fake_store_2 = FakeVectorStore(expected_count=2)
        monkeypatch.setattr(ingest_module, "VectorStoreManager", lambda: fake_store_2)

        load_or_ingest_corpus(["repoA"], [Path("fake/repoA")], force=True)

        assert fake_store_2.upsert_calls == 1