import chromadb

from pathlib import Path
from chromadb.utils import embedding_functions
from src.config import settings
from src.logging_config import get_logger
from src.logging_config import setup_logging
from src.chunking.ast_parser import CodeChunk, chunk_repo

logger = get_logger(__name__)


class VectorStoreManager:
    """
    Manages a persistent Chroma vector store with two separate collections:

    - 'code_chunks': embeds CodeChunk.code_text (raw source code)
    - 'doc_chunks': embeds CodeChunk.docstring / module docstring content

    These are kept in TWO separate collections rather than one mixed
    collection because code syntax and natural-language documentation
    occupy meaningfully different regions of embedding space. Forcing
    one shared collection to represent both equally well degrades
    retrieval precision for both: an identifier-style code search gets
    diluted by comparisons against prose docstrings, and a conceptual
    doc search gets diluted by comparisons against raw syntax. Keeping
    them separate lets each be queried independently, with results
    fused together later during hybrid retrieval.

    Uses chromadb.PersistentClient so the store survives process
    restarts without needing to re-embed anything.
    """

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.client = chromadb.PersistentClient(path=self.persist_dir, settings=chromadb.Settings(anonymized_telemetry=False))

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )

        self.code_collection = self.client.get_or_create_collection(
            name="code_chunks",
            embedding_function=self.embedding_fn,
        )
        self.doc_collection = self.client.get_or_create_collection(
            name="doc_chunks",
            embedding_function=self.embedding_fn,
        )

        logger.info(
            f"VectorStoreManager ready at {self.persist_dir} "
            f"(code_chunks: {self.code_collection.count()}, "
            f"doc_chunks: {self.doc_collection.count()})"
        )

    
    def upsert_chunks(self, chunks: list[CodeChunk], batch_size: int = 50) -> None:
        """
        Embed and store a list of CodeChunks across both collections,
        processing in batches to bound memory usage during local
        embedding (rather than encoding thousands of chunks at once).

        Each chunk's code_text always goes into 'code_chunks'. If a
        chunk has a non-None docstring, that docstring also gets
        embedded separately into 'doc_chunks' — so a function with a
        docstring contributes to BOTH collections, while one without
        only contributes to 'code_chunks'.

        The full CodeChunk is attached as metadata at upsert time (not
        retrieval time), so later filtering/citation doesn't require
        re-parsing the original file.
        """
        total = len(chunks)
        for batch_start in range(0, total, batch_size):
            batch = chunks[batch_start : batch_start + batch_size]

            code_ids, code_docs, code_metadatas = [], [], []
            doc_ids, doc_docs, doc_metadatas = [], [], []

            for chunk in batch:
                chunk_id = self._chunk_id(chunk)
                metadata = self._chunk_to_metadata(chunk)

                code_ids.append(chunk_id)
                code_docs.append(chunk.code_text)
                code_metadatas.append(metadata)

                if chunk.docstring is not None:
                    doc_ids.append(f"{chunk_id}_doc")
                    doc_docs.append(chunk.docstring)
                    doc_metadatas.append(metadata)

            try:
                self.code_collection.upsert(
                    ids=code_ids, documents=code_docs, metadatas=code_metadatas
                )
                if doc_ids:
                    self.doc_collection.upsert(
                        ids=doc_ids, documents=doc_docs, metadatas=doc_metadatas
                    )
                logger.info(
                    f"Upserted batch {batch_start}-{batch_start + len(batch)} of {total} chunks"
                )
            except Exception as e:
                logger.error(
                    f"Failed to upsert batch {batch_start}-{batch_start + len(batch)}: {e}"
                )


    @staticmethod
    def _chunk_id(chunk: CodeChunk) -> str:
        """Build a stable, unique ID for a chunk from its identifying fields."""
        return f"{chunk.repo_name}:{chunk.file_path}:{chunk.name}:{chunk.start_line}"


    @staticmethod
    def _chunk_to_metadata(chunk: CodeChunk) -> dict:
        """Convert a CodeChunk into a flat dict Chroma can store as metadata."""
        return {
            "repo_name": chunk.repo_name,
            "file_path": chunk.file_path,
            "chunk_type": chunk.chunk_type,
            "name": chunk.name,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
        }
    

    def query_code(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Query the code_chunks collection for the top_k most similar
        chunks to the given query string, by embedding similarity
        """
        return self._query_collection(self.code_collection, query, top_k)


    def query_docs(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Query the doc_chunks collection for the top_k most similar
        docstrings to the given query string, by embedding similarity
        """
        return self._query_collection(self.doc_collection, query, top_k)


    @staticmethod
    def _query_collection(collection, query: str, top_k: int) -> list[dict]:
        """
        Run a similarity query against a given collection and
        return results as a flat list of dicts (id, document text,
        metadata, and distance), rather than Chroma's raw nested
        response shape
        """
        try:
            results = collection.query(query_texts=[query], n_results=top_k)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

        flattened = []
        for i in range(len(results["ids"][0])):
            flattened.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return flattened


if __name__ == "__main__":
    # testing VectorStoreManager
    setup_logging()
    vsm = VectorStoreManager()

    # testing upsert_chunks()
    repo_root = Path("data/repos/click")
    chunks = chunk_repo(repo_root, "click")
    # print(f"Chunking produced {len(chunks)} chunks")
    # vsm.upsert_chunks(chunks)
    # print(f"code_chunks count: {vsm.code_collection.count()}")
    # print(f"doc_chunks count: {vsm.doc_collection.count()}")

    # testing query_code()
    # results = vsm.query_code("parse command line arguments", top_k=3)
    # for r in results:
    #     print(r["metadata"]["name"], "-", round(r["distance"], 3))
    #     print(r["document"][:80])
    #     print()

    # testing idempotent
    logger.info(f"Chunking produced {len(chunks)} chunks")
    logger.info(f"BEFORE upsert — code_chunks: {vsm.code_collection.count()}, doc_chunks: {vsm.doc_collection.count()}")

    vsm.upsert_chunks(chunks)

    logger.info(f"AFTER upsert — code_chunks: {vsm.code_collection.count()}, doc_chunks: {vsm.doc_collection.count()}")