import chromadb
import requests
import csv

from chromadb.utils import embedding_functions
from pathlib import Path
from src.config import settings
from src.logging_config import get_logger
from src.logging_config import setup_logging
from src.chunking.ast_parser import CodeChunk
from src.chunking.ast_parser import chunk_repo

logger = get_logger(__name__)


class NaiveBaselineStore:
    """
    A deliberately naive, single-collection Chroma store used as a
    control baseline for the RAG pipeline.

    Unlike VectorStoreManager, this does NOT split code and
    docstrings into separate collections, and callers do NOT get
    keyword search, reranking, or graph augmentation on top of it.
    That's intentional: this class exists to reproduce the failure
    modes of a naive top-k-cosine-similarity RAG implementation, so
    later days (BM25, RRF fusion, graph augmentation, reranking) have
    a concrete, measured baseline to improve against rather than a
    strawman.

    Uses the same local sentence-transformers embedding backend as
    VectorStoreManager (all-MiniLM-L6-v2).
    """
    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name="naive_baseline",
            embedding_function=self.embedding_fn,
        )
        logger.info(
            f"NaiveBaselineStore ready at {self.persist_dir} "
            f"(naive_baseline: {self.collection.count()})"
        )

    def upsert_chunks(self, chunks: list[CodeChunk], batch_size: int = 50) -> None:
        """
        Embed and store code_text for every chunk in the single
        'naive_baseline' collection, processing in batches to bound
        memory usage during local embedding.

        Only code_text is embedded here — no docstring split, no dual
        collections. That's the naive part: a real system needs the
        Day 6 split to keep code syntax and natural-language docs from
        diluting each other's embedding space, but today we want to
        see that problem happen, not solve it in advance.
        """
        total = len(chunks)
        for batch_start in range(0, total, batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            ids, docs, metadatas = [], [], []
            for chunk in batch:
                ids.append(self._chunk_id(chunk))
                docs.append(chunk.code_text)
                metadatas.append(self._chunk_to_metadata(chunk))
            try:
                self.collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
                logger.info(
                    f"Upserted batch {batch_start}-{batch_start + len(batch)} of {total} chunks"
                )
            except Exception as e:
                logger.error(
                    f"Failed to upsert batch {batch_start}-{batch_start + len(batch)}: {e}"
                )

    @staticmethod
    def _chunk_id(chunk: CodeChunk) -> str:
        """
        Build a stable, unique ID for a chunk from its identifying fields.
        Same scheme as VectorStoreManager — deterministic IDs keep
        re-ingestion idempotent instead of duplicating entries.
        """
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
    

def naive_query(store: "NaiveBaselineStore", question: str, top_k: int = 5) -> str:
    """
    Run the naive RAG baseline: embed the question, retrieve top_k
    chunks by cosine similarity ONLY (no BM25, no reranking, no graph
    expansion), stuff them into a single prompt, and call a local
    Ollama model for an answer with NO citation enforcement.

    This is intentionally the weakest possible retrieval strategy —
    it exists to give later days (BM25, RRF fusion, graph augmentation,
    reranking, guardrails) a real, measured baseline to improve on.
    """
    try:
        results = store.collection.query(query_texts=[question], n_results=top_k)
    except Exception as e:
        logger.error(f"Naive query failed during retrieval: {e}")
        return f"[ERROR: retrieval failed: {e}]"

    if not results["ids"][0]:
        logger.warning(f"No chunks retrieved for question: {question!r}")
        return "[No relevant context found in the corpus.]"

    context_blocks = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]
        context_blocks.append(
            f"# {meta['file_path']} :: {meta['name']} (lines {meta['start_line']}-{meta['end_line']})\n{doc}"
        )
    context = "\n\n".join(context_blocks)

    prompt = (
        "You are a helpful assistant answering questions about a Python codebase.\n"
        "Use the following code context to answer the question. If the context "
        "doesn't contain enough information, say so.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )

    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/generate",
            json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        answer = response.json()["response"]
        logger.info(f"Generated answer for question: {question!r}")
        return answer
    except Exception as e:
        logger.error(f"Naive query failed during generation: {e}")
        return f"[ERROR: generation failed: {e}]"
    

BASELINE_QUESTIONS = [
    "Where is authentication middleware defined and what does it check?",
    "How does the rate limiter work and which files use it?",
    "What happens when a payment fails? Trace the full code path.",
    "Find all places add_url_rule is called and explain the contexts.",
    "How does Flask handle routing for a URL?",
]


def run_baseline_eval(store: "NaiveBaselineStore", output_path: str = "eval/baseline_scores.csv") -> None:
    """
    Run the naive baseline against BASELINE_QUESTIONS, print each
    answer for manual review, and write results to a CSV with a blank
    relevance_score column for the user to fill in by hand (1-5).

    This is deliberately a manual-scoring step, not automated: RAGAS
    needs a gold-standard test set that doesn't exist yet, so
    for now the only honest way to evaluate "was this a good answer"
    is a human reading it.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for question in BASELINE_QUESTIONS:
        logger.info(f"Running baseline question: {question!r}")
        answer = naive_query(store, question, top_k=5)
        print(f"\nQUESTION: {question}")
        print(f"ANSWER:\n{answer}\n{'-'*80}")
        rows.append({"question": question, "answer": answer, "relevance_score": ""})

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "relevance_score"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote {len(rows)} baseline results to {output_path}")
    print(f"\nSaved to {output_path} — fill in relevance_score (1-5) for each row by hand.")


if __name__ == "__main__":
    # testing NaiveBaselineStore class
    setup_logging()
    store = NaiveBaselineStore()
    # chunks = chunk_repo(Path("data/repos/flask"), repo_name="flask")  # adjust to your actual repo path
    # store.upsert_chunks(chunks)
    # print(store.collection.count())

    # testing with all repos
    # repo_names = ["flask", "click", "jinja", "itsdangerous", "markupsafe", "werkzeug", "requests", "httpx"]
    # for repo_name in repo_names:
    #     chunks = chunk_repo(Path(f"data/repos/{repo_name}"), repo_name=repo_name)
    #     store.upsert_chunks(chunks)
    #     print(f"{repo_name}: {store.collection.count()} total chunks so far")

    # testing naive_query()
    # test_question = "How does Flask handle routing for a URL?"
    # print(f"\nQUESTION: {test_question}")
    # answer = naive_query(store, test_question, top_k=5)
    # print(f"\nANSWER:\n{answer}")

    # testing run_baseline_eval()
    run_baseline_eval(store)