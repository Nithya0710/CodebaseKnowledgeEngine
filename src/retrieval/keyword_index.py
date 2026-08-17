import re
import json

from rank_bm25 import BM25Okapi
from pathlib import Path
from src.logging_config import get_logger
from src.chunking.ast_parser import CodeChunk
from src.chunking.ast_parser import chunk_repo


logger = get_logger(__name__)

class BM25KeywordIndex:
    """
    Sparse keyword retrieval index over CodeChunk.code_text, using
    BM25Okapi with an identifier-aware tokenizer (split_identifiers).

    This is a separate, parallel retrieval channel to VectorStoreManager's
    dense embedding search — it does not use embeddings at all.
    BM25 scores chunks by exact sub-word term overlap (weighted by term
    frequency and inverse document frequency), which is what lets it find
    exact identifier matches that dense embeddings can miss.

    The index is built once over the full corpus at construction time.
    BM25Okapi has no incremental-update API — adding new chunks requires
    rebuilding the index from scratch, unlike Chroma's upsert.

    Persistence works by saving the source CodeChunks to disk (JSON) and
    rebuilding the BM25 index from them on load, rather than pickling
    BM25Okapi directly — rank_bm25's internal object structure isn't
    guaranteed stable across versions, so a pickled index could silently
    break after a library upgrade. Rebuilding from source chunks is
    slightly slower on startup but correct by construction every time.
    """

    def __init__(self, chunks: list[CodeChunk]):
        self.chunks: list[CodeChunk] = chunks
        self.bm25 = None

        if not chunks:
            logger.warning("BM25KeywordIndex constructed with an empty chunk list")
            return

        tokenized_corpus = [split_identifiers(chunk.code_text) for chunk in chunks]
        try:
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"Built BM25 index over {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            raise
    
    def query(self, question: str, top_k: int = 5) -> list[tuple[CodeChunk, float]]:
        """
        Score every chunk against the tokenized question using BM25,
        and return the top_k highest-scoring (chunk, score) pairs in
        descending order of relevance.
        """
        if not self.chunks or self.bm25 is None:
            logger.warning("Query attempted on empty BM25 index")
            return []

        tokenized_query = split_identifiers(question)
        try:
            scores = self.bm25.get_scores(tokenized_query)
        except Exception as e:
            logger.error(f"BM25 query failed: {e}")
            return []

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = ranked_indices[:top_k]

        return [(self.chunks[i], scores[i]) for i in top_indices]
    
    def save_chunks(self, path: str) -> None:
        """
        Persist the source CodeChunks (not the BM25Okapi object itself)
        to a JSON file, so the index can be rebuilt on next startup
        without re-running chunk_repo over the raw source files
        """
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            data = [chunk.model_dump() for chunk in self.chunks]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            logger.info(f"Saved {len(self.chunks)} chunks to {path}")
        except Exception as e:
            logger.error(f"Failed to save chunks to {path}: {e}")
            raise

    @classmethod
    def load(cls, path: str) -> "BM25KeywordIndex":
        """
        Load persisted CodeChunks from disk and rebuild the BM25 index
        from them. This re-tokenizes and re-indexes the full corpus on
        every load — slower than a cached index, but avoids any risk of
        a stale or version-incompatible pickled BM25Okapi object.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            chunks = [CodeChunk(**item) for item in data]
            logger.info(f"Loaded {len(chunks)} chunks from {path}, rebuilding BM25 index")
            return cls(chunks)
        except Exception as e:
            logger.error(f"Failed to load chunks from {path}: {e}")
            raise


# Matches sequences of non-alphanumeric characters (whitespace, punctuation, underscores)
# used to split raw text into "words" before camelCase splitting.
_WORD_BOUNDARY_RE = re.compile(r"[^a-zA-Z0-9]+")

# Matches camelCase boundaries within a single word:
#   - lowercase/digit followed by uppercase: "getUser" -> boundary before "U"
#   - uppercase followed by uppercase+lowercase: "HTTPServer" -> boundary before "S" (not each "H","T","T","P")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def split_identifiers(text: str) -> list[str]:
    """
    Tokenize source code text into lowercased sub-word tokens, splitting
    both snake_case (process_payment_result -> process, payment, result)
    and camelCase (getUserById -> get, user, by, id) identifiers into their
    component words.

    This exists because standard whitespace tokenization treats compound
    identifiers as single opaque tokens, so a query like 'get user by id'
    would never match the identifier 'getUserById' via naive string
    comparison. Splitting both the indexed documents AND future queries
    through this same function ensures they share a common vocabulary of
    sub-word tokens, which is what makes BM25 keyword matching actually
    work for source code identifiers.

    Non-alphanumeric characters (whitespace, punctuation, underscores,
    parentheses, etc.) are treated as word boundaries. Empty strings
    resulting from splitting are dropped.
    """
    words = [w for w in _WORD_BOUNDARY_RE.split(text) if w]

    tokens: list[str] = []
    for word in words:
        sub_tokens = _CAMEL_BOUNDARY_RE.split(word)
        tokens.extend(t.lower() for t in sub_tokens if t)

    return tokens


if __name__=='__main__':
    # testing camelCase splitting
    # print(split_identifiers("getUserById")) # expect: ['get', 'user', 'by', 'id']
    # print(split_identifiers("process_payment_result")) # expect: ['process', 'payment', 'result']
    # print(split_identifiers("def add_url_rule(self, rule: str, endpoint: Optional[str] = None):")) # expect something like: ['def', 'add', 'url', 'rule', 'self', 'rule', 'str', 'endpoint', 'optional', 'str', 'none']
    # print(split_identifiers("HTTPSConnectionPool")) # expect: ['https', 'connection', 'pool']

    # testing BM25KeywordIndex class
    chunks = chunk_repo(Path("data/repos/flask"), repo_name="flask")
    index = BM25KeywordIndex(chunks)
    # results = index.query("add_url_rule", top_k=5)
    # for chunk, score in results:
    #     print(f"{score:.4f}  {chunk.name}  ({chunk.file_path}:{chunk.start_line})")

    # testing persistence
    index.save_chunks("data/bm25_chunks.json")
    reloaded = BM25KeywordIndex.load("data/bm25_chunks.json")
    print(f"Reloaded index has {len(reloaded.chunks)} chunks")
    results2 = reloaded.query("add_url_rule", top_k=3)
    for chunk, score in results2:
        print(f"{score:.4f}  {chunk.name}  ({chunk.file_path}:{chunk.start_line})")