"""Environment check for a fresh machine: imports, Ollama, and corpus.

Run from the repo root:  python -m scripts.check_env
Exits non-zero if any check fails.
"""

import importlib
import json
import sys
from pathlib import Path

import requests

from src.config import settings
from src.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

REPO_NAMES = ["flask", "click", "jinja", "itsdangerous", "markupsafe", "werkzeug", "requests", "httpx"]
REQUIRED_MODULES = [
    "torch", "scipy", "sklearn", "sentence_transformers", "chromadb",
    "tree_sitter", "tree_sitter_python", "rank_bm25", "networkx", "pydantic_settings",
]
COMMITTED_CHUNKS_PATH = Path("data/bm25_chunks.json")
EXPECTED_GRAPH_NODES = 268


def check_imports() -> bool:
    """Import every required heavy dependency and log its version."""
    ok = True
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError as e:
            logger.error(f"[FAIL] import {name}: {e}")
            ok = False
            continue
        logger.info(f"[OK]   import {name} {getattr(module, '__version__', '')}")
    return ok


def check_ollama() -> bool:
    """Confirm the Ollama server answers and the configured model is pulled."""
    url = f"{settings.ollama_base_url}/api/tags"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"[FAIL] Ollama not reachable at {url}: {e}")
        return False

    models = [m["name"] for m in response.json().get("models", [])]
    if settings.ollama_model not in models:
        logger.error(
            f"[FAIL] Ollama model {settings.ollama_model!r} not pulled "
            f"(available: {models}). Run: ollama pull {settings.ollama_model}"
        )
        return False
    logger.info(f"[OK]   Ollama reachable, model {settings.ollama_model!r} present")
    return True


def _committed_chunk_ids() -> set[str] | None:
    """Read chunk IDs from the committed BM25 cache BEFORE ingestion can overwrite it."""
    if not COMMITTED_CHUNKS_PATH.exists():
        return None
    with open(COMMITTED_CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return {f"{c['repo_name']}:{c['file_path']}:{c['name']}:{c['start_line']}" for c in chunks}


def check_corpus() -> bool:
    """Load (or ingest) the corpus and verify it matches the committed chunk IDs.

    Ground-truth chunk_ids in eval/test_set.json come from the committed
    data/bm25_chunks.json, so the local corpus must reproduce exactly the
    same IDs; any difference means data/repos is not the same code.
    """
    missing = [n for n in REPO_NAMES if not Path(f"data/repos/{n}").is_dir()]
    if missing:
        logger.error(f"[FAIL] missing repos under data/repos/: {missing}")
        return False

    expected_ids = _committed_chunk_ids()

    from src.retrieval.ingest import load_or_ingest_corpus
    from src.retrieval.vector_store import VectorStoreManager

    repo_roots = [Path(f"data/repos/{n}") for n in REPO_NAMES]
    all_chunks, _, _, graph = load_or_ingest_corpus(REPO_NAMES, repo_roots)
    actual_ids = {VectorStoreManager._chunk_id(c) for c in all_chunks}

    logger.info(f"Corpus: {len(all_chunks)} chunks, graph {graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges")
    ok = True
    if graph.number_of_nodes() != EXPECTED_GRAPH_NODES:
        logger.warning(f"[WARN] graph has {graph.number_of_nodes()} nodes, expected {EXPECTED_GRAPH_NODES}")

    if expected_ids is None:
        logger.warning("[WARN] no committed data/bm25_chunks.json to compare against")
    elif actual_ids != expected_ids:
        only_old = sorted(expected_ids - actual_ids)
        only_new = sorted(actual_ids - expected_ids)
        logger.error(
            f"[FAIL] corpus differs from committed chunks: {len(only_old)} IDs missing, "
            f"{len(only_new)} new. First missing: {only_old[:5]}"
        )
        ok = False
    else:
        logger.info(f"[OK]   corpus matches committed chunk IDs ({len(actual_ids)} unique)")
    return ok


def main() -> int:
    """Run all checks; return a process exit code."""
    setup_logging()
    results = {"imports": check_imports(), "ollama": check_ollama()}
    results["corpus"] = check_corpus() if results["imports"] else False
    for name, passed in results.items():
        logger.info(f"{name}: {'PASS' if passed else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
