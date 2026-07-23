import logging
import sys

from src.config import settings

def setup_logging():
    """
    Configure the root logger. Call this explicitly, once, from a 
    real entrypoint (a script's __main__ block, a test fixture, or the
    FastAPI app's startup) — never automatically at import time.
    """
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name) 