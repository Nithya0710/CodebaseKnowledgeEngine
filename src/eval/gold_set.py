"""Loading and subsetting the gold-standard test set (eval/test_set.json)."""

import json
from pathlib import Path

from pydantic import BaseModel

DEFAULT_TEST_SET_PATH = Path("eval/test_set.json")

# The five questions used since the Day 5 naive baseline. The Ollama-vs-Gemini
# generation decision is made on these before the set is expanded to 24.
BASELINE5_QUESTIONS: tuple[str, ...] = (
    "Where is authentication middleware defined and what does it check?",
    "How does the rate limiter work and which files use it?",
    "What happens when a payment fails? Trace the full code path.",
    "Find all places add_url_rule is called and explain the contexts.",
    "How does Flask handle routing for a URL?",
)

SUBSETS: dict[str, tuple[str, ...]] = {"baseline5": BASELINE5_QUESTIONS}


class GoldEntry(BaseModel):
    """One gold-standard question with its reference answer and chunks."""
    question: str
    ground_truth_answer: str
    ground_truth_context_chunk_ids: list[str]


def load_gold_set(path: Path = DEFAULT_TEST_SET_PATH, subset: str | None = None) -> list[GoldEntry]:
    """Load the gold set, optionally restricted to a named subset.

    Subset entries are returned in the subset's own order. A subset question
    missing from the file is an error rather than a silent skip, since a
    typo would otherwise shrink the evaluation without anyone noticing.
    """
    with open(path, "r", encoding="utf-8") as f:
        entries = [GoldEntry.model_validate(e) for e in json.load(f)]

    if subset is None:
        return entries
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; known subsets: {sorted(SUBSETS)}")

    by_question = {e.question: e for e in entries}
    missing = [q for q in SUBSETS[subset] if q not in by_question]
    if missing:
        raise ValueError(f"Subset {subset!r} questions missing from {path}: {missing}")
    return [by_question[q] for q in SUBSETS[subset]]
