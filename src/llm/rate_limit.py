"""Client-side rate limiting and a persistent usage budget for hosted LLM APIs.

The limiter keeps calls under per-minute request and token limits by
waiting BEFORE a call, instead of reacting to 429s after the fact. The
budget enforces hard per-run and per-day caps across process restarts, and
refuses a call (raising) rather than ever exceeding them.
"""

import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.logging_config import get_logger

logger = get_logger(__name__)

WINDOW_SECONDS = 60.0


class BudgetExceeded(RuntimeError):
    """Raised instead of making a call that would exceed a configured cap."""


class SlidingWindowRateLimiter:
    """Thread-safe sliding 60 s window over requests and tokens.

    reserve() either records the call and returns 0.0, or records nothing
    and returns how many seconds to wait before trying again. A request
    larger than the whole token allowance is let through only when the
    window is empty, so it never overlaps other traffic.
    """

    def __init__(self, requests_per_minute: int, tokens_per_minute: int,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if requests_per_minute < 1 or tokens_per_minute < 1:
            raise ValueError("rate limits must be positive")
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self._clock = clock
        self._events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def _expire(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= WINDOW_SECONDS:
            self._events.popleft()

    def reserve(self, tokens: int) -> float:
        """Record a call of `tokens` if it fits now; otherwise return the wait in seconds."""
        with self._lock:
            now = self._clock()
            self._expire(now)
            used = sum(t for _, t in self._events)
            fits_requests = len(self._events) < self.rpm
            fits_tokens = used + tokens <= self.tpm or not self._events
            if fits_requests and fits_tokens:
                self._events.append((now, tokens))
                return 0.0
            # Wait until enough of the oldest events have left the window.
            freed, needed_tokens = 0, used + tokens - self.tpm
            for i, (ts, t) in enumerate(self._events):
                freed += t
                enough_requests = len(self._events) - (i + 1) < self.rpm
                if enough_requests and (freed >= needed_tokens or i == len(self._events) - 1):
                    return max(ts + WINDOW_SECONDS - now, 0.001)
            return WINDOW_SECONDS

    def correct(self, estimated: int, actual: int) -> None:
        """Replace the most recent matching estimate with the provider-reported token count."""
        with self._lock:
            for i in range(len(self._events) - 1, -1, -1):
                ts, t = self._events[i]
                if t == estimated:
                    self._events[i] = (ts, actual)
                    return


class UsageBudget:
    """Hard caps on calls per run, and calls and tokens per UTC day.

    Usage is persisted to a JSON file so the daily caps hold across runs.
    reserve() is pessimistic: it counts the call and its estimated tokens
    before the call is made, and raises BudgetExceeded instead of letting
    a call through that could exceed a cap. Once exceeded, every later
    reserve() also raises, so no further calls are made in this run.
    """

    def __init__(self, path: Path, max_calls_per_run: int, max_calls_per_day: int,
                 max_tokens_per_day: int, today: Callable[[], str] | None = None) -> None:
        self.path = path
        self.max_calls_per_run = max_calls_per_run
        self.max_calls_per_day = max_calls_per_day
        self.max_tokens_per_day = max_tokens_per_day
        self._today = today or (lambda: datetime.now(timezone.utc).date().isoformat())
        self._lock = threading.Lock()
        self.run_calls = 0
        self.exhausted = False

    def _load(self) -> dict:
        today = self._today()
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("date") == today:
                    return data
            except (OSError, json.JSONDecodeError) as e:
                # Fail closed: an unreadable usage file must not reset the daily count to zero.
                raise BudgetExceeded(f"Cannot read usage file {self.path}: {e}. Fix or delete it deliberately.") from e
        return {"date": today, "calls": 0, "tokens": 0}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def reserve(self, estimated_tokens: int) -> dict:
        """Count one call before it is made; raise BudgetExceeded if any cap would be passed."""
        with self._lock:
            if self.exhausted:
                raise BudgetExceeded("Budget already exhausted in this run; no further calls")
            data = self._load()
            reason = None
            if self.run_calls + 1 > self.max_calls_per_run:
                reason = f"per-run call cap {self.max_calls_per_run} reached"
            elif data["calls"] + 1 > self.max_calls_per_day:
                reason = f"daily call cap {self.max_calls_per_day} reached ({data['calls']} used today)"
            elif data["tokens"] + estimated_tokens > self.max_tokens_per_day:
                reason = (f"daily token cap {self.max_tokens_per_day} would be exceeded "
                          f"({data['tokens']} used today, ~{estimated_tokens} needed)")
            if reason:
                self.exhausted = True
                raise BudgetExceeded(reason)
            self.run_calls += 1
            data["calls"] += 1
            data["tokens"] += estimated_tokens
            self._save(data)
            return data

    def correct(self, estimated_tokens: int, actual_tokens: int) -> None:
        """Replace a call's estimated token count with the provider-reported one."""
        with self._lock:
            data = self._load()
            data["tokens"] = max(data["tokens"] - estimated_tokens + actual_tokens, 0)
            self._save(data)
