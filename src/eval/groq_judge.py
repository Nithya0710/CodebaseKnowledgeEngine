"""Groq-hosted RAGAS judge, rate limited and hard-capped to stay inside the free tier.

Every API call goes through two gates before it is sent:
  1. UsageBudget: per-run and per-day call caps and a per-day token cap,
     persisted across runs; a call that could exceed a cap is refused.
  2. SlidingWindowRateLimiter: waits until the call fits the per-minute
     request and token allowance.
Both are sized to a fraction (default 50%) of the account's free-tier
limits, which must be copied from the Groq console into .env; there are no
built-in defaults. The Groq SDK's own retries are disabled so that every
retry passes through (and is counted by) the gates.
"""

import asyncio
import math
import time
from pathlib import Path
from typing import Any

from pydantic import PrivateAttr

from src.config import settings
from src.llm.rate_limit import BudgetExceeded, SlidingWindowRateLimiter, UsageBudget
from src.logging_config import get_logger

logger = get_logger(__name__)

USAGE_PATH = Path(".cache/groq_usage.json")
JUDGE_CACHE_DIR = ".cache/ragas_judge_groq"
# Code tokenizes at roughly 3.5-4 characters per token; dividing by 3
# over-estimates, which errs on the side of staying under the limits.
CHARS_PER_TOKEN = 3


def estimate_tokens(messages: list[Any], max_output_tokens: int) -> int:
    """Conservative token estimate for one call: prompt characters / 3 plus the output cap."""
    chars = sum(len(str(getattr(m, "content", m))) for m in messages)
    return math.ceil(chars / CHARS_PER_TOKEN) + 20 * len(messages) + max_output_tokens


def actual_tokens(result: Any) -> int | None:
    """Total tokens reported by the provider for a ChatResult, if present."""
    usage = (getattr(result, "llm_output", None) or {}).get("token_usage") or {}
    total = usage.get("total_tokens")
    return int(total) if total is not None else None


def _gated_estimate(budget: UsageBudget, limiter: SlidingWindowRateLimiter, messages: list[Any],
                    max_output_tokens: int, real_tpm: int) -> int:
    """Budget check shared by the sync and async paths; returns the reserved estimate."""
    est = estimate_tokens(messages, max_output_tokens)
    if est > real_tpm:
        budget.exhausted = True
        raise BudgetExceeded(f"Single call needs ~{est} tokens, above the account's {real_tpm} tokens/minute")
    usage = budget.reserve(est)
    logger.info(f"Groq call {budget.run_calls}/{budget.max_calls_per_run} this run, "
                f"{usage['calls']}/{budget.max_calls_per_day} today, ~{est} tokens")
    return est


def _record_actual(budget: UsageBudget, limiter: SlidingWindowRateLimiter, est: int, result: Any) -> None:
    actual = actual_tokens(result)
    if actual is not None:
        budget.correct(est, actual)
        limiter.correct(est, actual)


def make_budgeted_chat_groq(budget: UsageBudget, limiter: SlidingWindowRateLimiter, real_tpm: int) -> Any:
    """Build a ChatGroq whose every call passes the budget and the rate limiter."""
    from langchain_groq import ChatGroq

    class BudgetedChatGroq(ChatGroq):
        _budget: UsageBudget = PrivateAttr()
        _limiter: SlidingWindowRateLimiter = PrivateAttr()
        _real_tpm: int = PrivateAttr()

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            est = _gated_estimate(self._budget, self._limiter, messages, self.max_tokens, self._real_tpm)
            while (wait := self._limiter.reserve(est)) > 0:
                logger.info(f"Rate limiter: waiting {wait:.1f}s")
                time.sleep(wait)
            result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            _record_actual(self._budget, self._limiter, est, result)
            return result

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            est = _gated_estimate(self._budget, self._limiter, messages, self.max_tokens, self._real_tpm)
            while (wait := self._limiter.reserve(est)) > 0:
                logger.info(f"Rate limiter: waiting {wait:.1f}s")
                await asyncio.sleep(wait)
            result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            _record_actual(self._budget, self._limiter, est, result)
            return result

    reasoning = {}
    if settings.groq_reasoning_effort:
        reasoning["reasoning_effort"] = settings.groq_reasoning_effort
    if settings.groq_reasoning_format:
        reasoning["reasoning_format"] = settings.groq_reasoning_format
    chat = BudgetedChatGroq(
        model=settings.groq_judge_model,
        api_key=settings.groq_api_key,
        temperature=0,
        max_tokens=settings.groq_max_output_tokens,
        max_retries=0,
        n=1,
        **reasoning,
    )
    chat._budget = budget
    chat._limiter = limiter
    chat._real_tpm = real_tpm
    return chat


def groq_limits_from_settings() -> tuple[int, int, int, int]:
    """Return (rpm, rpd, tpm, tpd) from settings, refusing to run if any is unset."""
    limits = {
        "GROQ_RPM": settings.groq_rpm, "GROQ_RPD": settings.groq_rpd,
        "GROQ_TPM": settings.groq_tpm, "GROQ_TPD": settings.groq_tpd,
    }
    missing = [k for k, v in limits.items() if not v]
    if missing or not settings.groq_api_key or not settings.groq_judge_model:
        raise RuntimeError(
            "Groq judge needs GROQ_API_KEY, GROQ_JUDGE_MODEL and the model's free-tier limits "
            f"(copied from console.groq.com > Settings > Limits) in .env. Missing: "
            f"{missing + [k for k in ('GROQ_API_KEY', 'GROQ_JUDGE_MODEL') if not getattr(settings, k.lower())]}"
        )
    return settings.groq_rpm, settings.groq_rpd, settings.groq_tpm, settings.groq_tpd


def build_groq_chat() -> tuple[Any, UsageBudget]:
    """Build the budgeted Groq chat model and its budget from settings."""
    rpm, rpd, tpm, tpd = groq_limits_from_settings()
    f = settings.groq_limit_fraction
    limiter = SlidingWindowRateLimiter(max(1, int(rpm * f)), max(1, int(tpm * f)))
    budget = UsageBudget(
        USAGE_PATH,
        max_calls_per_run=settings.groq_max_calls_per_run,
        max_calls_per_day=max(1, int(rpd * f)),
        max_tokens_per_day=max(1, int(tpd * f)),
    )
    logger.info(
        f"Groq judge {settings.groq_judge_model}: limiter {limiter.rpm} req/min, {limiter.tpm} tok/min; "
        f"caps {budget.max_calls_per_run} calls/run, {budget.max_calls_per_day} calls/day, "
        f"{budget.max_tokens_per_day} tok/day ({f:.0%} of free tier)"
    )
    return make_budgeted_chat_groq(budget, limiter, real_tpm=tpm), budget
