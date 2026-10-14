import json

import pytest

from src.llm.rate_limit import BudgetExceeded, SlidingWindowRateLimiter, UsageBudget


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class TestSlidingWindowRateLimiter:
    def test_requests_beyond_rpm_wait_until_oldest_leaves_window(self):
        clock = FakeClock()
        limiter = SlidingWindowRateLimiter(requests_per_minute=2, tokens_per_minute=10_000, clock=clock)
        assert limiter.reserve(10) == 0.0
        clock.t += 5
        assert limiter.reserve(10) == 0.0
        clock.t += 5
        assert limiter.reserve(10) == pytest.approx(50.0)  # first call was 10 s ago
        clock.t += 50
        assert limiter.reserve(10) == 0.0

    def test_tokens_beyond_tpm_wait(self):
        clock = FakeClock()
        limiter = SlidingWindowRateLimiter(requests_per_minute=100, tokens_per_minute=1000, clock=clock)
        assert limiter.reserve(600) == 0.0
        clock.t += 20
        assert limiter.reserve(600) == pytest.approx(40.0)
        clock.t += 40
        assert limiter.reserve(600) == 0.0

    def test_waiting_does_not_record_the_call(self):
        clock = FakeClock()
        limiter = SlidingWindowRateLimiter(requests_per_minute=1, tokens_per_minute=1000, clock=clock)
        limiter.reserve(10)
        for _ in range(5):
            assert limiter.reserve(10) > 0
        clock.t += 60
        assert limiter.reserve(10) == 0.0

    def test_oversized_request_runs_only_in_an_empty_window(self):
        clock = FakeClock()
        limiter = SlidingWindowRateLimiter(requests_per_minute=10, tokens_per_minute=1000, clock=clock)
        assert limiter.reserve(1500) == 0.0  # empty window: allowed alone
        clock.t += 1
        assert limiter.reserve(10) == pytest.approx(59.0)

    def test_correct_replaces_estimate_with_actual(self):
        clock = FakeClock()
        limiter = SlidingWindowRateLimiter(requests_per_minute=100, tokens_per_minute=1000, clock=clock)
        limiter.reserve(900)
        limiter.correct(900, 100)
        assert limiter.reserve(800) == 0.0

    def test_invalid_limits_rejected(self):
        with pytest.raises(ValueError):
            SlidingWindowRateLimiter(requests_per_minute=0, tokens_per_minute=10)


class TestUsageBudget:
    def make(self, tmp_path, run=10, calls=100, tokens=10_000, day="2026-10-14"):
        return UsageBudget(tmp_path / "usage.json", max_calls_per_run=run, max_calls_per_day=calls,
                           max_tokens_per_day=tokens, today=lambda: day)

    def test_per_run_cap_raises_and_stays_exhausted(self, tmp_path):
        budget = self.make(tmp_path, run=2)
        budget.reserve(10)
        budget.reserve(10)
        with pytest.raises(BudgetExceeded, match="per-run"):
            budget.reserve(10)
        assert budget.exhausted
        with pytest.raises(BudgetExceeded, match="already exhausted"):
            budget.reserve(1)

    def test_daily_call_cap_persists_across_runs(self, tmp_path):
        self.make(tmp_path, calls=2).reserve(10)
        second_run = self.make(tmp_path, calls=2)
        second_run.reserve(10)
        with pytest.raises(BudgetExceeded, match="daily call cap"):
            second_run.reserve(10)

    def test_daily_token_cap_refuses_before_the_call(self, tmp_path):
        budget = self.make(tmp_path, tokens=1000)
        budget.reserve(800)
        with pytest.raises(BudgetExceeded, match="daily token cap"):
            budget.reserve(300)

    def test_new_day_resets_daily_counts(self, tmp_path):
        self.make(tmp_path, calls=1, day="2026-10-14").reserve(10)
        self.make(tmp_path, calls=1, day="2026-10-15").reserve(10)

    def test_correct_uses_actual_tokens(self, tmp_path):
        budget = self.make(tmp_path)
        budget.reserve(1000)
        budget.correct(1000, 250)
        assert json.loads((tmp_path / "usage.json").read_text())["tokens"] == 250

    def test_corrupt_usage_file_fails_closed(self, tmp_path):
        (tmp_path / "usage.json").write_text("{not json")
        with pytest.raises(BudgetExceeded, match="Cannot read usage file"):
            self.make(tmp_path).reserve(10)
