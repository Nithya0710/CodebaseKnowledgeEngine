import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.eval import groq_judge
from src.eval.groq_judge import estimate_tokens, groq_limits_from_settings, make_budgeted_chat_groq
from src.llm.rate_limit import BudgetExceeded, SlidingWindowRateLimiter, UsageBudget


def fake_result(total_tokens=123):
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))],
                      llm_output={"token_usage": {"total_tokens": total_tokens}})


@pytest.fixture()
def groq_settings(monkeypatch):
    s = groq_judge.settings
    for k, v in {"groq_api_key": "test-key", "groq_judge_model": "test-model", "groq_rpm": 30,
                 "groq_rpd": 1000, "groq_tpm": 12000, "groq_tpd": 100000, "groq_max_output_tokens": 100}.items():
        monkeypatch.setattr(s, k, v)
    return s


@pytest.fixture()
def calls(monkeypatch):
    from langchain_groq import ChatGroq
    made = []

    async def fake_agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        made.append(messages)
        return fake_result()
    monkeypatch.setattr(ChatGroq, "_agenerate", fake_agenerate)
    return made


def make_chat(tmp_path, run_cap=5, tpm=12000):
    budget = UsageBudget(tmp_path / "u.json", max_calls_per_run=run_cap, max_calls_per_day=100,
                         max_tokens_per_day=1_000_000, today=lambda: "2026-10-14")
    limiter = SlidingWindowRateLimiter(30, tpm)
    return make_budgeted_chat_groq(budget, limiter, real_tpm=tpm), budget


class TestEstimateTokens:
    def test_overestimates_and_includes_output_cap(self):
        msgs = [HumanMessage(content="x" * 300)]
        assert estimate_tokens(msgs, max_output_tokens=50) == 100 + 20 + 50


class TestLimitsFromSettings:
    def test_missing_limits_refuse_to_run(self, groq_settings, monkeypatch):
        monkeypatch.setattr(groq_settings, "groq_tpd", None)
        with pytest.raises(RuntimeError, match="GROQ_TPD"):
            groq_limits_from_settings()

    def test_missing_key_refuses_to_run(self, groq_settings, monkeypatch):
        monkeypatch.setattr(groq_settings, "groq_api_key", "")
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            groq_limits_from_settings()

    def test_all_set_returns_limits(self, groq_settings):
        assert groq_limits_from_settings() == (30, 1000, 12000, 100000)


class TestBudgetedChatGroq:
    def test_sdk_retries_disabled_and_single_completion(self, groq_settings, tmp_path):
        chat, _ = make_chat(tmp_path)
        assert chat.max_retries == 0 and chat.n == 1
        assert chat.temperature <= 1e-8  # ChatGroq stores 0 as 1e-8; Groq requires > 0

    def test_calls_are_counted_and_actual_tokens_recorded(self, groq_settings, tmp_path, calls):
        chat, budget = make_chat(tmp_path)
        asyncio.run(chat._agenerate([HumanMessage(content="hello")]))
        assert len(calls) == 1 and budget.run_calls == 1
        assert '"tokens": 123' in (tmp_path / "u.json").read_text()

    def test_budget_cap_blocks_the_api_call(self, groq_settings, tmp_path, calls):
        chat, budget = make_chat(tmp_path, run_cap=1)
        asyncio.run(chat._agenerate([HumanMessage(content="first")]))
        with pytest.raises(BudgetExceeded):
            asyncio.run(chat._agenerate([HumanMessage(content="second")]))
        assert len(calls) == 1  # the refused call never reached the API

    def test_call_larger_than_account_tpm_is_refused(self, groq_settings, tmp_path, calls):
        chat, budget = make_chat(tmp_path, tpm=200)
        with pytest.raises(BudgetExceeded, match="tokens/minute"):
            asyncio.run(chat._agenerate([HumanMessage(content="x" * 3000)]))
        assert calls == [] and budget.exhausted
