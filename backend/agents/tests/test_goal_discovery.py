import pytest

from agents import goal_discovery


def make_state(message: object) -> dict:
    return {
        "profile": {"monthlyIncome": 50_000, "monthlyExpenses": 30_000},
        "user_message": message,
    }


class FakeStructuredLLM:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.called = False

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        self.called = True
        if self.error:
            raise self.error
        return self.response


@pytest.mark.asyncio
async def test_extracts_one_goal_with_amount_and_timeline(monkeypatch) -> None:
    fake = FakeStructuredLLM(goal_discovery.GoalExtraction(goals=[
        goal_discovery.ExtractedGoal(
            title="Buy a house",
            target_amount=5_000_000,
            horizon_months=60,
            priority=1,
        )
    ]))
    monkeypatch.setattr(goal_discovery, "llm", fake)

    goals = await goal_discovery.run_goal_discovery(
        make_state("I want to buy a house for 50 lakh in five years.")
    )

    assert goals == [{
        "title": "Buy a house",
        "target_amount": 5_000_000.0,
        "horizon_months": 60,
        "priority": 1,
    }]
    assert fake.called is True


@pytest.mark.asyncio
async def test_preserves_multiple_goals() -> None:
    fake = FakeStructuredLLM(goal_discovery.GoalExtraction(goals=[
        goal_discovery.ExtractedGoal(title="Child education", priority=1),
        goal_discovery.ExtractedGoal(title="Retirement", priority=2),
    ]))
    original = goal_discovery.llm
    goal_discovery.llm = fake
    try:
        goals = await goal_discovery.run_goal_discovery(
            make_state("I need to save for my child's education and retirement.")
        )
    finally:
        goal_discovery.llm = original

    assert [goal["title"] for goal in goals] == ["Child education", "Retirement"]
    assert all(goal["target_amount"] is None for goal in goals)
    assert all(goal["horizon_months"] is None for goal in goals)


@pytest.mark.asyncio
async def test_missing_amount_and_timeline_remain_missing(monkeypatch) -> None:
    fake = FakeStructuredLLM(goal_discovery.GoalExtraction(goals=[
        goal_discovery.ExtractedGoal(title="Pay off my credit card")
    ]))
    monkeypatch.setattr(goal_discovery, "llm", fake)

    goals = await goal_discovery.run_goal_discovery(make_state("I want to pay off my credit card."))

    assert goals == [{
        "title": "Pay off my credit card",
        "target_amount": None,
        "horizon_months": None,
        "priority": 0,
    }]


@pytest.mark.asyncio
async def test_no_goal_conversation_returns_empty_list(monkeypatch) -> None:
    fake = FakeStructuredLLM(goal_discovery.GoalExtraction(goals=[]))
    monkeypatch.setattr(goal_discovery, "llm", fake)

    goals = await goal_discovery.run_goal_discovery(make_state("Can you help me understand my monthly budget?"))

    assert goals == []


@pytest.mark.asyncio
async def test_empty_input_does_not_call_llm(monkeypatch) -> None:
    fake = FakeStructuredLLM(error=AssertionError("empty input must skip the LLM"))
    monkeypatch.setattr(goal_discovery, "llm", fake)

    assert await goal_discovery.run_goal_discovery(make_state("   ")) == []
    assert fake.called is False


@pytest.mark.asyncio
async def test_llm_failure_returns_empty_structured_result(monkeypatch) -> None:
    fake = FakeStructuredLLM(error=RuntimeError("Gemini unavailable"))
    monkeypatch.setattr(goal_discovery, "llm", fake)

    goals = await goal_discovery.run_goal_discovery(make_state("I want to save for a vehicle."))

    assert goals == []


@pytest.mark.asyncio
async def test_malformed_structured_response_returns_empty_list(monkeypatch) -> None:
    fake = FakeStructuredLLM(response={"unexpected": "shape"})
    monkeypatch.setattr(goal_discovery, "llm", fake)

    assert await goal_discovery.run_goal_discovery(make_state("I want to save for a vehicle.")) == []
