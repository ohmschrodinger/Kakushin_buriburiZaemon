import pytest

from agents import coach


class FakeStructuredLLM:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.prompt = ""

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        self.prompt = messages[0].content
        if self.error:
            raise self.error
        return self.response


def complete_state() -> dict:
    return {
        "financial_metrics": {
            "monthly_surplus": 10_000,
            "dti_ratio": 0.2,
            "emergency_months": 4.0,
            "savings": 120_000,
            "total_debt": 240_000,
        },
        "risk_score": 68.7,
        "risk_category": "moderate",
        "risk_breakdown": {"debt_score": 60.0},
        "scam_flags": [{
            "pattern": "Fake KYC asks for OTP",
            "warning": "Do not share OTP.",
            "confidence": 0.8,
        }],
        "eligible_schemes": [{
            "name": "PMJDY",
            "eligible": True,
            "eligibility_status": "eligible",
            "gap": None,
        }],
        "goals": [{
            "title": "Buy a house",
            "target_amount": 5_000_000,
            "horizon_months": 60,
            "priority": 1,
        }],
        "simulation_paths": {
            "status_quo": {"projected_value": 500_000, "success_prob": 0.2, "horizon_months": 36},
            "moderate": {"projected_value": 600_000, "success_prob": 0.3, "horizon_months": 36},
            "optimal": {"projected_value": 750_000, "success_prob": 0.4, "horizon_months": 36},
        },
        "decision_cards": [{"action": "Risk assessment", "why": "Moderate risk", "if_ignored": "Recorded context", "priority": 1}],
    }


@pytest.mark.asyncio
async def test_complete_state_produces_structured_plan_from_existing_analysis(monkeypatch) -> None:
    fake = FakeStructuredLLM(coach.ActionPlanResponse(actions=[
        coach.ActionItem(task="Review the OTP alert today.", daily_amount=None, deadline="today", scheme=None),
        coach.ActionItem(task="Review PMJDY application steps.", daily_amount=None, deadline="this month", scheme="PMJDY"),
    ]))
    monkeypatch.setattr(coach, "llm", fake)

    actions = await coach.run_coach(complete_state())

    assert actions[0]["task"] == "Review the OTP alert today."
    assert actions[1]["scheme"] == "PMJDY"
    assert "10000" in fake.prompt
    assert "68.7" in fake.prompt
    assert "Fake KYC asks for OTP" in fake.prompt
    assert "Buy a house" in fake.prompt
    assert "not predictions" in fake.prompt


@pytest.mark.asyncio
async def test_coach_does_not_modify_risk_or_analysis_values(monkeypatch) -> None:
    fake = FakeStructuredLLM(coach.ActionPlanResponse(actions=[]))
    monkeypatch.setattr(coach, "llm", fake)
    state = complete_state()
    original = (state["risk_score"], state["financial_metrics"].copy())

    await coach.run_coach(state)

    assert state["risk_score"] == original[0]
    assert state["financial_metrics"] == original[1]


@pytest.mark.asyncio
async def test_llm_failure_uses_state_grounded_fallback_in_priority_order(monkeypatch) -> None:
    fake = FakeStructuredLLM(error=RuntimeError("Gemini unavailable"))
    monkeypatch.setattr(coach, "llm", fake)

    actions = await coach.run_coach(complete_state())

    assert actions[0]["deadline"] == "today"
    assert "Fake KYC" in actions[0]["task"]
    assert any(action["scheme"] == "PMJDY" for action in actions)
    assert any("not predictions" in action["task"] for action in actions)
    assert all(set(action) == {"task", "daily_amount", "deadline", "scheme"} for action in actions)


@pytest.mark.asyncio
async def test_malformed_llm_output_uses_fallback(monkeypatch) -> None:
    fake = FakeStructuredLLM(response={"wrong": "shape"})
    monkeypatch.setattr(coach, "llm", fake)

    actions = await coach.run_coach(complete_state())

    assert actions
    assert any("Buy a house" in action["task"] for action in actions)


@pytest.mark.asyncio
async def test_missing_and_empty_state_are_safe_without_fabrication(monkeypatch) -> None:
    fake = FakeStructuredLLM(error=AssertionError("empty state must skip LLM"))
    monkeypatch.setattr(coach, "llm", fake)

    assert await coach.run_coach({}) == []
    assert await coach.run_coach({"financial_metrics": {"monthly_surplus": None}, "goals": []}) == []


@pytest.mark.asyncio
async def test_missing_optional_fields_do_not_create_amounts_or_schemes(monkeypatch) -> None:
    fake = FakeStructuredLLM(error=RuntimeError("fallback"))
    monkeypatch.setattr(coach, "llm", fake)

    actions = await coach.run_coach({
        "financial_metrics": {"monthly_surplus": None},
        "risk_score": None,
        "risk_category": None,
        "scam_flags": [],
        "eligible_schemes": [],
        "goals": [{"title": "Retirement", "target_amount": None, "horizon_months": None}],
        "simulation_paths": {},
    })

    assert len(actions) == 1
    assert actions[0]["daily_amount"] is None
    assert actions[0]["scheme"] is None
    assert "Retirement" in actions[0]["task"]


def test_coach_does_not_import_other_agents() -> None:
    assert not hasattr(coach, "run_financial_analysis")
    assert not hasattr(coach, "run_risk_assessment")
    assert not hasattr(coach, "run_explainability")