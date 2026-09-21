import pytest

from agents import explainability


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
            "annual_income": 600_000,
            "income_volatility": False,
            "savings": 120_000,
            "total_debt": 240_000,
        },
        "risk_score": 68.7,
        "risk_category": "moderate",
        "risk_breakdown": {"debt_score": 60.0},
        "scam_flags": [{
            "pattern": "Fake KYC asks for OTP",
            "source": "01_common_scams.md",
            "confidence": 0.8,
            "evidence": "Banks do not ask for OTP.",
        }],
        "eligible_schemes": [{
            "name": "PMJDY",
            "eligible": True,
            "eligibility_status": "eligible",
            "gap": None,
            "evidence": "Zero balance account.",
            "source_doc_ids": ["01_pmjdy.md"],
        }],
        "goals": [{"title": "Buy a house", "target_amount": 5_000_000, "horizon_months": 60, "priority": 1}],
        "simulation_paths": {
            "status_quo": {"projected_value": 500_000, "p10_value": 400_000, "p90_value": 600_000, "success_prob": 0.2, "horizon_months": 36, "monthly_data": [{}] * 36},
            "moderate": {"projected_value": 600_000, "p10_value": 500_000, "p90_value": 700_000, "success_prob": 0.3, "horizon_months": 36, "monthly_data": [{}] * 36},
            "optimal": {"projected_value": 750_000, "p10_value": 650_000, "p90_value": 850_000, "success_prob": 0.4, "horizon_months": 36, "monthly_data": [{}] * 36},
        },
    }


@pytest.mark.asyncio
async def test_complete_state_is_explained_from_authoritative_snapshot(monkeypatch) -> None:
    fake = FakeStructuredLLM(explainability.ExplanationResponse(cards=[
        explainability.DecisionCard(
            action="Financial snapshot",
            why="The calculated monthly surplus is 10000 and DTI ratio is 0.2.",
            if_ignored="This is a modeled explanation.",
            priority=1,
        )
    ]))
    monkeypatch.setattr(explainability, "llm", fake)

    cards = await explainability.run_explainability(complete_state())

    assert cards[0]["why"] == "The calculated monthly surplus is 10000 and DTI ratio is 0.2."
    assert "10000" in fake.prompt
    assert "68.7" in fake.prompt
    assert "Fake KYC asks for OTP" in fake.prompt
    assert "Buy a house" in fake.prompt
    assert "not predictions" in fake.prompt


@pytest.mark.asyncio
async def test_risk_score_is_not_modified_and_simulation_is_labeled(monkeypatch) -> None:
    fake = FakeStructuredLLM(explainability.ExplanationResponse(cards=[]))
    monkeypatch.setattr(explainability, "llm", fake)
    state = complete_state()
    original_score = state["risk_score"]

    cards = await explainability.run_explainability(state)

    assert state["risk_score"] == original_score
    modeled = next(card for card in cards if card["action"] == "Modeled scenarios")
    assert "not predictions" in modeled["why"]


@pytest.mark.asyncio
async def test_missing_sections_are_omitted_without_fabrication(monkeypatch) -> None:
    fake = FakeStructuredLLM(explainability.ExplanationResponse(cards=[]))
    monkeypatch.setattr(explainability, "llm", fake)

    cards = await explainability.run_explainability({
        "financial_metrics": {"monthly_surplus": 0},
        "risk_score": 0,
        "risk_category": "unknown",
        "scam_flags": [],
        "eligible_schemes": [],
        "goals": [],
        "simulation_paths": {},
    })

    assert [card["action"] for card in cards] == ["Financial snapshot", "Risk assessment"]
    assert "scheme" not in " ".join(card["why"] for card in cards).lower()


@pytest.mark.asyncio
async def test_empty_state_returns_empty_list_without_calling_llm(monkeypatch) -> None:
    fake = FakeStructuredLLM(error=AssertionError("empty state must skip LLM"))
    monkeypatch.setattr(explainability, "llm", fake)

    assert await explainability.run_explainability({}) == []


@pytest.mark.asyncio
async def test_llm_failure_uses_deterministic_factual_fallback(monkeypatch) -> None:
    fake = FakeStructuredLLM(error=RuntimeError("Gemini unavailable"))
    monkeypatch.setattr(explainability, "llm", fake)

    cards = await explainability.run_explainability(complete_state())

    financial = next(card for card in cards if card["action"] == "Financial snapshot")
    assert "10000" in financial["why"]
    assert "0.2" in financial["why"]


@pytest.mark.asyncio
async def test_malformed_llm_output_uses_fallback(monkeypatch) -> None:
    fake = FakeStructuredLLM(response={"wrong": "shape"})
    monkeypatch.setattr(explainability, "llm", fake)

    cards = await explainability.run_explainability(complete_state())

    assert cards
    assert any(card["action"] == "Risk assessment" for card in cards)


def test_explainability_does_not_import_or_call_other_agents() -> None:
    assert not hasattr(explainability, "run_financial_analysis")
    assert not hasattr(explainability, "run_risk_assessment")
    assert not hasattr(explainability, "run_future_simulation")