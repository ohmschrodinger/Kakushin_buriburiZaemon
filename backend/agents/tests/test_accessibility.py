import pytest

from agents import accessibility


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


def complete_state(language: str = "hi") -> dict:
    return {
        "profile": {"preferredLang": language},
        "financial_metrics": {"monthly_surplus": 10000, "dti_ratio": 0.2, "emergency_months": 4.0},
        "risk_score": 68.7,
        "risk_category": "moderate",
        "scam_flags": [{"pattern": "Fake KYC", "confidence": 0.8, "warning": "Do not share OTP."}],
        "eligible_schemes": [{"name": "PMJDY", "eligible": False, "eligibility_status": "potentially_eligible", "gap": "Confirm documents."}],
        "goals": [{"title": "Buy a house", "target_amount": 5000000, "horizon_months": 60}],
        "simulation_paths": {
            "moderate": {"projected_value": 600000, "success_prob": 0.3, "horizon_months": 36}
        },
        "decision_cards": [{"why": "Calculated explanation."}],
        "action_plan": [{"task": "Verify PMJDY requirements.", "deadline": "this month", "scheme": "PMJDY"}],
    }


@pytest.mark.asyncio
async def test_adapts_requested_language_without_changing_source_facts(monkeypatch) -> None:
    response = accessibility.AccessibilityResponse(text=(
        "Financial analysis: monthly surplus 10000, DTI ratio 0.2, emergency runway 4.0 months. "
        "Risk score 68.7. Fake KYC confidence 0.8. PMJDY is potentially_eligible. "
        "Buy a house target 5000000 in 60 months. Moderate is a modeled scenario, not a prediction: 600000, 0.3, 36 months."
    ))
    fake = FakeStructuredLLM(response=response)
    monkeypatch.setattr(accessibility, "llm", fake)

    text = await accessibility.run_accessibility(complete_state("hi"))

    assert text == response.text
    assert "simple Hindi" in fake.prompt
    assert "10000" in text and "68.7" in text and "5000000" in text
    assert "not a prediction" in text


@pytest.mark.asyncio
async def test_missing_language_defaults_to_english(monkeypatch) -> None:
    response = accessibility.AccessibilityResponse(text=(
        "Surplus 10000, DTI 0.2, runway 4.0, risk 68.7, goal 5000000 in 60 months."
    ))
    fake = FakeStructuredLLM(response=response)
    monkeypatch.setattr(accessibility, "llm", fake)

    await accessibility.run_accessibility({**complete_state("en"), "profile": {}})

    assert "simple English" in fake.prompt


@pytest.mark.asyncio
async def test_missing_explanation_and_actions_is_safe(monkeypatch) -> None:
    response = accessibility.AccessibilityResponse(text="Surplus 10000 and risk 68.7.")
    fake = FakeStructuredLLM(response=response)
    monkeypatch.setattr(accessibility, "llm", fake)
    state = complete_state("en")
    state.pop("decision_cards")
    state.pop("action_plan")

    text = await accessibility.run_accessibility(state)

    assert text != response.text
    assert "Financial analysis" in text
    assert "Action plan" not in text


@pytest.mark.asyncio
async def test_llm_failure_returns_fact_preserving_fallback(monkeypatch) -> None:
    fake = FakeStructuredLLM(error=RuntimeError("Gemini unavailable"))
    monkeypatch.setattr(accessibility, "llm", fake)

    text = await accessibility.run_accessibility(complete_state("en"))

    assert "10000" in text
    assert "68.7" in text
    assert "potentially_eligible" in text
    assert "not predictions" in text
    assert "Verify PMJDY requirements." in text


@pytest.mark.asyncio
async def test_malformed_llm_output_uses_fallback(monkeypatch) -> None:
    fake = FakeStructuredLLM(response={"wrong": "shape"})
    monkeypatch.setattr(accessibility, "llm", fake)

    text = await accessibility.run_accessibility(complete_state("en"))

    assert "Financial analysis" in text
    assert "Risk assessment" in text


@pytest.mark.asyncio
async def test_empty_state_returns_empty_without_calling_llm(monkeypatch) -> None:
    fake = FakeStructuredLLM(error=AssertionError("empty state must skip LLM"))
    monkeypatch.setattr(accessibility, "llm", fake)

    assert await accessibility.run_accessibility({}) == ""


@pytest.mark.asyncio
async def test_missing_numeric_value_falls_back_without_inventing_it(monkeypatch) -> None:
    fake = FakeStructuredLLM(response=accessibility.AccessibilityResponse(text="Risk score unknown."))
    monkeypatch.setattr(accessibility, "llm", fake)

    text = await accessibility.run_accessibility({
        "profile": {"preferredLang": "mr"},
        "risk_score": None,
        "risk_category": None,
        "financial_metrics": {"monthly_surplus": None},
    })

    assert text == ""


def test_accessibility_does_not_import_other_agents() -> None:
    assert not hasattr(accessibility, "run_coach")
    assert not hasattr(accessibility, "run_explainability")
    assert not hasattr(accessibility, "run_financial_analysis")