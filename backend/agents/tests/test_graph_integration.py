import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import graph
import main
from agents import accessibility, coach, explainability, goal_discovery
from state import FinancialProfileInput


class FakeStructuredLLM:
    def __init__(self, response):
        self.response = response

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        return self.response


def initial_state() -> dict:
    profile = FinancialProfileInput(
        userId="integration-user",
        monthlyIncome=50_000,
        monthlyExpenses=30_000,
        monthlyEmi=10_000,
        existingSavings=120_000,
        existingDebt=240_000,
        age=30,
        employmentType="salaried",
        hasBankAccount=True,
        hasInsurance=False,
        riskAppetite="moderate",
        preferredLang="en",
        userMessage="I want to buy a house in five years.",
        suspiciousInput="",
    ).model_dump()
    return {
        "profile": profile,
        "user_message": profile["userMessage"],
        "suspicious_input": profile["suspiciousInput"],
        "credibility_score": 0.0,
        "credibility_flags": [],
        "financial_metrics": {},
        "risk_score": 0.0,
        "risk_category": "unknown",
        "risk_breakdown": {},
        "scam_flags": [],
        "eligible_schemes": [],
        "goals": [],
        "simulation_paths": {},
        "decision_cards": [],
        "action_plan": [],
        "final_response": "",
    }


@pytest.fixture
def mock_external_agents(monkeypatch):
    monkeypatch.setattr("rag.retriever.query", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        goal_discovery,
        "llm",
        FakeStructuredLLM(goal_discovery.GoalExtraction(goals=[
            goal_discovery.ExtractedGoal(
                title="Buy a house",
                target_amount=5_000_000,
                horizon_months=60,
                priority=1,
            )
        ])),
    )
    monkeypatch.setattr(
        explainability,
        "llm",
        FakeStructuredLLM(explainability.ExplanationResponse(cards=[
            explainability.DecisionCard(
                action="Financial snapshot",
                why="The computed financial results are ready.",
                if_ignored="This is explanatory context.",
                priority=1,
            )
        ])),
    )
    monkeypatch.setattr(
        coach,
        "llm",
        FakeStructuredLLM(coach.ActionPlanResponse(actions=[
            coach.ActionItem(
                task="Review the house goal timeline.",
                daily_amount=None,
                deadline="this month",
                scheme=None,
            )
        ])),
    )
    monkeypatch.setattr(
        accessibility,
        "llm",
        FakeStructuredLLM(accessibility.AccessibilityResponse(
            text="The analysis and action plan are ready."
        )),
    )


@pytest.mark.asyncio
async def test_complete_graph_executes_all_nodes_and_preserves_outputs(mock_external_agents) -> None:
    result = await graph.graph.ainvoke(initial_state())

    assert result["credibility_score"] == 100.0
    assert result["credibility_flags"] == []
    assert result["financial_metrics"]["monthly_surplus"] == 10_000
    assert "dti_ratio" in result["financial_metrics"]
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_category"] in {"low", "moderate", "high"}
    assert isinstance(result["risk_breakdown"], dict)
    assert isinstance(result["scam_flags"], list)
    assert isinstance(result["eligible_schemes"], list)
    assert result["goals"] == [{
        "title": "Buy a house",
        "target_amount": 5_000_000.0,
        "horizon_months": 60,
        "priority": 1,
    }]
    assert set(result["simulation_paths"]) == {"status_quo", "moderate", "optimal"}
    assert all(len(path["monthly_data"]) == 36 for path in result["simulation_paths"].values())
    assert result["decision_cards"][0]["action"] == "Financial snapshot"
    assert result["action_plan"][0]["task"] == "Review the house goal timeline."
    assert "Financial analysis" in result["final_response"]
    assert "Action plan" in result["final_response"]
    assert "10000" in result["final_response"]


@pytest.mark.asyncio
async def test_agent_failure_is_propagated_without_fake_state(mock_external_agents, monkeypatch) -> None:
    async def fail_financial_analysis(state):
        raise RuntimeError("financial analysis unavailable")

    monkeypatch.setattr(graph, "run_financial_analysis", fail_financial_analysis)

    with pytest.raises(RuntimeError, match="financial analysis unavailable"):
        await graph.graph.ainvoke(initial_state())


def test_analyze_endpoint_uses_existing_graph(mock_external_agents, monkeypatch) -> None:
    monkeypatch.setattr(main, "preload_all_stores", lambda: None)

    with TestClient(main.app) as client:
        response = client.post("/analyze", json=initial_state()["profile"])

    assert response.status_code == 200
    body = response.json()
    assert body["userId"] == "integration-user"
    assert body["financialMetrics"]["monthly_surplus"] == 10_000
    assert set(body["simulationPaths"]) == {"status_quo", "moderate", "optimal"}
    assert body["actionPlan"][0]["task"] == "Review the house goal timeline."


def test_invalid_profile_is_rejected_before_graph_execution() -> None:
    with pytest.raises(ValidationError):
        FinancialProfileInput(
            userId="invalid-user",
            monthlyIncome=0,
            monthlyExpenses=30_000,
        )