import pytest

from agents import future_simulation


def make_result() -> dict:
    monthly_data = [
        {"month": month, "p10": 10.0, "median": 20.0, "p90": 30.0}
        for month in range(1, 37)
    ]
    return {
        name: {
            "projected_value": 20.0,
            "p10_value": 10.0,
            "p90_value": 30.0,
            "success_prob": 0.5,
            "horizon_months": 36,
            "monthly_data": monthly_data,
        }
        for name in ("status_quo", "moderate", "optimal")
    }


def make_state(**overrides: object) -> dict:
    state = {
        "profile": {
            "monthlyIncome": 50_000,
            "monthlyExpenses": 30_000,
            "monthlyEmi": 5_000,
            "existingSavings": 100_000,
            "existingDebt": 200_000,
            "riskAppetite": "moderate",
        },
        "financial_metrics": {},
        "goals": [],
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_calls_existing_engine_and_preserves_three_paths(monkeypatch) -> None:
    captured = {}

    def fake_simulation(**kwargs):
        captured.update(kwargs)
        return make_result()

    monkeypatch.setattr(future_simulation, "run_simulation", fake_simulation)

    result = await future_simulation.run_future_simulation(make_state())

    assert set(result) == {"status_quo", "moderate", "optimal"}
    assert all(len(path["monthly_data"]) == 36 for path in result.values())
    assert captured == {
        "monthly_income": 50_000,
        "monthly_expenses": 30_000,
        "monthly_emi": 5_000,
        "existing_savings": 100_000,
        "existing_debt": 200_000,
        "risk_appetite": "moderate",
        "goal_amount": 0.0,
    }


@pytest.mark.asyncio
async def test_highest_priority_goal_amount_is_forwarded(monkeypatch) -> None:
    captured = {}

    def fake_simulation(**kwargs):
        captured.update(kwargs)
        return make_result()

    monkeypatch.setattr(future_simulation, "run_simulation", fake_simulation)
    state = make_state(goals=[
        {"title": "Retirement", "target_amount": 2_000_000, "priority": 2},
        {"title": "House", "target_amount": 5_000_000, "priority": 1},
    ])

    await future_simulation.run_future_simulation(state)

    assert captured["goal_amount"] == 5_000_000


@pytest.mark.asyncio
async def test_financial_metrics_fill_missing_optional_profile_values(monkeypatch) -> None:
    captured = {}

    def fake_simulation(**kwargs):
        captured.update(kwargs)
        return make_result()

    monkeypatch.setattr(future_simulation, "run_simulation", fake_simulation)
    state = make_state(
        profile={"monthlyIncome": 40_000, "monthlyExpenses": 20_000},
        financial_metrics={"savings": 75_000, "total_debt": 125_000},
    )

    await future_simulation.run_future_simulation(state)

    assert captured["existing_savings"] == 75_000
    assert captured["existing_debt"] == 125_000
    assert captured["monthly_emi"] == 0


@pytest.mark.asyncio
async def test_missing_goals_and_malformed_values_are_safe(monkeypatch) -> None:
    captured = {}

    def fake_simulation(**kwargs):
        captured.update(kwargs)
        return make_result()

    monkeypatch.setattr(future_simulation, "run_simulation", fake_simulation)
    result = await future_simulation.run_future_simulation(make_state(
        profile={"monthlyIncome": "invalid", "monthlyExpenses": None},
        financial_metrics=None,
        goals=["invalid", {"target_amount": "bad"}],
    ))

    assert set(result) == {"status_quo", "moderate", "optimal"}
    assert captured["goal_amount"] == 0.0


@pytest.mark.asyncio
async def test_engine_errors_return_empty_simulation_paths(monkeypatch) -> None:
    def failed_simulation(**kwargs):
        raise RuntimeError("simulation failed")

    monkeypatch.setattr(future_simulation, "run_simulation", failed_simulation)

    assert await future_simulation.run_future_simulation(make_state()) == {}


@pytest.mark.asyncio
async def test_malformed_engine_output_returns_empty_paths(monkeypatch) -> None:
    monkeypatch.setattr(future_simulation, "run_simulation", lambda **kwargs: {"moderate": {}})

    assert await future_simulation.run_future_simulation(make_state()) == {}


def test_future_simulation_has_no_llm_dependency() -> None:
    assert not hasattr(future_simulation, "llm")