import pytest

from agents.risk_assessment import run_risk_assessment


def make_state(profile: dict, financial_metrics: dict | None = None) -> dict:
    return {
        "profile": profile,
        "financial_metrics": financial_metrics or {},
    }


@pytest.mark.asyncio
async def test_normal_profile_uses_financial_analysis_metrics() -> None:
    result = await run_risk_assessment(
        make_state(
            {
                "monthlyIncome": 50_000,
                "monthlyExpenses": 30_000,
                "monthlyEmi": 10_000,
                "existingSavings": 120_000,
                "employmentType": "salaried",
            },
            {
                "dti_ratio": 0.2,
                "monthly_surplus": 10_000,
                "emergency_months": 4,
            },
        )
    )

    assert result["risk_score"] == 68.7
    assert result["risk_category"] == "moderate"
    assert result["risk_breakdown"] == {
        "debt_score": 60.0,
        "emergency_score": 66.7,
        "income_score": 100.0,
        "surplus_score": 40.0,
        "investment_score": 80.0,
    }


@pytest.mark.asyncio
async def test_healthy_profile_is_low_risk() -> None:
    result = await run_risk_assessment(
        make_state(
            {
                "monthlyIncome": 100_000,
                "monthlyExpenses": 20_000,
                "monthlyEmi": 5_000,
                "existingSavings": 1_000_000,
                "employmentType": "salaried",
            }
        )
    )

    assert result["risk_score"] == 97.0
    assert result["risk_category"] == "low"


@pytest.mark.asyncio
async def test_high_obligations_and_unstable_income_are_high_risk() -> None:
    result = await run_risk_assessment(
        make_state(
            {
                "monthlyIncome": 20_000,
                "monthlyExpenses": 18_000,
                "monthlyEmi": 10_000,
                "existingSavings": 0,
                "employmentType": "daily_wage",
            }
        )
    )

    assert result["risk_score"] == 8.0
    assert result["risk_category"] == "high"


@pytest.mark.asyncio
async def test_zero_income_and_missing_values_are_safe() -> None:
    zero_income = await run_risk_assessment(
        make_state(
            {
                "monthlyIncome": 0,
                "monthlyExpenses": 0,
                "monthlyEmi": 0,
                "existingSavings": 0,
                "employmentType": "salaried",
            }
        )
    )
    missing = await run_risk_assessment(make_state({}))

    assert zero_income["risk_score"] == 30.0
    assert missing["risk_score"] == 30.0
    assert zero_income["risk_category"] == "high"
    assert 0 <= missing["risk_score"] <= 100


@pytest.mark.asyncio
async def test_invalid_values_and_boundary_scores_stay_in_range() -> None:
    invalid = await run_risk_assessment(
        make_state(
            {
                "monthlyIncome": "invalid",
                "monthlyExpenses": -1,
                "monthlyEmi": float("inf"),
                "existingSavings": -5,
                "employmentType": "unknown",
            }
        )
    )
    boundary = await run_risk_assessment(
        make_state(
            {
                "monthlyIncome": 1,
                "monthlyExpenses": 0,
                "monthlyEmi": 1_000_000,
                "existingSavings": 0,
                "employmentType": "unemployed",
            }
        )
    )

    assert 0 <= invalid["risk_score"] <= 100
    assert 0 <= boundary["risk_score"] <= 100
    assert all(0 <= score <= 100 for score in invalid["risk_breakdown"].values())
    assert all(0 <= score <= 100 for score in boundary["risk_breakdown"].values())
