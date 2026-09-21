import pytest

from agents.financial_analysis import run_financial_analysis


def make_state(**profile: object) -> dict:
    return {"profile": profile}


@pytest.mark.asyncio
async def test_calculates_financial_metrics_from_profile() -> None:
    metrics = await run_financial_analysis(
        make_state(
            monthlyIncome=50_000,
            monthlyExpenses=30_000,
            monthlyEmi=10_000,
            existingSavings=120_000,
            existingDebt=240_000,
            employmentType="salaried",
        )
    )

    assert metrics["dti_ratio"] == 0.2
    assert metrics["monthly_surplus"] == 10_000
    assert metrics["emergency_months"] == 4.0
    assert metrics["annual_income"] == 600_000
    assert metrics["total_debt"] == 240_000
    assert metrics["savings"] == 120_000
    assert metrics["income_volatility"] is False


@pytest.mark.asyncio
async def test_zero_income_and_zero_expenses_are_safe() -> None:
    metrics = await run_financial_analysis(
        make_state(
            monthlyIncome=0,
            monthlyExpenses=0,
            monthlyEmi=0,
            existingSavings=10_000,
        )
    )

    assert metrics["dti_ratio"] == 0.0
    assert metrics["monthly_surplus"] == 0.0
    assert metrics["emergency_months"] == 0.0


@pytest.mark.asyncio
async def test_missing_required_values_are_distinct_from_zero_and_optional_values_default() -> None:
    missing = await run_financial_analysis(make_state(monthlyIncome=50_000))
    zero = await run_financial_analysis(
        make_state(monthlyIncome=50_000, monthlyExpenses=0)
    )

    assert missing["monthly_surplus"] is None
    assert missing["emergency_months"] is None
    assert missing["dti_ratio"] == 0.0
    assert missing["savings"] == 0.0
    assert zero["monthly_surplus"] == 50_000
    assert zero["emergency_months"] == 0.0


@pytest.mark.asyncio
async def test_invalid_values_do_not_raise_or_produce_invalid_metrics() -> None:
    metrics = await run_financial_analysis(
        make_state(
            monthlyIncome="not-a-number",
            monthlyExpenses=-1,
            monthlyEmi="invalid",
            existingSavings=-5,
            existingDebt=float("inf"),
        )
    )

    assert metrics["dti_ratio"] is None
    assert metrics["monthly_surplus"] is None
    assert metrics["emergency_months"] is None
    assert metrics["annual_income"] is None
    assert metrics["savings"] == 0.0
    assert metrics["total_debt"] == 0.0
