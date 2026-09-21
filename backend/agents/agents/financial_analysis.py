"""Financial Analysis Agent - deterministic financial calculations."""

from __future__ import annotations

import math
from typing import TypedDict

from state import ArthSaathiState


class FinancialMetrics(TypedDict):
    monthly_surplus: float | None
    dti_ratio: float | None
    emergency_months: float | None
    annual_income: float | None
    income_volatility: bool
    total_debt: float
    savings: float


def _amount(value: object) -> float | None:
    """Return a finite non-negative amount, or None when it is unusable."""
    if value is None or isinstance(value, bool):
        return None

    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(amount) or amount < 0:
        return None
    return amount


def _optional_amount(profile: dict, key: str) -> float:
    """Apply the zero defaults declared for optional financial fields."""
    return _amount(profile.get(key, 0)) or 0.0


async def run_financial_analysis(state: ArthSaathiState) -> FinancialMetrics:
    p = state["profile"]

    income = _amount(p.get("monthlyIncome"))
    expenses = _amount(p.get("monthlyExpenses"))
    emi = _optional_amount(p, "monthlyEmi")
    savings = _optional_amount(p, "existingSavings")
    debt = _optional_amount(p, "existingDebt")

    monthly_surplus = (
        round(income - expenses - emi, 2)
        if income is not None and expenses is not None
        else None
    )
    dti_ratio = round(emi / income, 4) if income is not None and income > 0 else (
        0.0 if income == 0 else None
    )
    emergency_months = (
        round(savings / expenses, 2)
        if expenses is not None and expenses > 0
        else (0.0 if expenses == 0 else None)
    )
    annual_income = round(income * 12, 2) if income is not None else None
    income_volatile = p.get("employmentType") in {"gig", "farmer", "daily_wage"}

    return {
        "monthly_surplus": monthly_surplus,
        "dti_ratio": dti_ratio,
        "emergency_months": emergency_months,
        "annual_income": annual_income,
        "income_volatility": income_volatile,
        "total_debt": debt,
        "savings": savings,
    }
