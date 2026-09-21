"""Risk Assessment Agent - deterministic composite 0-100 health score."""

from __future__ import annotations

import math
from typing import TypedDict

from state import ArthSaathiState


class RiskAssessmentResult(TypedDict):
    risk_score: float
    risk_category: str
    risk_breakdown: dict[str, float]


def _amount(value: object) -> float | None:
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
    return _amount(profile.get(key, 0)) or 0.0


def _metric_or_profile(metrics: dict, key: str, profile_value: object) -> float | None:
    metric = _amount(metrics.get(key))
    return metric if metric is not None else _amount(profile_value)


async def run_risk_assessment(state: ArthSaathiState) -> RiskAssessmentResult:
    profile = state["profile"]
    metrics = state.get("financial_metrics") or {}

    income = _amount(profile.get("monthlyIncome"))
    expenses = _amount(profile.get("monthlyExpenses"))
    emi = _optional_amount(profile, "monthlyEmi")
    savings = _optional_amount(profile, "existingSavings")
    employment_type = profile.get("employmentType", "salaried")

    dti = _metric_or_profile(metrics, "dti_ratio", emi / income if income and income > 0 else None)
    surplus = _metric_or_profile(
        metrics,
        "monthly_surplus",
        income - expenses - emi if income is not None and expenses is not None else None,
    )
    emergency_months = _metric_or_profile(
        metrics,
        "emergency_months",
        savings / expenses if expenses and expenses > 0 else None,
    )

    debt_score = max(0.0, round(100 - ((dti or 0.0) * 200), 1))
    emergency_score = (
        min(100.0, round((emergency_months / 6) * 100, 1))
        if emergency_months is not None
        else 0.0
    )
    income_score = (
        float(
            {
                "salaried": 100,
                "self_employed": 75,
                "gig": 60,
                "farmer": 55,
                "daily_wage": 40,
                "unemployed": 10,
            }.get(employment_type, 60)
        )
        if income is not None and income > 0
        else 0.0
    )
    surplus_score = (
        min(100.0, max(0.0, round((surplus / income) * 200, 1)))
        if surplus is not None and income is not None and income > 0
        else 0.0
    )
    investment_score = (
        min(100.0, round((savings / (income * 3)) * 100, 1))
        if income is not None and income > 0
        else 0.0
    )

    risk_score = round(
        (debt_score      * 0.30) +
        (emergency_score * 0.25) +
        (income_score    * 0.20) +
        (surplus_score   * 0.15) +
        (investment_score * 0.10),
        1,
    )
    risk_score = min(100.0, max(0.0, risk_score))

    category = "low" if risk_score >= 70 else ("moderate" if risk_score >= 40 else "high")

    return {
        "risk_score": risk_score,
        "risk_category": category,
        "risk_breakdown": {
            "debt_score": debt_score,
            "emergency_score": emergency_score,
            "income_score": income_score,
            "surplus_score": surplus_score,
            "investment_score": investment_score,
        },
    }
