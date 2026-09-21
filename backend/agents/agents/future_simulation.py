"""Future Simulation Agent - validated wrapper around the Monte Carlo engine."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from simulation.monte_carlo import MONTHS, run_simulation
from state import ArthSaathiState


SCENARIO_NAMES = ("status_quo", "moderate", "optimal")


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _profile_or_metric(profile: Mapping[str, Any], metrics: Mapping[str, Any], profile_key: str, metric_key: str) -> object:
    value = profile.get(profile_key)
    return value if value is not None else metrics.get(metric_key, 0)


def _goal_amount(goals: object) -> float:
    if not isinstance(goals, list):
        return 0.0

    candidates: list[tuple[int, int, float]] = []
    for index, goal in enumerate(goals):
        if not isinstance(goal, Mapping):
            continue
        amount = _number(goal.get("target_amount", goal.get("targetAmount")))
        if amount is None or amount <= 0:
            continue
        priority = _number(goal.get("priority"))
        priority_value = int(priority) if priority is not None and priority >= 0 else 99
        candidates.append((priority_value, index, amount))

    if not candidates:
        return 0.0
    return min(candidates)[2]


def _valid_simulation(result: object) -> bool:
    if not isinstance(result, Mapping) or set(result) != set(SCENARIO_NAMES):
        return False

    for name in SCENARIO_NAMES:
        path = result.get(name)
        if not isinstance(path, Mapping):
            return False
        monthly_data = path.get("monthly_data")
        if not isinstance(monthly_data, list) or len(monthly_data) != MONTHS:
            return False
        for month in monthly_data:
            if not isinstance(month, Mapping):
                return False
            if not all(_number(month.get(key)) is not None for key in ("month", "p10", "median", "p90")):
                return False
    return True


async def run_future_simulation(state: ArthSaathiState) -> dict:
    profile = state.get("profile")
    if not isinstance(profile, Mapping):
        profile = {}
    metrics = state.get("financial_metrics")
    if not isinstance(metrics, Mapping):
        metrics = {}

    try:
        result = run_simulation(
            monthly_income=_profile_or_metric(profile, metrics, "monthlyIncome", "monthly_income"),
            monthly_expenses=_profile_or_metric(profile, metrics, "monthlyExpenses", "monthly_expenses"),
            monthly_emi=_profile_or_metric(profile, metrics, "monthlyEmi", "monthly_emi"),
            existing_savings=_profile_or_metric(profile, metrics, "existingSavings", "savings"),
            existing_debt=_profile_or_metric(profile, metrics, "existingDebt", "total_debt"),
            risk_appetite=profile.get("riskAppetite", "moderate"),
            goal_amount=_goal_amount(state.get("goals", [])),
        )
    except Exception:
        return {}

    return dict(result) if _valid_simulation(result) else {}