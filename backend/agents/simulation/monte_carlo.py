"""NumPy Monte Carlo engine for 36-month financial scenario projections."""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np


MONTHS = 36
ITERATIONS = 1000
INFLATION = 0.06 / 12
RANDOM_SEED = 42
MAX_AMOUNT = 1e15
MAX_ITERATIONS = 10000

RETURN_RATES = {
    "conservative": 0.07 / 12,
    "moderate": 0.10 / 12,
    "aggressive": 0.12 / 12,
}

VOLATILITY = {
    "conservative": 0.008,
    "moderate": 0.015,
    "aggressive": 0.025,
}

PATH_CONFIGS = {
    "status_quo": {"expense_cut": 0.00, "extra_debt_payment": 0.00},
    "moderate": {"expense_cut": 0.05, "extra_debt_payment": 0.10},
    "optimal": {"expense_cut": 0.10, "extra_debt_payment": 0.30},
}


class MonthlyData(TypedDict):
    month: int
    p10: float
    median: float
    p90: float


class SimulationPath(TypedDict):
    projected_value: float
    p10_value: float
    p90_value: float
    success_prob: float | None
    horizon_months: int
    monthly_data: list[MonthlyData]


class SimulationResult(TypedDict):
    status_quo: SimulationPath
    moderate: SimulationPath
    optimal: SimulationPath


def _amount(value: object) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(amount) or amount < 0:
        return 0.0
    return min(amount, MAX_AMOUNT)


def _validated_iterations(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("iterations must be an integer")
    if value < 1 or value > MAX_ITERATIONS:
        raise ValueError(f"iterations must be between 1 and {MAX_ITERATIONS}")
    return int(value)


def _rng(seed: object, generator: np.random.Generator | None) -> np.random.Generator:
    if generator is not None:
        if not isinstance(generator, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        return generator
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))):
        raise ValueError("seed must be an integer or None")
    return np.random.default_rng(seed)


def _safe_round(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(max(0.0, value), 2)


def run_simulation(
    monthly_income: float = 0,
    monthly_expenses: float = 0,
    monthly_emi: float = 0,
    existing_savings: float = 0,
    existing_debt: float = 0,
    risk_appetite: str = "moderate",
    goal_amount: float = 0,
    *,
    seed: int | None = RANDOM_SEED,
    iterations: int = ITERATIONS,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """Simulate status quo, moderate, and optimal 36-month trajectories.

    Each iteration tracks savings after investment returns, income, expenses,
    and debt service. The scenario adjustments are limited to expense cuts and
    additional debt repayment; they are modeling assumptions, not advice.
    """
    iteration_count = _validated_iterations(iterations)
    random = _rng(seed, rng)

    income = _amount(monthly_income)
    expenses = _amount(monthly_expenses)
    emi = _amount(monthly_emi)
    savings = _amount(existing_savings)
    debt = _amount(existing_debt)
    target = _amount(goal_amount)
    appetite = risk_appetite if risk_appetite in RETURN_RATES else "moderate"
    return_rate = RETURN_RATES[appetite]
    volatility = VOLATILITY[appetite]

    def simulate_path(expense_cut: float, extra_debt_payment: float) -> SimulationPath:
        balances = np.full(iteration_count, savings, dtype=np.float64)
        debts = np.full(iteration_count, debt, dtype=np.float64)
        monthly_balances: list[np.ndarray] = []

        for month in range(MONTHS):
            income_shock = np.maximum(0.0, 1.0 + random.normal(0.0, volatility, iteration_count))
            expense_shock = np.maximum(0.0, 1.0 + random.normal(0.0, volatility / 2, iteration_count))
            monthly_income_value = income * income_shock
            monthly_expenses_value = (
                expenses * ((1.0 + INFLATION) ** month) * expense_shock * (1.0 - expense_cut)
            )

            investment_return = np.maximum(0.0, 1.0 + random.normal(return_rate, volatility, iteration_count))
            balances = np.maximum(0.0, balances * investment_return)

            regular_debt_payment = np.minimum(debts, emi)
            base_cash_flow = monthly_income_value - monthly_expenses_value - emi
            extra_payment = np.minimum(
                np.maximum(0.0, debts - regular_debt_payment),
                np.maximum(0.0, base_cash_flow) * extra_debt_payment,
            )
            debt_payment = regular_debt_payment + extra_payment
            cash_flow = monthly_income_value - monthly_expenses_value - debt_payment

            balances = np.maximum(0.0, balances + cash_flow)
            debts = np.maximum(0.0, debts - debt_payment)
            monthly_balances.append(np.nan_to_num(balances, nan=0.0, posinf=MAX_AMOUNT, neginf=0.0))

        final_balances = monthly_balances[-1]
        monthly_data: list[MonthlyData] = []
        for index, values in enumerate(monthly_balances, start=1):
            monthly_data.append({
                "month": index,
                "p10": _safe_round(float(np.percentile(values, 10))),
                "median": _safe_round(float(np.percentile(values, 50))),
                "p90": _safe_round(float(np.percentile(values, 90))),
            })

        success_prob = None
        if target > 0:
            success_prob = round(float(np.mean(final_balances >= target)), 4)

        return {
            "projected_value": _safe_round(float(np.percentile(final_balances, 50))),
            "p10_value": _safe_round(float(np.percentile(final_balances, 10))),
            "p90_value": _safe_round(float(np.percentile(final_balances, 90))),
            "success_prob": success_prob,
            "horizon_months": MONTHS,
            "monthly_data": monthly_data,
        }

    return {
        name: simulate_path(**configuration)
        for name, configuration in PATH_CONFIGS.items()
    }