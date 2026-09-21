import math

import numpy as np
import pytest

from simulation.monte_carlo import MONTHS, run_simulation


def test_returns_three_36_month_paths_with_chart_data() -> None:
    result = run_simulation(50_000, 30_000, 10_000, 100_000, 200_000, iterations=100)

    assert list(result) == ["status_quo", "moderate", "optimal"]
    for path in result.values():
        assert path["horizon_months"] == MONTHS == 36
        assert len(path["monthly_data"]) == 36
        assert path["monthly_data"][0]["month"] == 1
        assert path["monthly_data"][-1]["month"] == 36
        assert {"p10", "median", "p90"} <= path["monthly_data"][0].keys()


def test_same_seed_is_reproducible_and_different_seed_changes_results() -> None:
    first = run_simulation(50_000, 30_000, 10_000, 100_000, 200_000, seed=7, iterations=100)
    repeat = run_simulation(50_000, 30_000, 10_000, 100_000, 200_000, seed=7, iterations=100)
    different = run_simulation(50_000, 30_000, 10_000, 100_000, 200_000, seed=8, iterations=100)

    assert first == repeat
    assert first != different


def test_scenarios_are_not_identical_copies() -> None:
    result = run_simulation(60_000, 30_000, 5_000, 50_000, 150_000, seed=12, iterations=100)

    medians = [path["projected_value"] for path in result.values()]
    assert len(set(medians)) == 3


def test_zero_and_missing_values_are_safe_and_finite() -> None:
    result = run_simulation(
        monthly_income=None,
        monthly_expenses=0,
        monthly_emi=None,
        existing_savings=0,
        existing_debt=None,
        iterations=10,
    )

    for path in result.values():
        assert path["success_prob"] is None
        for value in path["monthly_data"]:
            assert all(math.isfinite(number) for number in value.values())
        assert math.isfinite(path["projected_value"])
        assert math.isfinite(path["p10_value"])
        assert math.isfinite(path["p90_value"])


def test_invalid_and_extremely_large_financial_inputs_do_not_overflow() -> None:
    result = run_simulation(
        monthly_income="invalid",
        monthly_expenses=-1,
        monthly_emi=float("nan"),
        existing_savings=float("inf"),
        existing_debt=1e300,
        goal_amount=-100,
        iterations=5,
    )

    assert all(math.isfinite(path["projected_value"]) for path in result.values())


def test_invalid_simulation_parameters_raise_clear_errors() -> None:
    with pytest.raises(ValueError):
        run_simulation(iterations=0)
    with pytest.raises(ValueError):
        run_simulation(seed="not-an-integer")
    with pytest.raises(TypeError):
        run_simulation(rng=np.random.RandomState(1))
