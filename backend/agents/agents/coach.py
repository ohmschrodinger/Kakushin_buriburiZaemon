"""Coach Agent - structured, state-grounded financial action plans."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from state import ArthSaathiState


load_dotenv()


class ActionItem(BaseModel):
    task: str = Field(description="One practical action grounded in the supplied state")
    daily_amount: float | None = Field(default=None, description="Known INR amount per day, otherwise null")
    deadline: str | None = Field(default=None, description="Today, this week, this month, or null")
    scheme: str | None = Field(default=None, description="Existing scheme name when relevant, otherwise null")


class ActionPlanResponse(BaseModel):
    actions: list[ActionItem] = Field(default_factory=list)


class ActionItemDict(TypedDict):
    task: str
    daily_amount: float | None
    deadline: str | None
    scheme: str | None


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
)


PROMPT = """
You are ArthSaathi's behavioral Coach. Turn only the supplied analysis into a
short, practical, ordered action plan.

Rules:
- Use only facts present in the snapshot. Do not calculate new financial metrics.
- Do not invent income, expenses, debts, savings, goals, schemes, alerts, or amounts.
- Address scam alerts promptly when they exist, but do not claim a scam is guaranteed.
- For schemes, distinguish eligible from potentially eligible and never upgrade eligibility.
- Connect actions to stated goals when available.
- Treat Monte Carlo outputs as modeled scenarios; they are not predictions or guarantees.
- If a required fact is missing, say that verification is needed rather than guessing.
- Do not adapt language or produce a general explanation; create practical tasks only.
- Return actions in priority order, most urgent first. Use 0 for unknown amounts.

Each action must contain exactly:
- task: concise practical action
- daily_amount: an amount only when it is explicitly available or directly present in the snapshot, otherwise null
- deadline: today, this week, this month, or null
- scheme: an existing scheme name when the action concerns one, otherwise null

Return structured data only.

Authoritative analysis snapshot:
{snapshot}
"""


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _display(value: object) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    number = _number(value)
    if number is None:
        return "unknown"
    return str(int(number)) if number.is_integer() else str(number)


def _snapshot(state: ArthSaathiState) -> dict[str, Any]:
    metrics = _mapping(state.get("financial_metrics"))
    risk = {
        "score": state.get("risk_score"),
        "category": state.get("risk_category"),
        "breakdown": dict(_mapping(state.get("risk_breakdown"))),
    }
    scams = state.get("scam_flags") if isinstance(state.get("scam_flags"), list) else []
    schemes = state.get("eligible_schemes") if isinstance(state.get("eligible_schemes"), list) else []
    goals = state.get("goals") if isinstance(state.get("goals"), list) else []
    simulations = _mapping(state.get("simulation_paths"))
    explanations = state.get("decision_cards") if isinstance(state.get("decision_cards"), list) else []

    return {
        "financial_analysis": {
            "monthly_surplus": metrics.get("monthly_surplus"),
            "dti_ratio": metrics.get("dti_ratio"),
            "emergency_months": metrics.get("emergency_months"),
            "savings": metrics.get("savings"),
            "total_debt": metrics.get("total_debt"),
        },
        "risk_assessment": risk,
        "scam_alerts": [
            {
                "pattern": item.get("pattern"),
                "warning": item.get("warning"),
                "confidence": item.get("confidence"),
            }
            for item in scams
            if isinstance(item, Mapping)
        ],
        "scheme_matches": [
            {
                "name": item.get("name"),
                "eligible": item.get("eligible"),
                "eligibility_status": item.get("eligibility_status"),
                "gap": item.get("gap"),
                "how_to_apply": item.get("how_to_apply"),
            }
            for item in schemes
            if isinstance(item, Mapping)
        ],
        "goals": [
            {
                "title": item.get("title"),
                "target_amount": item.get("target_amount"),
                "horizon_months": item.get("horizon_months"),
                "priority": item.get("priority"),
            }
            for item in goals
            if isinstance(item, Mapping)
        ],
        "modeled_simulation": {
            name: {
                "projected_value": path.get("projected_value"),
                "success_prob": path.get("success_prob"),
                "horizon_months": path.get("horizon_months"),
            }
            for name, path in simulations.items()
            if isinstance(path, Mapping)
        },
        "explainability_context": explanations,
    }


def _has_data(snapshot: Mapping[str, Any]) -> bool:
    def meaningful(value: object) -> bool:
        if isinstance(value, Mapping):
            return any(meaningful(item) for item in value.values())
        if isinstance(value, list):
            return bool(value)
        return value is not None

    return meaningful(snapshot)


def _fallback_plan(snapshot: Mapping[str, Any]) -> list[ActionItemDict]:
    actions: list[ActionItemDict] = []
    scams = snapshot.get("scam_alerts")
    risk = _mapping(snapshot.get("risk_assessment"))
    financial = _mapping(snapshot.get("financial_analysis"))
    goals = snapshot.get("goals")
    schemes = snapshot.get("scheme_matches")
    simulations = snapshot.get("modeled_simulation")

    if scams:
        first = _mapping(scams[0])
        actions.append({
            "task": f"Review the existing scam alert: {_display(first.get('pattern'))} and verify any contact through an official channel.",
            "daily_amount": None,
            "deadline": "today",
            "scheme": None,
        })

    category = risk.get("category")
    surplus = _number(financial.get("monthly_surplus"))
    if category in {"high", "moderate"} or (surplus is not None and surplus < 0):
        task = f"Review the recorded {category or 'financial'} risk assessment before taking on new commitments."
        if surplus is not None and surplus < 0:
            task = f"Review the recorded monthly deficit of {_display(abs(surplus))} before setting a savings target."
        actions.append({"task": task, "daily_amount": None, "deadline": "this week", "scheme": None})

    if isinstance(goals, list) and goals:
        goal = next((item for item in goals if isinstance(item, Mapping)), None)
        if goal:
            details = str(goal.get("title") or "the stated goal")
            if goal.get("target_amount") is not None:
                details += f" ({_display(goal.get('target_amount'))} target)"
            actions.append({
                "task": f"Define the next step for the stated goal: {details}.",
                "daily_amount": None,
                "deadline": "this month",
                "scheme": None,
            })

    if isinstance(schemes, list):
        for item in schemes:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name")
            if not name:
                continue
            if item.get("eligible") is True:
                task = f"Review the application steps for the matched scheme: {name}."
            elif item.get("eligibility_status") == "potentially_eligible":
                task = f"Verify the remaining eligibility requirements for {name}."
            else:
                continue
            actions.append({"task": task, "daily_amount": None, "deadline": "this month", "scheme": str(name)})
            break

    if simulations:
        actions.append({
            "task": "Review the modeled simulation scenarios as planning context; they are not predictions or guarantees.",
            "daily_amount": None,
            "deadline": "this month",
            "scheme": None,
        })

    return actions[:5]


def _normalise_response(response: object) -> list[ActionItemDict]:
    if isinstance(response, ActionPlanResponse):
        actions = response.actions
    elif isinstance(response, Mapping):
        actions = ActionPlanResponse.model_validate(response).actions
    else:
        return []

    normalised: list[ActionItemDict] = []
    for action in actions:
        task = action.task.strip()
        if not task:
            continue
        daily_amount = _number(action.daily_amount)
        if daily_amount is not None and daily_amount < 0:
            daily_amount = None
        normalised.append({
            "task": task,
            "daily_amount": daily_amount,
            "deadline": action.deadline.strip() if isinstance(action.deadline, str) and action.deadline.strip() else None,
            "scheme": action.scheme.strip() if isinstance(action.scheme, str) and action.scheme.strip() else None,
        })
    return normalised


async def run_coach(state: ArthSaathiState) -> list[ActionItemDict]:
    snapshot = _snapshot(state)
    if not _has_data(snapshot):
        return []

    try:
        response = await llm.with_structured_output(ActionPlanResponse).ainvoke([
            HumanMessage(content=PROMPT.format(snapshot=json.dumps(snapshot, default=str, indent=2)))
        ])
        actions = _normalise_response(response)
        return actions if actions else _fallback_plan(snapshot)
    except Exception:
        return _fallback_plan(snapshot)