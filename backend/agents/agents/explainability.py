"""Explainability Agent - LLM narration of already-computed results."""

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


class DecisionCard(BaseModel):
    action: str = Field(description="Short factual section title")
    why: str = Field(description="Factual explanation using only supplied data")
    if_ignored: str = Field(description="Clarification or limitation, not a new recommendation")
    priority: int = Field(default=0, ge=0)


class ExplanationResponse(BaseModel):
    cards: list[DecisionCard] = Field(default_factory=list)


class DecisionCardDict(TypedDict):
    action: str
    why: str
    if_ignored: str
    priority: int


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
)


PROMPT = """
You are ArthSaathi's Explainability Agent. Narrate computed financial results
clearly and factually for the user.

Use only the supplied snapshot. Do not recalculate, alter, or question any
number. Do not create financial advice, action plans, goals, schemes, scam
patterns, or simulation outcomes. Do not add facts that are not present.

Label information accurately:
- calculated facts come from Financial Analysis or Risk Assessment
- retrieved information comes from scam or scheme evidence
- modeled scenarios come from the Monte Carlo simulation and are not predictions
- absent fields are unknown and must remain unknown

Create one factual Decision Card for each available section. Omit sections
with no data. Keep the existing card fields: action, why, if_ignored, priority.
The `action` field is only a short section title; do not turn it into a task or
recommendation. Return structured data only.

Authoritative snapshot:
{snapshot}
"""


def _display(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return str(value).lower()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "unknown"
    return str(int(number)) if number.is_integer() else str(number)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _snapshot(state: ArthSaathiState) -> dict[str, Any]:
    metrics = _mapping(state.get("financial_metrics"))
    risk_breakdown = _mapping(state.get("risk_breakdown"))
    scams = state.get("scam_flags") if isinstance(state.get("scam_flags"), list) else []
    schemes = state.get("eligible_schemes") if isinstance(state.get("eligible_schemes"), list) else []
    goals = state.get("goals") if isinstance(state.get("goals"), list) else []
    simulations = _mapping(state.get("simulation_paths"))

    return {
        "calculated_financial_analysis": {
            "monthly_surplus": metrics.get("monthly_surplus"),
            "dti_ratio": metrics.get("dti_ratio"),
            "emergency_months": metrics.get("emergency_months"),
            "annual_income": metrics.get("annual_income"),
            "income_volatility": metrics.get("income_volatility"),
            "savings": metrics.get("savings"),
            "total_debt": metrics.get("total_debt"),
        },
        "calculated_risk_assessment": {
            "risk_score": state.get("risk_score"),
            "risk_category": state.get("risk_category"),
            "risk_breakdown": dict(risk_breakdown),
        },
        "retrieved_scam_information": [
            {
                "pattern": scam.get("pattern"),
                "source": scam.get("source"),
                "confidence": scam.get("confidence"),
                "evidence": scam.get("evidence"),
            }
            for scam in scams
            if isinstance(scam, Mapping)
        ],
        "retrieved_scheme_information": [
            {
                "name": scheme.get("name"),
                "eligibility_status": scheme.get("eligibility_status"),
                "eligible": scheme.get("eligible"),
                "gap": scheme.get("gap"),
                "evidence": scheme.get("evidence"),
                "source_doc_ids": scheme.get("source_doc_ids"),
            }
            for scheme in schemes
            if isinstance(scheme, Mapping)
        ],
        "stated_goals": [
            {
                "title": goal.get("title"),
                "target_amount": goal.get("target_amount"),
                "horizon_months": goal.get("horizon_months"),
                "priority": goal.get("priority"),
            }
            for goal in goals
            if isinstance(goal, Mapping)
        ],
        "modeled_simulation_scenarios": {
            name: {
                "projected_value": path.get("projected_value"),
                "p10_value": path.get("p10_value"),
                "p90_value": path.get("p90_value"),
                "success_prob": path.get("success_prob"),
                "horizon_months": path.get("horizon_months"),
                "monthly_points": len(path.get("monthly_data", [])) if isinstance(path.get("monthly_data"), list) else 0,
            }
            for name, path in simulations.items()
            if isinstance(path, Mapping)
        },
    }


def _has_data(snapshot: Mapping[str, Any]) -> bool:
    def meaningful(value: object) -> bool:
        if isinstance(value, Mapping):
            return any(meaningful(item) for item in value.values())
        if isinstance(value, list):
            return bool(value)
        return value is not None

    return meaningful(snapshot)


def _fallback_cards(snapshot: Mapping[str, Any]) -> list[DecisionCardDict]:
    cards: list[DecisionCardDict] = []
    financial = _mapping(snapshot.get("calculated_financial_analysis"))
    risk = _mapping(snapshot.get("calculated_risk_assessment"))
    simulations = _mapping(snapshot.get("modeled_simulation_scenarios"))

    if any(value is not None for value in financial.values()):
        cards.append({
            "action": "Financial snapshot",
            "why": (
                f"Calculated values: monthly surplus {_display(financial.get('monthly_surplus'))}, "
                f"DTI ratio {_display(financial.get('dti_ratio'))}, and emergency runway "
                f"{_display(financial.get('emergency_months'))} months."
            ),
            "if_ignored": "This card reports the available calculations and does not add a recommendation.",
            "priority": 1,
        })

    if risk.get("risk_score") is not None or risk.get("risk_category") is not None:
        cards.append({
            "action": "Risk assessment",
            "why": f"The calculated risk score is {_display(risk.get('risk_score'))}/100 and the category is {_display(risk.get('risk_category'))}.",
            "if_ignored": "The score and category are reproduced from Risk Assessment without modification.",
            "priority": 2,
        })

    scams = snapshot.get("retrieved_scam_information")
    if scams:
        cards.append({
            "action": "Scam information",
            "why": f"Retrieved scam-pattern evidence is available for {_display(len(scams))} alert(s); it comes from the scam knowledge base.",
            "if_ignored": "No additional scam determination is made beyond the retrieved evidence.",
            "priority": 3,
        })

    schemes = snapshot.get("retrieved_scheme_information")
    if schemes:
        cards.append({
            "action": "Scheme information",
            "why": f"Retrieved government-scheme information is available for {_display(len(schemes))} scheme record(s). Unknown eligibility conditions remain unknown.",
            "if_ignored": "This card reports matched scheme information without adding eligibility claims.",
            "priority": 4,
        })

    goals = snapshot.get("stated_goals")
    if goals:
        cards.append({
            "action": "Stated goals",
            "why": f"Goal Discovery found {_display(len(goals))} goal(s) explicitly stated by the user.",
            "if_ignored": "Missing goal amounts or timelines remain unknown rather than being estimated.",
            "priority": 5,
        })

    if simulations:
        cards.append({
            "action": "Modeled scenarios",
            "why": f"Monte Carlo provides {_display(len(simulations))} modeled scenario(s) with monthly points; these are scenarios, not predictions of the actual future.",
            "if_ignored": "The simulation does not establish what will actually happen.",
            "priority": 6,
        })

    return cards


def _normalise_response(response: object) -> list[DecisionCardDict]:
    if isinstance(response, ExplanationResponse):
        cards = response.cards
    elif isinstance(response, Mapping):
        cards = ExplanationResponse.model_validate(response).cards
    else:
        return []

    return [
        {
            "action": card.action.strip(),
            "why": card.why.strip(),
            "if_ignored": card.if_ignored.strip(),
            "priority": card.priority,
        }
        for card in cards
        if card.action.strip() and card.why.strip() and card.if_ignored.strip()
    ]


async def run_explainability(state: ArthSaathiState) -> list[DecisionCardDict]:
    snapshot = _snapshot(state)
    if not _has_data(snapshot):
        return []

    try:
        response = await llm.with_structured_output(ExplanationResponse).ainvoke([
            HumanMessage(content=PROMPT.format(snapshot=json.dumps(snapshot, default=str, indent=2)))
        ])
        cards = _normalise_response(response)
        return cards if cards else _fallback_cards(snapshot)
    except Exception:
        return _fallback_cards(snapshot)