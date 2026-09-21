"""Accessibility Agent - language and readability adaptation only."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from state import ArthSaathiState


load_dotenv()


class AccessibilityResponse(BaseModel):
    text: str = Field(description="Presentation of the supplied facts without changing them")


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.4,
)


LANG_INSTRUCTIONS = {
    "en": "Use simple English, short sections, and clear bullet-like sentences.",
    "hi": "Use simple Hindi with everyday words. Hinglish is acceptable. Keep numbers exactly unchanged.",
    "mr": "Use simple Marathi with everyday words. Keep numbers exactly unchanged.",
    "kn": "Use simple Kannada with everyday words. Keep numbers exactly unchanged.",
}


PROMPT = """
You are ArthSaathi's accessibility and presentation layer.

{language_instruction}

Present the supplied final analysis clearly. This is a translation and
readability task, not a financial-analysis or recommendation task.

Rules:
- Preserve every supplied numerical value exactly; do not recalculate or round.
- Preserve the risk score, scam alerts, scheme eligibility uncertainty, goals,
  and Coach actions and their meaning.
- Keep retrieved information distinct from calculated facts.
- Describe Monte Carlo output as modeled scenarios, never as predictions,
  guarantees, or certain outcomes.
- Do not invent missing values, explanations, actions, schemes, goals, or alerts.
- If a section is missing, omit it or say it is unavailable; do not fill it in.
- Do not add recommendations and do not change action intent.
- Keep the response concise and readable with section labels.

Return structured data containing only the final `text`.

Authoritative state snapshot:
{snapshot}
"""


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _display(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "unknown"
    return str(int(number)) if number.is_integer() else str(number)


def _snapshot(state: ArthSaathiState) -> dict[str, Any]:
    metrics = _mapping(state.get("financial_metrics"))
    return {
        "financial_analysis": dict(metrics),
        "risk_assessment": {
            "risk_score": state.get("risk_score"),
            "risk_category": state.get("risk_category"),
            "risk_breakdown": dict(_mapping(state.get("risk_breakdown"))),
        },
        "scam_flags": state.get("scam_flags") if isinstance(state.get("scam_flags"), list) else [],
        "scheme_matches": state.get("eligible_schemes") if isinstance(state.get("eligible_schemes"), list) else [],
        "goals": state.get("goals") if isinstance(state.get("goals"), list) else [],
        "modeled_simulation": dict(_mapping(state.get("simulation_paths"))),
        "explanation": state.get("decision_cards") if isinstance(state.get("decision_cards"), list) else [],
        "action_plan": state.get("action_plan") if isinstance(state.get("action_plan"), list) else [],
    }


def _has_data(snapshot: Mapping[str, Any]) -> bool:
    def meaningful(value: object) -> bool:
        if isinstance(value, Mapping):
            return any(meaningful(item) for item in value.values())
        if isinstance(value, list):
            return bool(value)
        return value is not None

    return meaningful(snapshot)


def _numeric_facts(snapshot: Mapping[str, Any]) -> list[str]:
    facts: list[str] = []

    def visit(value: object, key: str = "") -> None:
        if isinstance(value, Mapping):
            for name, item in value.items():
                visit(item, str(name))
        elif isinstance(value, list):
            for item in value:
                visit(item, key)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                facts.append(_display(value))

    visit(snapshot)
    return list(dict.fromkeys(facts))


def _preserves_numbers(text: str, snapshot: Mapping[str, Any]) -> bool:
    return all(number in text for number in _numeric_facts(snapshot))


def _fallback_response(snapshot: Mapping[str, Any]) -> str:
    sections: list[str] = []
    financial = _mapping(snapshot.get("financial_analysis"))
    risk = _mapping(snapshot.get("risk_assessment"))

    if financial:
        values = [
            f"Monthly surplus: {_display(financial.get('monthly_surplus'))}",
            f"DTI ratio: {_display(financial.get('dti_ratio'))}",
            f"Emergency runway: {_display(financial.get('emergency_months'))} months",
        ]
        sections.append("Financial analysis\n" + "\n".join(values))

    if risk.get("risk_score") is not None or risk.get("risk_category") is not None:
        sections.append(
            f"Risk assessment\nScore: {_display(risk.get('risk_score'))}/100\n"
            f"Category: {_display(risk.get('risk_category'))}"
        )

    scams = snapshot.get("scam_flags")
    if scams:
        lines = ["Scam alerts"]
        for item in scams:
            if isinstance(item, Mapping):
                lines.append(
                    f"{_display(item.get('pattern'))} | confidence: {_display(item.get('confidence'))} | "
                    f"{_display(item.get('warning'))}"
                )
        sections.append("\n".join(lines))

    schemes = snapshot.get("scheme_matches")
    if schemes:
        lines = ["Scheme information"]
        for item in schemes:
            if isinstance(item, Mapping):
                status = item.get("eligibility_status", item.get("eligible"))
                lines.append(f"{_display(item.get('name'))}: status {_display(status)}; gap: {_display(item.get('gap'))}")
        sections.append("\n".join(lines))

    goals = snapshot.get("goals")
    if goals:
        lines = ["Goals"]
        for item in goals:
            if isinstance(item, Mapping):
                lines.append(
                    f"{_display(item.get('title'))}: amount {_display(item.get('target_amount'))}; "
                    f"timeline {_display(item.get('horizon_months'))} months"
                )
        sections.append("\n".join(lines))

    simulations = snapshot.get("modeled_simulation")
    if simulations:
        lines = ["Modeled scenarios (not predictions)"]
        for name, item in simulations.items():
            if isinstance(item, Mapping):
                lines.append(
                    f"{name}: projected {_display(item.get('projected_value'))}; "
                    f"success probability {_display(item.get('success_prob'))}; "
                    f"horizon {_display(item.get('horizon_months'))} months"
                )
        sections.append("\n".join(lines))

    actions = snapshot.get("action_plan")
    if actions:
        lines = ["Action plan"]
        for item in actions:
            if isinstance(item, Mapping):
                lines.append(
                    f"{_display(item.get('task'))} | deadline: {_display(item.get('deadline'))} | "
                    f"scheme: {_display(item.get('scheme'))}"
                )
        sections.append("\n".join(lines))

    explanation = snapshot.get("explanation")
    if explanation:
        sections.append("Explanation\n" + "\n".join(
            _display(item.get("why")) for item in explanation if isinstance(item, Mapping) and item.get("why")
        ))

    return "\n\n".join(sections)


def _language(state: ArthSaathiState) -> tuple[str, str]:
    profile = _mapping(state.get("profile"))
    requested = profile.get("preferredLang", "en")
    language = requested if isinstance(requested, str) and requested in LANG_INSTRUCTIONS else "en"
    return language, LANG_INSTRUCTIONS[language]


async def run_accessibility(state: ArthSaathiState) -> str:
    snapshot = _snapshot(state)
    if not _has_data(snapshot):
        return ""

    language, language_instruction = _language(state)
    prompt = PROMPT.format(
        language_instruction=language_instruction,
        snapshot=json.dumps(snapshot, default=str, ensure_ascii=False, indent=2),
    )

    try:
        response = await llm.with_structured_output(AccessibilityResponse).ainvoke([
            HumanMessage(content=prompt)
        ])
        if isinstance(response, AccessibilityResponse):
            text = response.text.strip()
        elif isinstance(response, Mapping):
            text = AccessibilityResponse.model_validate(response).text.strip()
        else:
            text = ""
        if text and _preserves_numbers(text, snapshot):
            return text
    except Exception:
        pass

    return _fallback_response(snapshot)