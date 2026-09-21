"""Goal Discovery Agent - structured extraction of explicit user goals."""

from __future__ import annotations

import math
import os
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from state import ArthSaathiState

load_dotenv()


class ExtractedGoal(BaseModel):
    title: str = Field(description="Short name preserving the user's stated goal")
    target_amount: float | None = Field(default=None, description="Explicit INR amount, or null")
    horizon_months: int | None = Field(default=None, description="Explicit timeline in months, or null")
    priority: int | None = Field(default=None, description="Explicit priority, or null")


class GoalExtraction(BaseModel):
    goals: list[ExtractedGoal] = Field(default_factory=list)


class Goal(TypedDict):
    title: str
    target_amount: float | None
    horizon_months: int | None
    priority: int


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2,
)


PROMPT = """
You are ArthSaathi's financial goal extraction assistant.

Extract only financial goals explicitly stated in the user's message.
Do not create recommendations, defaults, or goals from the user's income,
employment, expenses, or any other profile field. General statements such as
"manage my money better" are not a specific goal unless the user names one.

For every explicit goal:
- title: short English title preserving the user's meaning
- target_amount: explicit amount in INR, otherwise null
- horizon_months: explicit timeline converted to months, otherwise null
- priority: explicit priority if stated, otherwise null

Return an empty goals list when no explicit financial goal is present.

User message:
{user_message}
"""


def _conversation(state: ArthSaathiState) -> str:
    message = state.get("user_message")
    if isinstance(message, str) and message.strip():
        return message.strip()

    profile = state.get("profile")
    if isinstance(profile, dict):
        profile_message = profile.get("userMessage")
        if isinstance(profile_message, str):
            return profile_message.strip()
    return ""


def _amount(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    return round(amount, 2) if math.isfinite(amount) and amount >= 0 else None


def _months(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        months = float(value)
    except (TypeError, ValueError):
        return None
    return int(months) if math.isfinite(months) and months >= 0 and months.is_integer() else None


def _priority(value: object) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        priority = float(value)
    except (TypeError, ValueError):
        return 0
    return int(priority) if math.isfinite(priority) and priority > 0 and priority.is_integer() else 0


def _normalise_goals(response: object) -> list[Goal]:
    if isinstance(response, GoalExtraction):
        extracted = response.goals
    elif isinstance(response, dict):
        extracted = GoalExtraction.model_validate(response).goals
    else:
        return []

    goals: list[Goal] = []
    for item in extracted:
        title = item.title.strip()
        if not title:
            continue
        goals.append({
            "title": title,
            "target_amount": _amount(item.target_amount),
            "horizon_months": _months(item.horizon_months),
            "priority": _priority(item.priority),
        })
    return goals


async def run_goal_discovery(state: ArthSaathiState) -> list[Goal]:
    user_message = _conversation(state)
    if not user_message:
        return []

    try:
        extractor = llm.with_structured_output(GoalExtraction)
        response = await extractor.ainvoke([
            HumanMessage(content=PROMPT.format(user_message=user_message))
        ])
        return _normalise_goals(response)
    except Exception:
        return []