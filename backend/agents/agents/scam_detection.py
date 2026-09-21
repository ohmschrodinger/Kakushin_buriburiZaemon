"""Scam Detection Agent - FAISS semantic search on scam_patterns."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TypedDict

from state import ArthSaathiState


logger = logging.getLogger(__name__)
SIMILARITY_THRESHOLD = 1.5


class ScamFlag(TypedDict):
    pattern: str
    source: str
    score: float
    scam_type: str
    confidence: float
    evidence: str
    warning: str


def _input_text(state: ArthSaathiState) -> str:
    suspicious_input = state.get("suspicious_input")
    if isinstance(suspicious_input, str) and suspicious_input.strip():
        return suspicious_input.strip()

    user_message = state.get("user_message")
    if isinstance(user_message, str):
        return user_message.strip()
    return ""


def _scam_type(source: object) -> str:
    if not isinstance(source, str) or not source.strip():
        return "knowledge_base_match"
    return Path(source).stem.removeprefix("01_").removeprefix("02_").removeprefix("03_").removeprefix("04_")


def _confidence(score: float) -> float:
    return round(max(0.0, min(1.0, 1 - (score / 2))), 2)


async def run_scam_detection(state: ArthSaathiState) -> list[ScamFlag]:
    suspicious = _input_text(state)
    if len(suspicious) < 5:
        return []

    try:
        from rag.retriever import query
        results = query(suspicious, store="scam_patterns", top_k=3)
    except FileNotFoundError as error:
        logger.warning("Scam-pattern FAISS index unavailable: %s", error)
        return []
    except Exception as error:
        logger.warning("Scam-pattern retrieval failed: %s", error)
        return []

    flags: list[ScamFlag] = []
    for result in results or []:
        if not isinstance(result, dict):
            continue

        try:
            score = float(result["score"])
            pattern = str(result["text"]).strip()
            source = str(result["source"])
        except (KeyError, TypeError, ValueError):
            continue

        if not math.isfinite(score) or score < 0 or score >= SIMILARITY_THRESHOLD or not pattern:
            continue

        flags.append({
            "pattern": pattern[:200],
            "source": source,
            "score": round(score, 4),
            "scam_type": _scam_type(source),
            "confidence": _confidence(score),
            "evidence": pattern[:500],
            "warning": (
                "This message is similar to a known scam pattern in the "
                f"{source} knowledge-base document. Do not share OTPs, PINs, "
                "passwords, or pay upfront fees; verify through an official channel."
            ),
        })

    return flags
