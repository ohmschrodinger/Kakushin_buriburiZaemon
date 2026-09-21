"""Rule-based government scheme matching with FAISS evidence enrichment."""

from __future__ import annotations

import logging
import math
from typing import TypedDict

from state import ArthSaathiState


logger = logging.getLogger(__name__)
SIMILARITY_THRESHOLD = 1.5


class SchemeMatch(TypedDict, total=False):
    name: str
    eligible: bool
    eligibility_status: str
    gap: str | None
    annual_benefit: float
    how_to_apply: str | None
    source_doc_ids: list[str]
    evidence: str | None


SCHEME_SOURCES = {
    "PM Jan Dhan Yojana (PMJDY)": "01_pmjdy.md",
    "PM MUDRA Yojana": "02_pmmy_mudra.md",
    "Atal Pension Yojana (APY)": "03_apy.md",
    "PM Suraksha Bima Yojana (PMSBY)": "04_pmsby_pmjjby.md",
    "PM Jeevan Jyoti Bima Yojana (PMJJBY)": "04_pmsby_pmjjby.md",
    "PMAY": "05_pmay.md",
    "National Pension System (NPS)": "06_nps_kisan_svanidhi.md",
    "PM Kisan Samman Nidhi": "06_nps_kisan_svanidhi.md",
    "PM SVANidhi": "06_nps_kisan_svanidhi.md",
    "Stand Up India": "06_nps_kisan_svanidhi.md",
    "PM Vishwakarma": "06_nps_kisan_svanidhi.md",
}


def _amount(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    return amount if math.isfinite(amount) and amount >= 0 else None


def _age(profile: dict) -> int | None:
    value = _amount(profile.get("age"))
    return int(value) if value is not None and value.is_integer() and value >= 0 else None


def _text(value: object) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _record(
    name: str,
    status: str,
    gap: str | None,
    annual_benefit: float = 0.0,
) -> SchemeMatch:
    return {
        "name": name,
        "eligible": status == "eligible",
        "eligibility_status": status,
        "gap": gap,
        "annual_benefit": annual_benefit,
        "how_to_apply": None,
        "source_doc_ids": [],
        "evidence": None,
    }


def _matches_business_type(business_type: str, terms: tuple[str, ...]) -> bool:
    return any(term in business_type for term in terms)


def _rule_matches(profile: dict) -> list[SchemeMatch]:
    monthly_income = _amount(profile.get("monthlyIncome"))
    annual_income = monthly_income * 12 if monthly_income is not None else None
    age = _age(profile)
    gender = _text(profile.get("gender"))
    employment = _text(profile.get("employmentType")) or "salaried"
    business_type = _text(profile.get("businessType"))
    caste = _text(profile.get("casteCategory")) or "general"
    bank_account = profile.get("hasBankAccount", True) is True
    has_insurance = profile.get("hasInsurance", False) is True
    rural = profile.get("isRural", False) is True
    land = _amount(profile.get("landHoldingAcres"))
    matches: list[SchemeMatch] = []

    if not bank_account:
        status = "potentially_eligible" if age is None else (
            "eligible" if age >= 10 else "not_matching"
        )
        matches.append(_record(
            "PM Jan Dhan Yojana (PMJDY)",
            status,
            "Confirm the applicant is an Indian citizen aged 10 or above." if age is None else (
                "Age is below the documented minimum of 10 years." if age < 10 else None
            ),
        ))

    if not has_insurance:
        if not bank_account:
            matches.extend([
                _record("PM Suraksha Bima Yojana (PMSBY)", "not_matching", "A savings bank account is required."),
                _record("PM Jeevan Jyoti Bima Yojana (PMJJBY)", "not_matching", "A savings bank account is required."),
            ])
        elif age is None:
            matches.extend([
                _record("PM Suraksha Bima Yojana (PMSBY)", "potentially_eligible", "Confirm age is between 18 and 70 years.", 200000),
                _record("PM Jeevan Jyoti Bima Yojana (PMJJBY)", "potentially_eligible", "Confirm age is between 18 and 50 years.", 200000),
            ])
        else:
            pmsby_ok = 18 <= age <= 70
            pmjjby_ok = 18 <= age <= 50
            matches.extend([
                _record("PM Suraksha Bima Yojana (PMSBY)", "eligible" if pmsby_ok else "not_matching", None if pmsby_ok else "Age must be between 18 and 70 years.", 200000 if pmsby_ok else 0),
                _record("PM Jeevan Jyoti Bima Yojana (PMJJBY)", "eligible" if pmjjby_ok else "not_matching", None if pmjjby_ok else "Age must be between 18 and 50 years.", 200000 if pmjjby_ok else 0),
            ])

    if employment in {"gig", "daily_wage", "self_employed", "farmer"}:
        if age is not None and not 18 <= age <= 40:
            matches.append(_record("Atal Pension Yojana (APY)", "not_matching", "Age must be between 18 and 40 years."))
        elif not bank_account:
            matches.append(_record("Atal Pension Yojana (APY)", "not_matching", "An active savings bank account is required."))
        else:
            matches.append(_record("Atal Pension Yojana (APY)", "potentially_eligible", "Confirm mobile linkage and that the applicant is not an income-tax payer.", 60000))

    if employment in {"self_employed", "gig"}:
        matches.append(_record(
            "PM MUDRA Yojana",
            "eligible" if business_type else "potentially_eligible",
            None if business_type else "Provide the income-generating business activity and documents.",
        ))
    elif employment == "farmer":
        matches.append(_record("PM MUDRA Yojana", "not_matching", "The documented MUDRA criteria exclude farmers; consider farmer-specific schemes."))

    if employment == "farmer":
        if land is None:
            gap = "Confirm land is held in the farmer's name and check documented exclusions."
            matches.append(_record("PM Kisan Samman Nidhi", "potentially_eligible", gap, 6000))
        elif land > 0:
            gap = "Confirm land records, bank/Aadhaar linkage, and exclusions such as income-tax payer status."
            matches.append(_record("PM Kisan Samman Nidhi", "potentially_eligible", gap, 6000))
        else:
            matches.append(_record("PM Kisan Samman Nidhi", "not_matching", "Landholding information does not show a positive holding."))

    if rural:
        matches.append(_record("PMAY-Gramin", "potentially_eligible", "Confirm SECC criteria, housing condition, and local verification.", 120000))
    elif annual_income is None or annual_income <= 1800000:
        subsidy = 0 if annual_income is None else (
            267000 if annual_income < 600000 else
            235000 if annual_income < 1200000 else 230000
        )
        matches.append(_record("PMAY-Urban", "potentially_eligible", "Confirm income category, no pucca house ownership, first-time buyer status, and required co-ownership.", subsidy))

    if employment in {"salaried", "self_employed"}:
        matches.append(_record("National Pension System (NPS)", "potentially_eligible", "NPS is voluntary and market-linked; confirm the desired account and tax situation."))

    if _matches_business_type(business_type, ("street vendor", "vegetable vendor", "fruit vendor")):
        matches.append(_record("PM SVANidhi", "potentially_eligible", "Confirm operation by March 24, 2020 and a vending certificate or ULB letter of recommendation.", 1200))

    if gender == "female" or caste in {"sc", "st"}:
        if age is not None and age <= 18:
            matches.append(_record("Stand Up India", "not_matching", "The applicant must be above 18 years of age."))
        else:
            matches.append(_record("Stand Up India", "potentially_eligible", "Confirm a greenfield enterprise and the required SC/ST or women ownership/shareholding."))

    if _matches_business_type(business_type, ("artisan", "carpenter", "cobbler", "potter", "blacksmith", "weaver", "goldsmith")):
        matches.append(_record("PM Vishwakarma", "potentially_eligible", "Confirm the trade is one of the 18 covered trades and that the applicant is not enrolled in excluded schemes.", 15000))

    return matches


def _application_guidance(text: str) -> str | None:
    lowered = text.lower()
    marker = lowered.find("how to apply")
    if marker < 0:
        marker = lowered.find("how to enroll")
    if marker < 0:
        return None
    guidance = text[marker:].strip()
    for heading in ("\n## ", "\n### "):
        if heading in guidance:
            guidance = guidance.split(heading, 1)[0].strip()
    return guidance[:500]


def _retrieve_support(match: SchemeMatch) -> None:
    expected_source = SCHEME_SOURCES.get(match["name"])
    if expected_source is None:
        return
    try:
        from rag.retriever import query
        results = query(f"{match['name']} eligibility and how to apply", store="government_schemes", top_k=3)
    except FileNotFoundError as error:
        logger.warning("Government-schemes FAISS index unavailable: %s", error)
        return
    except Exception as error:
        logger.warning("Government-schemes retrieval failed: %s", error)
        return

    for result in results or []:
        if not isinstance(result, dict):
            continue
        try:
            score = float(result["score"])
            source = result["source"]
            text = result["text"]
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(score) or score < 0 or score >= SIMILARITY_THRESHOLD:
            continue
        if source != expected_source or not isinstance(text, str) or not text.strip():
            continue
        match["source_doc_ids"] = [source]
        match["evidence"] = text.strip()[:500]
        match["how_to_apply"] = _application_guidance(text)
        return


async def run_scheme_matching(state: ArthSaathiState) -> list[SchemeMatch]:
    profile = state.get("profile")
    if not isinstance(profile, dict):
        return []
    matches = _rule_matches(profile)
    for match in matches:
        _retrieve_support(match)
    return matches