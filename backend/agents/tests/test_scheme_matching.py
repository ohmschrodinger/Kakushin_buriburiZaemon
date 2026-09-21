import pytest

from agents.scheme_matching import run_scheme_matching


def make_state(profile: dict) -> dict:
    return {"profile": profile}


def result_by_name(results: list[dict], name: str) -> dict:
    return next(result for result in results if result["name"] == name)


def fake_retriever(monkeypatch, source_by_name: dict[str, str] | None = None):
    source_by_name = source_by_name or {}

    def query(text: str, store: str, top_k: int) -> list[dict]:
        source = next((value for key, value in source_by_name.items() if key in text), "01_pmjdy.md")
        return [{
            "text": "## How to Apply\nVisit the official bank or CSC with required documents.",
            "source": source,
            "chunk_idx": 0,
            "store": store,
            "score": 0.4,
        }]

    monkeypatch.setattr("rag.retriever.query", query)


@pytest.mark.asyncio
async def test_clear_match_uses_documented_rules_and_retrieved_support(monkeypatch) -> None:
    fake_retriever(monkeypatch, {"PM Jan Dhan": "01_pmjdy.md"})
    results = await run_scheme_matching(make_state({
        "age": 30, "monthlyIncome": 20_000, "employmentType": "gig",
        "hasBankAccount": False, "hasInsurance": True, "isRural": True,
    }))
    pmjdy = result_by_name(results, "PM Jan Dhan Yojana (PMJDY)")
    assert pmjdy["eligible"] is True
    assert pmjdy["eligibility_status"] == "eligible"
    assert pmjdy["source_doc_ids"] == ["01_pmjdy.md"]
    assert "How to Apply" in pmjdy["evidence"]


@pytest.mark.asyncio
async def test_known_disqualifier_is_not_marked_eligible(monkeypatch) -> None:
    fake_retriever(monkeypatch)
    results = await run_scheme_matching(make_state({
        "age": 75, "monthlyIncome": 20_000, "monthlyExpenses": 10_000,
        "employmentType": "salaried", "hasBankAccount": True, "hasInsurance": False,
    }))
    pmsby = result_by_name(results, "PM Suraksha Bima Yojana (PMSBY)")
    assert pmsby["eligible"] is False
    assert pmsby["eligibility_status"] == "not_matching"
    assert "18 and 70" in pmsby["gap"]


@pytest.mark.asyncio
async def test_insufficient_information_is_potential_not_definite(monkeypatch) -> None:
    fake_retriever(monkeypatch)
    results = await run_scheme_matching(make_state({
        "monthlyIncome": 25_000, "employmentType": "gig",
        "hasBankAccount": True, "hasInsurance": False,
    }))
    apy = result_by_name(results, "Atal Pension Yojana (APY)")
    mudra = result_by_name(results, "PM MUDRA Yojana")
    assert apy["eligibility_status"] == "potentially_eligible"
    assert mudra["eligibility_status"] == "potentially_eligible"
    assert apy["eligible"] is False
    assert mudra["eligible"] is False


@pytest.mark.asyncio
async def test_multiple_potential_matches_are_returned(monkeypatch) -> None:
    fake_retriever(monkeypatch)
    results = await run_scheme_matching(make_state({
        "age": 30, "monthlyIncome": 30_000, "employmentType": "self_employed",
        "businessType": "street vendor", "gender": "female",
        "hasBankAccount": True, "hasInsurance": False,
    }))
    names = {result["name"] for result in results}
    assert {"PM MUDRA Yojana", "PM SVANidhi", "Stand Up India"} <= names


@pytest.mark.asyncio
async def test_no_matching_profile_returns_empty_list(monkeypatch) -> None:
    def fail_query(*args, **kwargs):
        raise AssertionError("no candidates should not query FAISS")

    monkeypatch.setattr("rag.retriever.query", fail_query)
    results = await run_scheme_matching(make_state({
        "age": 75, "monthlyIncome": 2_000_000, "employmentType": "unemployed",
        "hasBankAccount": True, "hasInsurance": True, "isRural": False,
    }))
    assert results == []


@pytest.mark.asyncio
async def test_missing_faiss_index_keeps_rule_results_and_logs(monkeypatch, caplog) -> None:
    def missing_index(*args, **kwargs):
        raise FileNotFoundError("government_schemes.faiss")

    monkeypatch.setattr("rag.retriever.query", missing_index)
    with caplog.at_level("WARNING"):
        results = await run_scheme_matching(make_state({
            "age": 30, "monthlyIncome": 20_000, "employmentType": "gig",
            "hasBankAccount": False,
        }))
    assert results
    assert all(result["source_doc_ids"] == [] for result in results)
    assert "FAISS index unavailable" in caplog.text


@pytest.mark.asyncio
async def test_retriever_failure_does_not_crash_matching(monkeypatch, caplog) -> None:
    def failed_query(*args, **kwargs):
        raise RuntimeError("retriever unavailable")

    monkeypatch.setattr("rag.retriever.query", failed_query)
    with caplog.at_level("WARNING"):
        results = await run_scheme_matching(make_state({
            "age": 30, "monthlyIncome": 20_000, "employmentType": "gig",
            "hasBankAccount": False,
        }))
    assert results
    assert "retrieval failed" in caplog.text


@pytest.mark.asyncio
async def test_missing_optional_fields_are_safe(monkeypatch) -> None:
    fake_retriever(monkeypatch)
    results = await run_scheme_matching(make_state({
        "monthlyIncome": "not-a-number", "employmentType": "farmer", "hasBankAccount": True,
    }))
    assert result_by_name(results, "PM Kisan Samman Nidhi")["eligibility_status"] == "potentially_eligible"
    assert all("name" in result and "eligible" in result for result in results)