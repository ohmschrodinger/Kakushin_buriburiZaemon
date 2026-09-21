import pytest

from agents.scam_detection import run_scam_detection


def make_state(
    suspicious_input: object = None,
    user_message: object = "",
) -> dict:
    return {
        "profile": {},
        "suspicious_input": suspicious_input,
        "user_message": user_message,
    }


@pytest.mark.asyncio
async def test_scam_related_input_returns_retrieved_evidence(monkeypatch) -> None:
    calls = []

    def fake_query(text: str, store: str, top_k: int) -> list[dict]:
        calls.append((text, store, top_k))
        return [{
            "text": "Fake KYC update asks for OTP and urgent account verification.",
            "source": "01_common_scams.md",
            "chunk_idx": 1,
            "store": "scam_patterns",
            "score": 0.4,
        }]

    monkeypatch.setattr("rag.retriever.query", fake_query)

    flags = await run_scam_detection(
        make_state(suspicious_input="Share your OTP now or your bank account will be blocked")
    )

    assert calls == [(
        "Share your OTP now or your bank account will be blocked",
        "scam_patterns",
        3,
    )]
    assert len(flags) == 1
    assert flags[0]["source"] == "01_common_scams.md"
    assert flags[0]["scam_type"] == "common_scams"
    assert flags[0]["score"] == 0.4
    assert flags[0]["confidence"] == 0.8
    assert "OTP" in flags[0]["evidence"]


@pytest.mark.asyncio
async def test_normal_input_with_no_relevant_result_returns_no_flags(monkeypatch) -> None:
    def fake_query(text: str, store: str, top_k: int) -> list[dict]:
        return [{
            "text": "Known scam pattern document.",
            "source": "02_social_engineering.md",
            "score": 2.2,
        }]

    monkeypatch.setattr("rag.retriever.query", fake_query)

    flags = await run_scam_detection(
        make_state(user_message="I want to understand how to make a monthly budget")
    )

    assert flags == []


@pytest.mark.asyncio
async def test_empty_or_malformed_input_skips_retrieval(monkeypatch) -> None:
    def fail_query(*args, **kwargs):
        raise AssertionError("empty input must not query FAISS")

    monkeypatch.setattr("rag.retriever.query", fail_query)

    assert await run_scam_detection(make_state(suspicious_input="   ")) == []
    assert await run_scam_detection(make_state(suspicious_input=12345)) == []


@pytest.mark.asyncio
async def test_missing_index_returns_no_flags_and_logs_warning(monkeypatch, caplog) -> None:
    def missing_index(*args, **kwargs):
        raise FileNotFoundError("scam_patterns.faiss")

    monkeypatch.setattr("rag.retriever.query", missing_index)

    with caplog.at_level("WARNING"):
        flags = await run_scam_detection(make_state(user_message="I won a lottery prize"))

    assert flags == []
    assert "FAISS index unavailable" in caplog.text


@pytest.mark.asyncio
async def test_retriever_failure_returns_no_flags_and_logs_warning(monkeypatch, caplog) -> None:
    def failed_query(*args, **kwargs):
        raise RuntimeError("embedding service unavailable")

    monkeypatch.setattr("rag.retriever.query", failed_query)

    with caplog.at_level("WARNING"):
        flags = await run_scam_detection(make_state(user_message="Please check this offer"))

    assert flags == []
    assert "retrieval failed" in caplog.text
