"""Review checkpoints through real Temporal/Postgres and the HTTP inference adapter."""

import json
import os

import httpx
import pytest
from feature_harness import feature_environment, peak_children

from intramind_runtime.store import rows

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("unknown", [False, True])
async def test_review_replays_scoped_evidence_and_preserves_unknown_compute(store, monkeypatch, unknown):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment and opt-in required")
    from api.background.common.models import ModelActivities
    from api.background.directive_review.activities import DirectiveReviewActivities
    from api.background.directive_review.leaves import LEAVES
    from api.background.directive_review.policy import DirectiveReviewPolicy
    from api.background.directive_review.workflows import WORKFLOWS
    from api.config import settings
    from tests.background.directive_review.test_planning import CHUNK, completion, verdict

    parsing, searching = [], []

    class Ingestion:
        def parse_document(self, content, filename, timeout):
            parsing.append((content, filename, timeout))
            if len(parsing) == 1:
                raise OSError("Transient parser outage before a result was committed")
            return {"document": {"doc_content": "# Phụ cấp\nTăng phụ cấp.\n# Giờ làm\nGiảm giờ làm."}}

    class Documents:
        def get_chunks_by_document(self, doc_id):
            assert doc_id == "ref-1"
            return [CHUNK]

    class Retriever:
        async def asearch(self, **kwargs):
            assert kwargs["document_ids"] == ["ref-1"]
            searching.append(kwargs)
            return [CHUNK]

    options = {"extractor_concurrency": 1, "assess_concurrency": 1,
               "max_draft_chunks": 1 if unknown else 2,
               "assess_session_timeout_seconds": 10 if unknown else 90,
               "executive_summary": not unknown, "doc_ref_augment_enabled": False,
               "evidence_verifier_enabled": False, "retrieval_quality_gate_enabled": False}
    policy = DirectiveReviewPolicy.capture(settings, "test", options).model_dump(mode="json")

    async def respond(reservation, payload):
        monkeypatch.setattr(settings, "directive_review_assess_concurrency", 16)
        monkeypatch.setattr(settings, "directive_review_executive_summary", False)
        monkeypatch.setattr(settings, "enable_tools", False)
        maximum = payload["max_tokens"]
        if maximum == 2048:
            text = ("Đề xuất tăng phụ cấp kiêm nhiệm lên 30%." if "Tăng phụ cấp" in
                    payload["messages"][-1]["content"] else "Rút ngắn ca làm còn sáu giờ.")
            return completion(json.dumps({"provisions": [{"text": text, "confidence": 0.9}]}))
        if maximum == 768:
            assert payload["parallel_tool_calls"] is True
            if unknown:
                raise httpx.ReadTimeout("Compute continues after the response is lost")
            if any(message["role"] == "tool" for message in payload["messages"]):
                return completion("Đã thu thập đủ căn cứ.")
            return completion(calls=[{"query": "phụ cấp", "mode": "hybrid"}])
        if maximum == 1536:
            return completion(json.dumps(verdict(), ensure_ascii=False))
        assert maximum == 400
        return completion("Hai nội dung cần điều chỉnh theo văn bản tham chiếu.")

    def activities(factory):
        review = DirectiveReviewActivities(factory, model_profile="test", store_factory=Documents,
                                           ingestion_factory=Ingestion, retriever_factory=Retriever)
        return [*review.registered(), ModelActivities(factory, leaves=LEAVES).plan]

    async with feature_environment(
        store, name="directive-review", workflows=WORKFLOWS, build_activities=activities,
        respond=respond, allow_tool_calls=True,
    ) as env:
        client = env.runtime_client("user:test")
        try:
            upload = await client.put_bytes(b"uploaded draft", content_type="application/pdf")
        finally:
            await client.close()
        status, result = await env.submit({"file": upload, "filename": "draft.pdf",
            "reference_document_ids": ["ref-1"], "options": options}, configuration=policy)
        assert len(parsing) == 2
        assert parsing[0] == parsing[1]
        assert len(result["provisions"]) == (1 if unknown else 2)
        assert len({p["provision_id"] for p in result["provisions"]}) == len(result["provisions"])
        assert result["telemetry"]["extraction"]["draft_chunks_truncated"] == int(unknown)
        assert result["refused"] is False
        if unknown:
            assert status["state"] == "PARTIAL"
            assert result["verdicts"][0]["unclear_reason"] == "TIMEOUT"
            async with store.engine.connect() as connection:
                attempts = await rows(connection, "SELECT * FROM runtime_attempts")
                uncertain = [attempt for attempt in attempts if attempt["state"] == "UNKNOWN"]
                assert len(uncertain) == 1 and uncertain[0]["compute_held"]
                assert status["reserved"] > 0
        else:
            assert status["state"] == "SUCCEEDED"
            assert len(searching) == 2 and len(env.calls) == 9
            assert result["summary_counts"]["NEEDS_REVISION"] == 2
            assert "Hai nội dung cần điều chỉnh" in result["report_markdown"]
            assert all(v["citations"][0]["doc_id"] == "ref-1" for v in result["verdicts"])
        before = len(env.calls)
        history = await env.temporal.get_workflow_handle(status["root_id"]).fetch_history()
        assert peak_children(history) == 1
        await env.replay(status["root_id"])
        assert len(env.calls) == before
