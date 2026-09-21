"""Real Temporal/ledger recovery for the bounded AI post-answer handoff."""

import asyncio
import os
from contextlib import suppress
from types import SimpleNamespace

import pytest
from feature_harness import feature_environment

pytestmark = pytest.mark.integration


async def test_buffer_worker_restart_keeps_masking_and_independent_receipts(store, monkeypatch):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment required")
    from api.background.post_answer import activities, submission
    from api.background.post_answer.workflows import post_answer

    masked, audit_calls, query_calls = [], [], []
    lost_ack = asyncio.Event()

    def mask(value, *, names):
        masked.append((value, names))
        return value.replace("0912345678", "[PHONE]")

    monkeypatch.setattr(activities, "mask_for_trace", mask)

    class Backend:
        async def write_audit(self, record):
            audit_calls.append(record)
            if len(audit_calls) == 1:
                lost_ack.set()
                raise ConnectionError("Receipt lost after durable BE acceptance")
            return "audit-receipt"

        async def log_query(self, **record):
            query_calls.append(record)
            return "query-receipt"

        async def close(self):
            pass

    def build(factory):
        return activities.PostAnswerActivities(factory, Backend).registered()

    async def unexpected_inference(*args):
        pytest.fail("post-answer delivery must not acquire inference capacity")

    limits = {"max_batch_size": 1, "max_delay_seconds": 0, "max_item_bytes": 1_048_576,
              "max_pending_per_tenant": 2, "max_pending_per_partition": 1}
    async with feature_environment(store, name="post-answer", workflows=[post_answer],
            build_activities=build, respond=unexpected_inference, buffering=limits) as env:
        monkeypatch.setattr(submission, "RuntimeClient", lambda url, token, tenant: env.runtime_client(tenant))
        config = SimpleNamespace(url="http://runtime", service_token=SimpleNamespace(get_secret_value=lambda: "x" * 32),
                                 post_answer_accept_timeout_seconds=5)
        source = submission.PostAnswerIntent(request_id="trace-restart", user_id="test", answer="call 0912345678",
            audit={"request_id": "trace-restart", "user_id": "test", "query": "who 0912345678"},
            query_log={"query": "who 0912345678", "language": "vi", "source_document_ids": ["document"]})
        client = env.runtime_client("user:test")
        run_id = None
        try:
            item = await submission.accept(config, source)
            assert item and await submission.accept(config, source) == item
            await asyncio.wait_for(lost_ack.wait(), timeout=30)
            run_id = (await client.get_buffered(item))["run_id"]
            await asyncio.wait_for(env.restart_worker(), timeout=30)
            await asyncio.wait_for(env.temporal.get_workflow_handle(run_id).result(), timeout=30)
            status = await client.get_run(run_id)
            assert status["state"] == "SUCCEEDED"
            assert status["spent"] == status["reserved"] == 0 and env.calls == []
            assert await client.read_json(status["result"]) == {
                "receipts": {"audit": "audit-receipt", "query_log": "query-receipt"}}
            assert len(masked) == 4
            assert len(audit_calls) == 2 and audit_calls[0] == audit_calls[1]
            assert len(query_calls) == 1 and query_calls[0]["request_id"] == source.request_id
            assert "0912345678" not in str(audit_calls + query_calls)
            history = await env.temporal.get_workflow_handle(run_id).fetch_history()
            assert "0912345678" not in str(history)
            await env.replay(run_id)
            assert len(audit_calls) == 2 and len(query_calls) == 1 and len(masked) == 4
        finally:
            if run_id:
                with suppress(Exception):
                    handle = env.temporal.get_workflow_handle(run_id)
                    if (await handle.describe()).close_time is None:
                        await handle.terminate("disposable post-answer test cleanup")
            await client.close()
