"""Qualify native PPTX model leaves through the ledger; not a full deck workflow."""

import os
from pathlib import Path

import pytest
from feature_harness import feature_environment
from pptx_workflows import checkpoint_models

from intramind_runtime.sdk import durable_activity

pytestmark = pytest.mark.integration


async def test_pptx_native_prompts_checkpoint_publication_retry_and_temporal_replay(store, monkeypatch):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment required")
    import llmai
    from api.background.common.models import ModelActivities
    from api.background.pptx.leaves import LEAVES

    def forbidden(*args, **kwargs):
        raise AssertionError("Durable PPTX must not construct a native inference client")

    monkeypatch.setattr(llmai, "get_client", forbidden)
    requests, publication_attempts = [], []
    accepted = {"model": "test", "context_tokens": 16384,
                "parameters": {"DISABLE_THINKING": "true"}}
    defaults = {"temperature": 0.2, "max_tokens": 2000}
    fixture = Path(__file__).resolve().parents[2] / "mta-ai-intramind/tests/tools/fixtures/pptx-brief-4848c757.json"

    async def respond(reservation, payload):
        assert reservation.operation.attempt_timeout_seconds == 37
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        requests.append(payload)
        monkeypatch.setenv("CUSTOM_MODEL", "model-changed-on-worker")
        monkeypatch.setenv("LLM_CONTEXT_LENGTH", "99999")
        if len(requests) == 1:
            assert "Toàn bộ tài liệu" in payload["messages"][-1]["content"]
            answer = "# Báo cáo\nNội dung đã đọc."
        else:
            assert payload["response_format"]["json_schema"]["name"] == "presentation_brief"
            assert "Nội dung đã đọc." in payload["messages"][-1]["content"]
            answer = fixture.read_text(encoding="utf-8")
        return {"choices": [{"message": {"content": answer}}]}

    def activities(factory):
        def spec(leaf, arguments):
            return {"leaf": leaf, "model_profile": "test", "capacity_profile_id": "test-v1",
                    "defaults": defaults, "attempt_timeout_seconds": 37,
                    "arguments": {"generation": accepted, **arguments}}

        @durable_activity(name="test.pptx.prepare/v1")
        async def prepare(inputs):
            client = factory(inputs["tenant_id"])
            try:
                source = await client.read_json(inputs["input"])
                return await client.put_json(spec("pptx.condense/v1", {
                    "sections": [source["content"]], "target_tokens": 900}))
            finally:
                await client.close()

        @durable_activity(name="test.pptx.brief-input/v1")
        async def brief_input(inputs):
            client = factory(inputs["tenant_id"])
            try:
                condensed = (await client.read_json(inputs["input"]))["value"]
                return await client.put_json(spec("pptx.brief/v1", {
                    "distilled_context": "\n\n".join(condensed), "heading_skeleton": "# Báo cáo",
                    "instructions": "6 slide", "preferences": {"exact_slide_count": 6},
                    "key_points_context": ""}))
            finally:
                await client.close()

        @durable_activity(name="test.pptx.publish/v1")
        async def publish(inputs):
            publication_attempts.append(inputs["input"])
            if len(publication_attempts) == 1:
                raise OSError("Lost publication acknowledgement after inference")
            return inputs["input"]

        return [prepare, brief_input, publish, ModelActivities(factory, leaves=LEAVES).plan]

    async with feature_environment(store, name="pptx-model-checkpoint", workflows=[checkpoint_models],
                                   build_activities=activities, respond=respond) as env:
        status, result = await env.submit({"content": "# Báo cáo\nToàn bộ tài liệu."})
        assert status["state"] == "SUCCEEDED"
        assert len(requests) == len(env.calls) == 2
        assert len(publication_attempts) == 2 and publication_attempts[0] == publication_attempts[1]
        assert result["value"]["brief"]["presentation_strategy"]["slide_count"]["recommended"] == 6
        await env.replay(status["root_id"])
        assert len(requests) == 2
