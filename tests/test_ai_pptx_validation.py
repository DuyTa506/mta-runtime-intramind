"""Real Temporal/Postgres coherence, context rejection and layout publication recovery."""

import json
import os
from collections import defaultdict
from hashlib import sha256

import httpx
import pytest
from feature_harness import feature_environment, peak_children
from pptx_workflows import checkpoint_validation
from temporalio.worker import Replayer

from intramind_runtime.sdk import durable_activity

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("oversized_audit", [False, True])
async def test_validation_preserves_repair_policy_layout_and_budget_through_recovery(store, monkeypatch, oversized_audit):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment required")
    import llmai
    from api.background.pptx.policy import PptxPolicy
    from api.background.pptx.validation import ValidationActivities
    from api.background.pptx.validation_workflows import WORKFLOWS
    from api.config import PptxLLMConfig, PptxSettings
    from tests.background.pptx.test_validation import Replies, prepared_validation

    def forbidden(*args, **kwargs):
        raise AssertionError("Validation cannot create a native inference client")

    monkeypatch.setattr(llmai, "get_client", forbidden)
    policy = PptxPolicy.capture(PptxSettings(llm=PptxLLMConfig(
        base_url="http://fake/v1", model="test", context_tokens=16000, timeout_seconds=37,
        concurrency=2)), "test", {
            "model_profile": "test", "capacity_profile_id": "test-v1", "model": "test",
            "context_limit": 16384, "response_formats": ["text", "json_schema"],
            "endpoint_fingerprint": sha256(b"http://fake").hexdigest(),
        }, environment={"DISABLE_THINKING": "true"}).model_copy(update={"history_window": 1})
    _, _, _, context, accepted, plan, brief, deck, packets = await prepared_validation()
    replies, joins, layouts, publications, rejections = Replies(), defaultdict(list), [], [], []

    async def tokenize(request):
        payload = json.loads(request.content)
        if request.url.path == "/apply-template":
            return httpx.Response(200, json={"prompt": json.dumps(payload["messages"])})
        assert request.url.path == "/tokenize"
        oversized = oversized_audit and "final coherence auditor" in payload["content"]
        if oversized:
            rejections.append(payload["content"])
        return httpx.Response(200, json={"tokens": [1] * (16384 if oversized else 10)})

    async def respond(reservation, payload):
        assert reservation.operation.attempt_timeout_seconds == 37
        assert payload["model"] == "test" and payload["chat_template_kwargs"] == {"enable_thinking": False}
        monkeypatch.setenv("CUSTOM_MODEL", "changed-worker-model")
        monkeypatch.setenv("FULL_PIPELINE_LLM_CONCURRENCY", "8")
        return {"choices": [{"message": {"content": replies.answer(payload, payload["max_tokens"])}}]}

    def activities(factory):
        class LostAcknowledgements(ValidationActivities):
            @durable_activity(name="pptx.validation-advance/v1")
            async def advance(self, inputs):
                result = await super().advance(inputs)
                attempts = joins[inputs["state"]["sha256"]]
                attempts.append(result)
                if len(attempts) == 1:
                    raise OSError("Repair join acknowledgement lost after state persistence")
                assert result == attempts[0]
                return result

            @durable_activity(name="pptx.validation-finish/v1")
            async def finish(self, inputs):
                result = await super().finish(inputs)
                layouts.append(result)
                if len(layouts) == 1:
                    raise OSError("Layout artifact acknowledgement lost")
                assert result == layouts[0]
                return result

        @durable_activity(name="test.pptx.validation-seed/v1")
        async def seed(inputs):
            client = factory(inputs["tenant_id"])
            try:
                snapshot = dict(accepted, policy=await client.read_json(inputs["configuration"]))
                source = await client.put_json({
                    "plan": await client.put_json(snapshot),
                    "prepared": await client.put_json(context.model_dump(mode="json")), "model_calls": 0,
                })
                planning = await client.put_json({
                    "context": source, "brief": await client.put_json(brief.model_dump(mode="json")),
                    "deck_plan": await client.put_json(plan.model_dump(mode="json")), "model_calls": 0,
                })
                return await client.put_json({
                    "planning": planning, "deck": await client.put_json(deck.model_dump(mode="json")),
                    "packets": await client.put_json(packets), "model_calls": 0, "issues": [],
                })
            finally:
                await client.close()

        @durable_activity(name="test.pptx.validation-publish/v1")
        async def publish(inputs):
            publications.append(inputs["input"])
            if len(publications) == 1:
                raise OSError("Publication acknowledgement lost")
            return inputs["input"]

        return [*LostAcknowledgements(factory, model_profile="test").registered(), seed, publish]

    workflows = [checkpoint_validation, *WORKFLOWS]
    async with feature_environment(store, name="pptx-validation-checkpoint", workflows=workflows,
                                   build_activities=activities, respond=respond, tokenize_prompt=tokenize) as env:
        status, result = await env.submit({}, configuration=policy.model_dump(mode="json"))
        assert len(publications) == len(layouts) == 2 and publications[0] == publications[1]
        assert len(joins) == (1 if oversized_audit else 2)
        assert all(len(attempts) == 2 for attempts in joins.values())
        if oversized_audit:
            assert len(rejections) == 1 and replies.counts == {("repair", (10,)): 1}
            assert any(i["slide"] == 0 and "RequestPreparationError" in i["issues"][0] for i in result["issues"])
        else:
            assert not rejections and replies.counts == {
                ("audit", (1, 2, 4, 6, 10, 12)): 1, ("repair", (2,)): 1,
                ("repair", (4,)): 1, ("repair", (10,)): 1, ("repair", (12,)): 1,
            }
            assert any(i["slide"] == 4 and i.get("type") == "ValueError" for i in result["issues"])
        count = sum(replies.counts.values())
        assert len(env.calls) == result["model_calls"] == count
        assert status["operations"] == {"SUCCEEDED": count}
        assert status["reserved"] == 0 and status["spent"] == count * 30
        client = env.runtime_client("user:test")
        try:
            result_deck = await client.read_json(result["deck"])
            assert [s["index"] for s in result_deck["slides"]] == [1, 2, 4, 6, 10, 12]
            assert result_deck["slides"][2] == deck.slides[2].model_dump(mode="json")
            decisions = await client.read_json(result["layouts"])
            assert len(decisions) == 6 and result["form_usage"]
            state = await client.read_json(result["state"])
            assert state["window"] == 2
            assert [w["slide_index"] for w in state["work"]] == ([10] if oversized_audit else [2, 4, 10, 12])
        finally:
            await client.close()

        phases, rollovers = [], []

        async def replay_chain(workflow_id, run_id=None):
            history = await env.temporal.get_workflow_handle(workflow_id, run_id=run_id).fetch_history()
            await Replayer(workflows=workflows).replay_workflow(history)
            start = history.events[0].workflow_execution_started_event_attributes
            envelope = (await env.temporal.data_converter.decode(start.input.payloads))[0]
            assert envelope["root_id"] == status["root_id"] and len(json.dumps(envelope)) < 65536
            if start.workflow_type.name == "pptx.validation/v1":
                phases.append(peak_children(history))
            for event in history.events:
                if event.HasField("child_workflow_execution_started_event_attributes"):
                    child = event.child_workflow_execution_started_event_attributes.workflow_execution
                    await replay_chain(child.workflow_id, child.run_id)
                if event.HasField("workflow_execution_continued_as_new_event_attributes"):
                    continued = event.workflow_execution_continued_as_new_event_attributes
                    checkpoint = (await env.temporal.data_converter.decode(continued.input.payloads))[0]
                    assert checkpoint["configuration"] == envelope["configuration"] and checkpoint["deadline"] == envelope["deadline"]
                    assert checkpoint["input"]["pptx_validation_checkpoint"] == 1
                    rollovers.append(checkpoint)
                    await replay_chain(workflow_id, continued.new_execution_run_id)

        await replay_chain(status["root_id"])
        assert len(rollovers) == (0 if oversized_audit else 1)
        assert phases == ([1] if oversized_audit else [2, 2])
        assert len(env.calls) == count and len(rejections) == int(oversized_audit)
