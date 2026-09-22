"""Manuscript waves survive failed drafts, lost acknowledgements and history rollover."""

import json
import os
from collections import defaultdict
from hashlib import sha256

import pytest
from feature_harness import feature_environment, peak_children
from pptx_workflows import checkpoint_manuscript
from temporalio.worker import Replayer

from intramind_runtime.sdk import durable_activity

pytestmark = pytest.mark.integration


async def test_manuscript_keeps_native_partial_deck_and_accounting_across_rollover(store, monkeypatch):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment required")
    import llmai
    from api.background.common.models import ModelActivities
    from api.background.pptx.manuscript import ManuscriptActivities
    from api.background.pptx.manuscript_leaves import LEAVES
    from api.background.pptx.manuscript_workflows import WORKFLOWS
    from api.background.pptx.policy import PptxPolicy
    from api.config import PptxLLMConfig, PptxSettings
    from tests.background.pptx.test_manuscript import Replies, prepared_manuscript

    def forbidden(*args, **kwargs):
        raise AssertionError("Manuscript must use admitted operations, never a native LLM client")

    monkeypatch.setattr(llmai, "get_client", forbidden)
    policy = PptxPolicy.capture(PptxSettings(llm=PptxLLMConfig(
        base_url="http://fake/v1", model="test", context_tokens=16000, timeout_seconds=37,
        concurrency=2)), "test", {
            "model_profile": "test", "capacity_profile_id": "test-v1", "model": "test",
            "context_limit": 16384, "response_formats": ["text", "json_schema"],
            "endpoint_fingerprint": sha256(b"http://fake").hexdigest(),
        }, environment={"DISABLE_THINKING": "true", "FULL_PIPELINE_MANUSCRIPT_WAVE": "2"}
    ).model_copy(update={"history_window": 1})
    _, _, _, source_context, accepted, plan, brief = await prepared_manuscript(wave=2)
    replies, publications, joins = Replies(), [], defaultdict(list)

    async def respond(reservation, payload):
        assert reservation.operation.attempt_timeout_seconds == 37
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["model"] == "test"
        monkeypatch.setenv("CUSTOM_MODEL", "changed-worker-model")
        monkeypatch.setenv("FULL_PIPELINE_MANUSCRIPT_WAVE", "8")
        answer = replies.answer(payload, payload["max_tokens"])
        return {"choices": [{"message": {"content": answer}}]}

    def activities(factory):
        class LostJoinAcknowledgement(ManuscriptActivities):
            @durable_activity(name="pptx.manuscript-advance/v1")
            async def advance(self, inputs):
                result = await super().advance(inputs)
                attempts = joins[inputs["state"]["sha256"]]
                attempts.append(result)
                if len(attempts) == 1:
                    raise OSError("Join acknowledgement lost after immutable state was written")
                assert result == attempts[0]
                return result

        @durable_activity(name="test.pptx.manuscript-seed/v1")
        async def seed(inputs):
            client = factory(inputs["tenant_id"])
            try:
                snapshot = dict(accepted, policy=await client.read_json(inputs["configuration"]))
                context = await client.put_json({
                    "plan": await client.put_json(snapshot),
                    "prepared": await client.put_json(source_context.model_dump(mode="json")),
                    "model_calls": 0,
                })
                return await client.put_json({
                    "context": context, "brief": await client.put_json(brief.model_dump(mode="json")),
                    "deck_plan": await client.put_json(plan.model_dump(mode="json")), "model_calls": 0,
                })
            finally:
                await client.close()

        @durable_activity(name="test.pptx.manuscript-publish/v1")
        async def publish(inputs):
            publications.append(inputs["input"])
            if len(publications) == 1:
                raise OSError("Publication acknowledgement lost after manuscript completion")
            return inputs["input"]

        return [*LostJoinAcknowledgement(factory, model_profile="test").registered(),
                ModelActivities(factory, leaves=LEAVES).plan, seed, publish]

    workflows = [checkpoint_manuscript, *WORKFLOWS]
    async with feature_environment(store, name="pptx-manuscript-checkpoint", workflows=workflows,
                                   build_activities=activities, respond=respond) as env:
        status, result = await env.submit({}, configuration=policy.model_dump(mode="json"))
        assert len(publications) == 2 and publications[0] == publications[1]
        assert len(joins) == 2 and all(len(attempts) == 2 for attempts in joins.values())
        assert replies.counts == {("draft", (2, 3)): 2, ("draft", (4, 5)): 2,
                                 ("audit", (2, 3)): 1, ("repair", (2,)): 2,
                                 ("draft", (6,)): 1, ("audit", (6,)): 1}
        count = sum(replies.counts.values())
        assert len(env.calls) == result["model_calls"] == count
        assert status["operations"] == {"SUCCEEDED": count}
        assert status["reserved"] == 0 and status["spent"] == count * 30
        assert any(issue["issue"] == "batch_lost" and issue["slide"] == [4, 5]
                   for issue in result["issues"])
        client = env.runtime_client("user:test")
        try:
            deck = await client.read_json(result["deck"])
            assert [s["index"] for s in deck["slides"]] == [1, 2, 3, 6]
            assert "999" not in deck["raw_markdown"]
            state = await client.read_json(result["state"])
            assert state["claimed_chunks"] and state["cursor"] == len(state["batches"]) == 3
            assert len(state["journal"]) == 3 and state["window"] == 2
        finally:
            await client.close()

        phases, rollovers = [], []

        async def replay_chain(workflow_id, run_id=None):
            history = await env.temporal.get_workflow_handle(workflow_id, run_id=run_id).fetch_history()
            await Replayer(workflows=workflows).replay_workflow(history)
            start = history.events[0].workflow_execution_started_event_attributes
            envelope = (await env.temporal.data_converter.decode(start.input.payloads))[0]
            assert envelope["root_id"] == status["root_id"] and len(json.dumps(envelope)) < 65536
            if start.workflow_type.name == "pptx.manuscript/v1":
                phases.append(peak_children(history))
            for event in history.events:
                if event.HasField("child_workflow_execution_started_event_attributes"):
                    child = event.child_workflow_execution_started_event_attributes.workflow_execution
                    await replay_chain(child.workflow_id, child.run_id)
                if event.HasField("workflow_execution_continued_as_new_event_attributes"):
                    continued = event.workflow_execution_continued_as_new_event_attributes
                    checkpoint = (await env.temporal.data_converter.decode(continued.input.payloads))[0]
                    assert checkpoint["configuration"] == envelope["configuration"]
                    assert checkpoint["deadline"] == envelope["deadline"]
                    assert checkpoint["input"]["pptx_manuscript_checkpoint"] == 1
                    rollovers.append(checkpoint)
                    await replay_chain(workflow_id, continued.new_execution_run_id)

        await replay_chain(status["root_id"])
        assert len(rollovers) == 1 and phases == [2, 1]
        assert len(env.calls) == count
