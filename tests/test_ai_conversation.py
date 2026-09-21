"""Real ledger/Temporal recovery for gateway-owned conversation model work."""

import asyncio
import os

import pytest
from feature_harness import feature_environment
from temporalio.worker import Replayer

from intramind_runtime.sdk import durable_activity

pytestmark = pytest.mark.integration


def feature():
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment required")
    from api.background.common.models import ModelActivities
    from api.background.conversation.activities import ConversationActivities
    from api.background.conversation.leaves import LEAVES
    from api.background.conversation.workflows import WORKFLOWS

    return ModelActivities, ConversationActivities, LEAVES, WORKFLOWS


def policy(kind, **changes):
    return {"task_type": f"conversation.{kind}/v1", "model_profile": "test",
            "capacity_profile_id": "test-v1", "disable_thinking": True,
            "attempt_timeout_seconds": 37, "history_window": 1, **changes}


async def replay_all(env, root_id, workflows):
    handle = env.temporal.get_workflow_handle(root_id)
    histories = 0
    while True:
        history = await handle.fetch_history()
        await Replayer(workflows=workflows).replay_workflow(history)
        histories += 1
        previous = history.events[0].workflow_execution_started_event_attributes.continued_execution_run_id
        if not previous:
            return histories
        handle = env.temporal.get_workflow_handle(root_id, run_id=previous)


async def test_compact_restarts_mid_fold_rolls_over_and_retries_result_without_repeating_calls(store):
    Models, Activities, leaves, workflows = feature()
    second_call, release = asyncio.Event(), asyncio.Event()
    prompts, publications = [], []

    async def respond(reservation, payload):
        assert reservation.operation.attempt_timeout_seconds == 37
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        prompts.append(payload["messages"][-1]["content"])
        ordinal = len(prompts)
        if ordinal > 1:
            assert f"carry-{ordinal - 1}" in prompts[-1]
        if ordinal == 2:
            second_call.set()
            await release.wait()
        return {"choices": [{"message": {"content": f"carry-{ordinal}"}}]}

    class InterruptedPublication(Activities):
        @durable_activity(name="conversation.finish/v1")
        async def finish(self, inputs):
            result = await super().finish(inputs)
            publications.append(result)
            if len(publications) == 1:
                raise OSError("Lost artifact publication acknowledgement")
            return result

    def activities(factory):
        return [*InterruptedPublication(factory, model_profile="unused").registered(),
                Models(factory, leaves=leaves).plan]

    async with feature_environment(store, name="conversation.compact", workflows=workflows,
                                   build_activities=activities, respond=respond) as env:
        source = {"user_id": "test", "summary": "old-summary",
                  "turns": [{"role": "user", "content": f"turn-{i}"} for i in range(41)]}
        task = asyncio.create_task(env.submit(source, configuration=policy("compact")))
        try:
            await asyncio.wait_for(second_call.wait(), 30)
            await asyncio.wait_for(env.restart_worker(), 20)
            release.set()
            status, result = await task
            assert result == {"summary": "carry-3", "absorbed": 41}
            assert len(env.calls) == 3 and len(publications) == 2
            assert publications[0] == publications[1]
            assert status["spent"] == 90 and status["reserved"] == 0
            assert await replay_all(env, status["root_id"], workflows) == 3
            assert len(env.calls) == 3
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("kind,body,expected", [
    ("title", {"question": "Nghị định?", "answer": "a" * 501}, {"title": "Nghị định mới"}),
    ("compact", {"summary": "unchanged", "turns": []}, {"summary": "unchanged", "absorbed": 0}),
])
async def test_title_and_empty_compact_have_exact_call_budgets_and_replay(store, kind, body, expected):
    Models, Activities, leaves, workflows = feature()

    async def respond(reservation, payload):
        assert kind == "title"
        assert reservation.operation.max_output_tokens == 48
        assert payload["temperature"] == 0.2
        assert "a" * 501 not in payload["messages"][-1]["content"]
        return {"choices": [{"message": {"content": '"Tiêu đề: Nghị định   mới"'}}]}

    def activities(factory):
        return [*Activities(factory, model_profile="unused").registered(), Models(factory, leaves=leaves).plan]

    async with feature_environment(store, name=f"conversation.{kind}", workflows=workflows,
                                   build_activities=activities, respond=respond) as env:
        status, result = await env.submit({"user_id": "test", **body}, configuration=policy(kind))
        assert result == expected
        assert len(env.calls) == (1 if kind == "title" else 0)
        assert status["reserved"] == 0 and status["spent"] == (30 if kind == "title" else 0)
        await env.replay(status["root_id"])
