"""Real workflow/ledger replay with existing Markdown parser and renderer."""

import json
import os

import pytest
from feature_harness import feature_environment, peak_children

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("repair_fails", [False, True])
@pytest.mark.parametrize("rollover", [False, True])
async def test_translation_checkpoints_partial_batches_and_retries_publication(
    store, monkeypatch, repair_fails, rollover,
):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment required")
    from api.background.common.models import ModelActivities
    from api.background.translation.activities import TranslationActivities
    from api.background.translation.policy import TranslationPolicy
    from api.background.translation.workflows import WORKFLOWS
    from api.config import TranslationSettings
    from tools.translation.durable.leaves import LEAVES

    settings = TranslationSettings.model_validate({
        "llm": {"max_completion_tokens": 600, "timeout": 19},
        "engine": {"max_retries": 2, "thread_count": 1, "history_window": 1 if rollover else 128},
    })
    policy = TranslationPolicy.capture(settings, model_profile="test", src_lang="en",
                                       dst_lang="vi", rules={}).model_dump(mode="json")
    requests, publications = [], []

    async def respond(reservation, payload):
        assert reservation.operation.attempt_timeout_seconds == 19
        user = payload["messages"][-1]["content"]
        # Source JSON is the final fenced JSON object inside the prompt.
        raw = user.split("```json\n")[-1].split("\n```")[0]
        items = json.loads(raw)
        requests.append(items)
        settings.engine.thread_count = 64
        settings.engine.max_retries = 0
        settings.enabled = False
        output = {key: ("Xin chào thế giới" if "Hello" in value else "Một câu thứ hai.")
                  for key, value in items.items()}
        if len(requests) == 1:
            output = {key: value for key, value in output.items() if "Xin chào" in value}
        elif repair_fails:
            output = {key: "" for key in items}
        return {"choices": [{"message": {"content": json.dumps(output, ensure_ascii=False)}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20}}

    def publish(**kwargs):
        publications.append(kwargs)
        if len(publications) == 1:
            raise OSError("Publication response lost after object persistence")
        assert kwargs == publications[0]
        return "translation-artifact"

    def activities(factory):
        feature = TranslationActivities(factory, model_profile="test", publication=publish)
        return [*feature.registered(), ModelActivities(factory, leaves=LEAVES).plan]

    async with feature_environment(store, name="translation", workflows=WORKFLOWS,
                                    build_activities=activities, respond=respond) as env:
        client = env.runtime_client("user:test")
        try:
            uploaded = await client.put_bytes(b"# Hello world\n\nA second sentence.\n\n```python\nx = 1\n```\n",
                                              content_type="text/markdown")
            status, result = await env.submit({"file": uploaded, "filename": "source.md",
                "src_lang": "en", "dst_lang": "vi", "bilingual": True}, configuration=policy)
            assert status["state"] == ("PARTIAL" if repair_fails else "SUCCEEDED")
            assert len(publications) == 2
            assert len(requests) == (3 if repair_fails else 2)
            assert all("Hello world" not in call.values() for call in requests[1:])
            assert result["missing_count"] == int(repair_fails)
            assert result["total_tokens"] == len(requests) * 20
            rendered = await client.read_bytes({"key": result["object_key"], "sha256": result["sha256"],
                "size": result["output_size_bytes"], "content_type": "text/markdown"})
            assert b"x = 1" in rendered and "Xin chào thế giới" in rendered.decode()
            history = await env.temporal.get_workflow_handle(status["root_id"]).fetch_history()
            peaks = [peak_children(history)]
            previous_run = history.events[0].workflow_execution_started_event_attributes.continued_execution_run_id
            assert bool(previous_run) is rollover
            while previous_run:
                from temporalio.worker import Replayer
                previous = await env.temporal.get_workflow_handle(status["root_id"], run_id=previous_run).fetch_history()
                peaks.append(peak_children(previous))
                await Replayer(workflows=WORKFLOWS).replay_workflow(previous)
                previous_run = previous.events[0].workflow_execution_started_event_attributes.continued_execution_run_id
            assert max(peaks) == 1
            await env.replay(status["root_id"])
            assert len(requests) == (3 if repair_fails else 2)
        finally:
            await client.close()
