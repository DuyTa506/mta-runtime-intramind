"""Native planning repairs and bounded history on real Temporal/PostgreSQL."""

import json
import os
import re
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pytest
from feature_harness import feature_environment, peak_children
from pptx_workflows import checkpoint_planning
from temporalio.worker import Replayer

from intramind_runtime.sdk import durable_activity

pytestmark = pytest.mark.integration


async def test_planning_recovers_failed_batches_and_publication_without_replanning(store, monkeypatch):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment required")
    import llmai
    from api.background.common.models import ModelActivities
    from api.background.pptx.activities import PptxActivities
    from api.background.pptx.leaves import LEAVES
    from api.background.pptx.planning import PlanningActivities
    from api.background.pptx.planning_leaves import LEAVES as PLAN_LEAVES
    from api.background.pptx.planning_workflows import WORKFLOWS as PLAN_WORKFLOWS
    from api.background.pptx.policy import PptxPolicy
    from api.background.pptx.workflows import WORKFLOWS
    from api.config import PptxLLMConfig, PptxSettings
    from tools.pptx.documents import LoadedCorpus

    def forbidden(*args, **kwargs):
        raise AssertionError("Planning must use the broker, never a native LLM client")

    monkeypatch.setattr(llmai, "get_client", forbidden)
    policy = PptxPolicy.capture(PptxSettings(llm=PptxLLMConfig(
        base_url="http://fake/v1", model="test", context_tokens=16000, timeout_seconds=37,
        concurrency=2)), "test", {
            "model_profile": "test", "capacity_profile_id": "test-v1", "model": "test",
            "context_limit": 16384, "response_formats": ["text", "json_schema"],
            "endpoint_fingerprint": sha256(b"http://fake").hexdigest(),
        }, environment={"DISABLE_THINKING": "true"}).model_copy(update={"history_window": 1})
    source = "# Nghiên cứu\n## Chương 1\nTrước: 42 đơn vị. Sau: 84 đơn vị.\n## Chương 2\nBằng chứng bổ sung."
    fixture = Path(__file__).resolve().parents[2] / "mta-ai-intramind/tests/tools/fixtures/pptx-brief-4848c757.json"
    schemas, batches, loads, publications = Counter(), [], [], []

    def load(ids, **kwargs):
        loads.append(ids)
        return LoadedCorpus([("Source", source)], [], {"doc": "Source"})

    async def respond(reservation, payload):
        assert reservation.operation.attempt_timeout_seconds == 37
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["model"] == "test"
        monkeypatch.setenv("CUSTOM_MODEL", "changed-worker-model")
        monkeypatch.setenv("DECK_PLAN_TOKENS_PER_SLIDE", "2000")
        monkeypatch.setenv("FULL_PIPELINE_LLM_CONCURRENCY", "8")
        schema = payload.get("response_format", {}).get("json_schema", {}).get("name", "reading")
        schemas[schema] += 1
        prompt = payload["messages"][-1]["content"]
        if schema == "reading":
            answer = source
        elif schema == "presentation_brief":
            assert "Chỉ dùng dữ liệu nguồn" in prompt
            answer = "invalid JSON" if schemas[schema] == 1 else fixture.read_text(encoding="utf-8")
        elif schema == "deck_allocation":
            answer = json.dumps({"presentation_title": "Kết quả nghiên cứu",
                "narrative_summary": "Các kết quả và bằng chứng đã được ghi nhận.",
                "budgets": [{"group_index": 0, "slides": 3}, {"group_index": 1, "slides": 3}]})
        elif schema == "section_outline":
            # The heading list is authoritative; the brief may mention both.
            group = re.search(r"Headings you may cite \(and only these\):\n(.+?)\n\n", prompt, re.S).group(1)
            group = "Chương 2" if "Chương 2" in group else "Chương 1"
            answer = json.dumps({"slides": [{"index": i, "title": f"Nội dung {i}",
                "purpose": f"Trình bày bằng chứng {i}", "content_type": "text",
                "source_sections": [group]} for i in range(1, 4)]})
        elif schema == "slide_details":
            indexes = [int(i) for i in re.findall(r"--- Slide (\d+) \(", prompt)]
            batches.append((indexes, payload["max_tokens"]))
            answer = "invalid JSON" if indexes in ([1, 2, 3, 4], [3, 4]) else json.dumps({
                "slides": [{"index": i, "key_message": f"Bằng chứng {i}",
                    "required_points": ["Giữ nguyên số liệu nguồn"], "visual_plan": {"kind": "none"}}
                    for i in indexes]})
        else:
            raise AssertionError(f"Unexpected planning schema: {schema}")
        return {"choices": [{"message": {"content": answer}}]}

    def activities(factory):
        @durable_activity(name="test.pptx.planning-publish/v1")
        async def publish(inputs):
            publications.append(inputs["input"])
            if len(publications) == 1:
                raise OSError("Publication acknowledgement lost after planning")
            return inputs["input"]

        return [*PptxActivities(factory, model_profile="test", source_loader=load).registered(),
                *PlanningActivities(factory, model_profile="test").registered(),
                ModelActivities(factory, leaves=LEAVES | PLAN_LEAVES).plan, publish]

    workflows = [checkpoint_planning, *WORKFLOWS, *PLAN_WORKFLOWS]
    async with feature_environment(store, name="pptx-planning-checkpoint", workflows=workflows,
                                   build_activities=activities, respond=respond) as env:
        status, result = await env.submit({"document_ids": ["doc"], "language": "vi", "n_slides": 6,
            "instructions": "Chỉ dùng dữ liệu nguồn", "avoid_layout_repetition": True},
            configuration=policy.model_dump(mode="json"))
        assert loads == [["doc"]] and len(publications) == 2 and publications[0] == publications[1]
        assert schemas == {"reading": 2, "presentation_brief": 2, "deck_allocation": 1,
                           "section_outline": 2, "slide_details": 5}
        count = sum(schemas.values())
        assert len(env.calls) == result["model_calls"] == count
        assert status["operations"] == {"SUCCEEDED": count}
        assert status["reserved"] == 0 and status["spent"] == count * 30
        assert batches.count(([1, 2, 3, 4], 4000)) == 2
        assert ([1, 2], 4000) in batches and ([3, 4], 4000) in batches and ([5, 6], 2000) in batches
        assert result["reports"]["assembly"]["fallback_slides"] == [3, 4]
        assert result["reports"]["expansion"]["split_batches"][0]["recovered"] == [1, 2]
        client = env.runtime_client("user:test")
        try:
            plan = await client.read_json(result["deck_plan"])
            assert [s["index"] for s in plan["slides"]] == list(range(1, 7))
            assert all(s["source_document"] for s in plan["slides"])
            source_context = await client.read_json(result["context"])
            accepted = await client.read_json(source_context["plan"])
            assert accepted["config"]["avoid_layout_repetition"]
            assert accepted["config"]["preferences"]["exact_slide_count"] == 6
        finally:
            await client.close()

        phases, rollovers = [], []

        async def replay_chain(workflow_id, run_id=None):
            history = await env.temporal.get_workflow_handle(workflow_id, run_id=run_id).fetch_history()
            await Replayer(workflows=workflows).replay_workflow(history)
            start = history.events[0].workflow_execution_started_event_attributes
            envelope = (await env.temporal.data_converter.decode(start.input.payloads))[0]
            assert envelope["root_id"] == status["root_id"] and len(json.dumps(envelope)) < 65536
            if start.workflow_type.name == "pptx.planning/v1":
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
                    if start.workflow_type.name == "pptx.planning/v1":
                        assert checkpoint["input"]["pptx_planning_checkpoint"] == 1
                        rollovers.append(checkpoint)
                    await replay_chain(workflow_id, continued.new_execution_run_id)

        await replay_chain(status["root_id"])
        assert len(rollovers) == 5 and max(phases) == 2
        assert len(env.calls) == count
