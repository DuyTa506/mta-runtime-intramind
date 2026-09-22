"""A separate Temporal worker renders and publishes without reopening inference."""

import asyncio
import json
import os
import re
import sys
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pytest
from feature_harness import feature_environment
from pptx_workflows import checkpoint_render
from temporalio.api.workflowservice.v1 import SetWorkerDeploymentCurrentVersionRequest
from temporalio.client import WorkflowFailureError
from temporalio.common import VersioningBehavior, WorkerDeploymentVersion
from temporalio.service import RPCError
from temporalio.worker import Replayer, Worker, WorkerDeploymentConfig

from intramind_runtime.sdk import durable_activity

pytestmark = pytest.mark.integration


async def test_pptx_root_keeps_accepted_policy_across_every_phase_and_publication_retry(store, monkeypatch):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment required")
    import llmai
    from api.background.common.models import ModelActivities
    from api.background.pptx.activities import PptxActivities
    from api.background.pptx.leaves import LEAVES
    from api.background.pptx.manuscript import ManuscriptActivities
    from api.background.pptx.manuscript_leaves import LEAVES as MANUSCRIPT_LEAVES
    from api.background.pptx.manuscript_workflows import WORKFLOWS as MANUSCRIPT_WORKFLOWS
    from api.background.pptx.planning import PlanningActivities
    from api.background.pptx.planning_leaves import LEAVES as PLAN_LEAVES
    from api.background.pptx.planning_workflows import WORKFLOWS as PLAN_WORKFLOWS
    from api.background.pptx.policy import PptxPolicy
    from api.background.pptx.render import RenderActivities
    from api.background.pptx.render_workflows import WORKFLOWS as RENDER_WORKFLOWS
    from api.background.pptx.validation import ValidationActivities
    from api.background.pptx.validation_workflows import WORKFLOWS as VALIDATION_WORKFLOWS
    from api.background.pptx.workflows import WORKFLOWS
    from api.config import PptxLLMConfig, PptxSettings
    from tests.background.pptx.test_render import deck_bytes
    from tools.pptx.documents import LoadedCorpus
    from tools.pptx.engine.utils.llm_calls.validate_slide_manuscript import (
        AUDIT_SYSTEM_PROMPT,
        COHERENCE_SYSTEM_PROMPT,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A PPTX root must use recorded broker inference")

    monkeypatch.setattr(llmai, "get_client", forbidden)
    queue = "pptx-root-render-" + uuid4().hex
    policy = PptxPolicy.capture(PptxSettings(llm=PptxLLMConfig(
        base_url="http://fake/v1", model="test", context_tokens=16000,
        timeout_seconds=37, concurrency=2)), "test", {
            "model_profile": "test", "capacity_profile_id": "test-v1", "model": "test",
            "context_limit": 16384, "response_formats": ["text", "json_schema"],
            "endpoint_fingerprint": sha256(b"http://fake").hexdigest(),
        }, environment={"DISABLE_THINKING": "true"}, render_task_queue=queue,
    ).model_copy(update={"history_window": 1})
    source = "# Nghiên cứu\n## Chương 1\nKết quả: 42 đơn vị.\n## Chương 2\nBằng chứng bổ sung."
    fixture = Path(__file__).resolve().parents[2] / "mta-ai-intramind/tests/tools/fixtures/pptx-brief-4848c757.json"
    schemas, loads, exports, publications, published = Counter(), [], [], [], []

    def load(ids, **kwargs):
        loads.append(ids)
        return LoadedCorpus([("Source", source)], [], {"doc": "Source"})

    async def respond(reservation, payload):
        assert reservation.operation.attempt_timeout_seconds == 37
        assert payload["model"] == "test"
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        monkeypatch.setenv("CUSTOM_MODEL", "changed-worker-model")
        monkeypatch.setenv("BACKGROUND_RUNTIME__PPTX_TASK_QUEUE", "wrong-after-acceptance")
        system, prompt = [message["content"] for message in payload["messages"]]
        schema = payload.get("response_format", {}).get("json_schema", {}).get("name")
        if schema == "presentation_brief":
            answer = fixture.read_text(encoding="utf-8")
        elif schema == "deck_allocation":
            answer = json.dumps({"presentation_title": "Kết quả nghiên cứu",
                "narrative_summary": "Các kết quả và bằng chứng đã được ghi nhận.",
                "budgets": [{"group_index": 0, "slides": 3}, {"group_index": 1, "slides": 3}]})
        elif schema == "section_outline":
            headings = re.search(r"Headings you may cite \(and only these\):\n(.+?)\n\n", prompt, re.S).group(1)
            group = "Chương 2" if "Chương 2" in headings else "Chương 1"
            answer = json.dumps({"slides": [{"index": i, "title": f"Nội dung {i}",
                "purpose": f"Trình bày bằng chứng {i}", "content_type": "text",
                "source_sections": [group]} for i in range(1, 4)]})
        elif schema == "slide_details":
            indexes = [int(i) for i in re.findall(r"--- Slide (\d+) \(", prompt)]
            answer = json.dumps({"slides": [{"index": i, "key_message": "Kết quả nguồn",
                "required_points": ["42 đơn vị"], "visual_plan": {"kind": "none"}} for i in indexes]})
        elif system in (AUDIT_SYSTEM_PROMPT, COHERENCE_SYSTEM_PROMPT):
            schema = "coherence" if system == COHERENCE_SYSTEM_PROMPT else "audit"
            marker = "FULL MANUSCRIPT\n" if schema == "coherence" else "DRAFT SLIDES\n"
            indexes = re.findall(r"(?m)^## Slide (\d+)", prompt.split(marker, 1)[1])
            assert indexes
            answer = "\n".join(f"Slide {i}: PASS" for i in indexes)
        elif "REQUESTED BATCH\n" in prompt:
            schema = "manuscript"
            batch = prompt.split("REQUESTED BATCH\n", 1)[1].split("GLOBAL HEADING", 1)[0]
            indexes = re.findall(r"(?m)^## Slide (\d+)", batch)
            assert indexes
            answer = "\n\n---\n\n".join(f"## Slide {i} — Kết quả nguồn\n"
                "> Core message: Kết quả được xác nhận.\n> Audience move: Hiểu bằng chứng.\n"
                "> Source anchors: Chương 1\n\n### Narrative\n"
                "Kết quả có 42 đơn vị, theo tài liệu nguồn.\n\n### Visual intent\nKhông có.\n"
                for i in indexes)
        else:
            assert schema is None
            schema, answer = "reading", source
        schemas[schema] += 1
        return {"choices": [{"message": {"content": answer}}]}

    def activities(factory):
        return [*PptxActivities(factory, model_profile="test", source_loader=load).registered(),
                *PlanningActivities(factory, model_profile="test").registered(),
                *ManuscriptActivities(factory, model_profile="test").registered(),
                *ValidationActivities(factory, model_profile="test").registered(),
                ModelActivities(factory, leaves=LEAVES | PLAN_LEAVES | MANUSCRIPT_LEAVES).plan]

    async def export(job, directory):
        exports.append(deepcopy(job))
        (directory / "deck.pptx").write_bytes(deck_bytes(len(job["deck"]["slides"])))

    def publication(**kwargs):
        publications.append(deepcopy(kwargs))
        return "stable-root-publication"

    class LostPublicationAck(RenderActivities):
        @durable_activity(name="pptx.publish/v1")
        async def publish(self, inputs):
            result = await super().publish(inputs)
            published.append(result)
            if len(published) == 1:
                raise OSError("publication committed but activity acknowledgement lost")
            assert result == published[0]
            return result

    workflows = [*WORKFLOWS, *PLAN_WORKFLOWS, *MANUSCRIPT_WORKFLOWS, *VALIDATION_WORKFLOWS]
    async with feature_environment(store, name="pptx", workflows=workflows,
                                   build_activities=activities, respond=respond) as env:
        version = WorkerDeploymentVersion(queue, "render-build-1")
        async with Worker(env.temporal, task_queue=queue, workflows=RENDER_WORKFLOWS,
            activities=LostPublicationAck(env.runtime_client, exporter=export,
                                         publication=publication).registered(),
            max_concurrent_activities=1,
            deployment_config=WorkerDeploymentConfig(version=version, use_worker_versioning=True,
                default_versioning_behavior=VersioningBehavior.PINNED)):
            for attempt in range(30):
                try:
                    await env.temporal.workflow_service.set_worker_deployment_current_version(
                        SetWorkerDeploymentCurrentVersionRequest(namespace=env.temporal.namespace,
                            deployment_name=version.deployment_name, build_id=version.build_id,
                            identity="pptx-root-qualification"))
                    break
                except RPCError:
                    if attempt == 29:
                        raise
                    await asyncio.sleep(0.5)
            status, result = await env.submit({"document_ids": ["doc"], "unit_id": "verified-unit",
                "format": "detailed", "n_slides": 6, "style": "academic",
                "avoid_layout_repetition": False}, configuration=policy.model_dump(mode="json"))
            assert loads == [["doc"]] and len(exports) == 1
            assert len(publications) == len(published) == 2 and publications[0] == publications[1]
            assert publications[0]["task_id"] == "rtw_" + status["root_id"]
            assert publications[0]["owner_user_id"] == "test"
            assert publications[0]["owner_unit_id"] == "verified-unit"
            assert publications[0]["source_request"]["avoid_layout_repetition"] is False
            assert result["slide_count"] == 6 and result["artifact_id"] == "stable-root-publication"
            assert result["style"] == "academic" and result["tool_info"]["model_name"] == "test"
            count = sum(schemas.values())
            assert count == len(env.calls) == result["llm_call_count"]
            assert schemas["coherence"] == 1 and schemas["manuscript"] >= 1
            assert status["operations"] == {"SUCCEEDED": count}
            assert status["reserved"] == 0 and status["spent"] == count * 30

            phases, starts = [], []
            replayer = Replayer(workflows=[*workflows, *RENDER_WORKFLOWS])

            async def replay_chain(workflow_id, run_id=None):
                history = await env.temporal.get_workflow_handle(workflow_id, run_id=run_id).fetch_history()
                await replayer.replay_workflow(history)
                start = history.events[0].workflow_execution_started_event_attributes
                envelope = (await env.temporal.data_converter.decode(start.input.payloads))[0]
                starts.append(envelope)
                assert envelope["root_id"] == status["root_id"] and envelope["tenant_id"] == "user:test"
                assert envelope["configuration"] == starts[0]["configuration"]
                assert envelope["deadline"] == starts[0]["deadline"]
                for event in history.events:
                    if event.HasField("start_child_workflow_execution_initiated_event_attributes"):
                        child = event.start_child_workflow_execution_initiated_event_attributes
                        if workflow_id == status["root_id"]:
                            phases.append(child.workflow_type.name)
                            assert (child.task_queue.name == queue) == (child.workflow_type.name == "pptx.render/v1")
                    if event.HasField("child_workflow_execution_started_event_attributes"):
                        child = event.child_workflow_execution_started_event_attributes.workflow_execution
                        await replay_chain(child.workflow_id, child.run_id)
                    if event.HasField("workflow_execution_continued_as_new_event_attributes"):
                        await replay_chain(workflow_id, event.workflow_execution_continued_as_new_event_attributes.new_execution_run_id)

            await replay_chain(status["root_id"])
            assert phases == [f"pptx.{phase}/v1" for phase in ("context", "planning", "manuscript", "validation", "render")]
            assert len(env.calls) == count and len(exports) == 1 and len(publications) == 2


async def test_separate_render_worker_recovers_lost_artifact_and_publication_ack(store, tmp_path):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment required")
    from api.background.pptx.render import RenderActivities, run_process
    from api.background.pptx.render_workflows import WORKFLOWS
    from api.background.pptx.worker import with_heartbeats
    from tests.background.pptx.test_render import copy_checkpoint, deck_bytes, prepared_render

    blobs, validation, _ = await prepared_render(keep=True, multi=True)
    queue = "pptx-render-" + uuid4().hex
    exports, render_results, publications, published = [], [], [], []
    cancelling, stopped = False, asyncio.Event()
    marker = tmp_path / "render-pid"

    async def export(job, directory):
        exports.append(job)
        if cancelling:
            try:
                await run_process([sys.executable, "-c",
                    "import os,sys,time; open(sys.argv[1],'w').write(str(os.getpid())); time.sleep(60)",
                    str(marker)], timeout=60, log_path=directory / "render.log")
            finally:
                stopped.set()
            raise AssertionError("Cancelled export must not publish")
        (directory / "deck.pptx").write_bytes(deck_bytes(len(job["deck"]["slides"])))

    async def exporting(job, directory):
        return await with_heartbeats(export(job, directory))

    def publication(**kwargs):
        publications.append(deepcopy(kwargs))
        return "idempotent-pptx-publication"

    class LostAcknowledgements(RenderActivities):
        @durable_activity(name="pptx.render/v1")
        async def render(self, inputs):
            result = await super().render(inputs)
            render_results.append(result)
            if len(render_results) == 1:
                raise OSError("render artifact persisted but activity acknowledgement lost")
            return result

        @durable_activity(name="pptx.publish/v1")
        async def publish(self, inputs):
            result = await super().publish(inputs)
            published.append(result)
            if len(published) == 1:
                raise OSError("publication committed but activity acknowledgement lost")
            assert result == published[0]
            return result

    def activities(factory):
        @durable_activity(name="test.pptx.render-seed/v1")
        async def seed(inputs):
            client = factory(inputs["tenant_id"])
            try:
                ref = await copy_checkpoint(blobs, client, validation, root_id=inputs["root_id"],
                                            started_at=inputs["started_at"])
                return {"validation": ref, "queue": queue}
            finally:
                await client.close()
        return [seed]

    async def forbidden(*args):
        raise AssertionError("Rendering must not dispatch inference")

    workflows = [checkpoint_render, *WORKFLOWS]
    async with feature_environment(store, name="pptx-render-checkpoint", workflows=workflows,
                                   build_activities=activities, respond=forbidden) as env:
        version = WorkerDeploymentVersion(queue, "render-build-1")
        renderer = Worker(
            env.temporal, task_queue=queue, workflows=WORKFLOWS,
            activities=LostAcknowledgements(env.runtime_client, exporter=exporting,
                                            publication=publication).registered(),
            max_concurrent_activities=1,
            default_heartbeat_throttle_interval=timedelta(seconds=2),
            max_heartbeat_throttle_interval=timedelta(seconds=5),
            deployment_config=WorkerDeploymentConfig(version=version, use_worker_versioning=True,
                default_versioning_behavior=VersioningBehavior.PINNED),
        )
        async with renderer:
            for attempt in range(30):
                try:
                    await env.temporal.workflow_service.set_worker_deployment_current_version(
                        SetWorkerDeploymentCurrentVersionRequest(namespace=env.temporal.namespace,
                            deployment_name=version.deployment_name, build_id=version.build_id,
                            identity="pptx-render-qualification"))
                    break
                except RPCError:
                    if attempt == 29:
                        raise
                    await asyncio.sleep(0.5)
            status, result = await env.submit({})
            assert status["reserved"] == status["spent"] == 0 and not env.calls
            assert result["slide_count"] == 6 and result["llm_call_count"] == 7
            assert result["corpus"]["primary"] == "Chính"
            assert len(exports) == len(render_results) == len(publications) == len(published) == 2
            assert publications[0] == publications[1]
            assert publications[0]["task_id"] == "rtw_" + status["root_id"]
            client = env.runtime_client("user:test")
            try:
                accepted_render = await client.read_json(render_results[-1])
                data = await client.read_bytes(accepted_render["file"])
                assert result["output_size_bytes"] == len(data) and data.startswith(b"PK")
                assert result["object_key"] == accepted_render["file"]["key"]
            finally:
                await client.close()
            root = await env.temporal.get_workflow_handle(status["root_id"]).fetch_history()
            await Replayer(workflows=workflows).replay_workflow(root)
            for event in root.events:
                if event.HasField("start_child_workflow_execution_initiated_event_attributes"):
                    assert event.start_child_workflow_execution_initiated_event_attributes.task_queue.name == queue
                if event.HasField("child_workflow_execution_started_event_attributes"):
                    child = event.child_workflow_execution_started_event_attributes.workflow_execution
                    history = await env.temporal.get_workflow_handle(child.workflow_id, run_id=child.run_id).fetch_history()
                    await Replayer(workflows=workflows).replay_workflow(history)
                    for transition in history.events:
                        if transition.HasField("activity_task_scheduled_event_attributes"):
                            assert transition.activity_task_scheduled_event_attributes.task_queue.name == queue
            assert len(exports) == len(publications) == 2 and not env.calls

            cancelling = True
            client = env.runtime_client("user:test")
            now = datetime.now(UTC)
            try:
                ref = await copy_checkpoint(blobs, client, validation, root_id="cancel-root",
                                            started_at=now.isoformat())
            finally:
                await client.close()
            handle = await env.temporal.start_workflow("pptx.render/v1", {
                "root_id": "cancel-root", "tenant_id": "user:test", "child_path": ["render"],
                "input": ref, "control_queue": queue, "started_at": now.isoformat(),
                "deadline": (now + timedelta(seconds=90)).isoformat(),
            }, id=queue + "-cancel", task_queue=queue)
            async with asyncio.timeout(10):
                while not marker.exists():
                    await asyncio.sleep(0.01)
            await handle.cancel()
            with pytest.raises(WorkflowFailureError):
                await handle.result()
            await asyncio.wait_for(stopped.wait(), timeout=10)
            process = Path(f"/proc/{marker.read_text()}/stat")
            if process.exists():
                assert process.read_text().split(")", 1)[1].split()[0] == "Z"
            await Replayer(workflows=workflows).replay_workflow(await handle.fetch_history())
            assert len(exports) == 3 and len(publications) == 2 and not env.calls
