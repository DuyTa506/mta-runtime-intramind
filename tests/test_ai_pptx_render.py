"""A separate Temporal worker renders and publishes without reopening inference."""

import asyncio
import os
import sys
from copy import deepcopy
from datetime import UTC, datetime, timedelta
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
