"""Cross-repo rewrite: real Temporal + PostgreSQL, fake inference/document I/O.

Run with the AI dependency environment plus the built runtime/Temporal SDK and
RUNTIME_TEST_AI_FEATURES=yes. No production engine or document is accessed.
"""

import asyncio
import os
from contextlib import suppress
from uuid import uuid4

import httpx
import pytest
from conftest import pool
from fakes import MemoryArtifacts
from feature_harness import peak_children
from temporalio.api.workflowservice.v1 import SetWorkerDeploymentCurrentVersionRequest
from temporalio.client import Client
from temporalio.common import VersioningBehavior, WorkerDeploymentVersion
from temporalio.service import RPCError
from temporalio.worker import Replayer, Worker, WorkerDeploymentConfig

from intramind_runtime.api import create_app
from intramind_runtime.client import RuntimeClient
from intramind_runtime.contracts import EngineResult
from intramind_runtime.executor import Executor
from intramind_runtime.preparation import LlamaCppPromptSizer
from intramind_runtime.temporal_adapter import BrokerActivities, OutboxPublisher

pytestmark = pytest.mark.integration


async def test_real_rewrite_children_publish_and_replay(store, monkeypatch):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment and opt-in required")
    from api.background.rewrite import activities as rewrite_activities
    from api.background.rewrite.activities import RewriteActivities
    from api.background.rewrite.workflows import WORKFLOWS
    from api.config import RewriteSettings

    address = os.environ["RUNTIME_TEST_TEMPORAL_ADDRESS"]
    if not address.startswith("127.0.0.1:"):
        pytest.fail("disposable loopback Temporal required")
    uid, namespace = uuid4().hex, "intramind-runtime-test"
    queue, token = "rewrite-test-" + uid, "rewrite-test-service-token-1234567890"
    client = await Client.connect(address, namespace=namespace)
    blobs = MemoryArtifacts()
    settings = RewriteSettings()
    settings.llm.concurrency = 1
    settings.llm.timeout_seconds = 13
    settings.limits.segment_target_chars = 200
    monkeypatch.setattr(rewrite_activities.config, "get_settings", lambda: settings)
    source = "\n\n".join(f"Đoạn văn số {i} trình bày công tác quản lý đô thị trên địa bàn phường "
        "năm 2025. Các đơn vị thực hiện nhiệm vụ theo đúng kế hoạch và báo cáo kết quả đầy đủ."
        for i in range(4))
    monkeypatch.setattr(rewrite_activities, "load_document", lambda *a, **k: ("Báo cáo", source))
    spec = pool(target=1).model_copy(update={"context_limit": 16384})
    await store.configure_pool(spec, 1)

    async def tokenize(request):
        if request.url.path.endswith("apply-template"):
            return httpx.Response(200, json={"prompt": "fake template"})
        return httpx.Response(200, json={"tokens": [1] * 10})

    sizing_client = httpx.AsyncClient(base_url="http://fake/", transport=httpx.MockTransport(tokenize))
    preparer = LlamaCppPromptSizer(sizing_client, model="test", capacity_profile_id=spec.profile_id,
                                  context_limit=spec.context_limit, token_margin=2, expected_output_tokens=50)
    app = create_app(store, blobs, token, {"rewrite/v1": {"deadline_seconds": 120,
        "budget_limit": 20000, "task_queue": queue}}, queue, {"test": preparer})

    def runtime_client(tenant):
        return RuntimeClient("http://runtime", token, tenant, client=httpx.AsyncClient(
            base_url="http://runtime", transport=httpx.ASGITransport(app),
            headers={"Authorization": f"Bearer {token}", "X-Tenant-ID": tenant}))

    class EchoEngine:
        calls = []
        timeouts = []

        async def execute(self, reservation, payload):
            self.calls.append(reservation.attempt_id)
            self.timeouts.append(reservation.operation.attempt_timeout_seconds)
            settings.llm.concurrency = 16
            settings.llm.timeout_seconds = 1
            user = payload["messages"][-1]["content"]
            body = user.split('"""')[1].strip("\n")
            return EngineResult(body={"choices": [{"message": {"content": body}}]},
                                input_tokens=10, output_tokens=30)

    engine = EchoEngine()
    executor = Executor(store, blobs, engine, "p", "rewrite-executor")
    publisher = OutboxPublisher(store, client, "rewrite-outbox")
    broker = BrokerActivities(store, blobs)
    features = RewriteActivities(runtime_client, model_profile="test")
    version = WorkerDeploymentVersion("rewrite-test-" + uid, "build-1")
    worker = Worker(client, task_queue=queue, workflows=WORKFLOWS,
        activities=[broker.submit_or_attach, broker.finish, *features.registered()],
        deployment_config=WorkerDeploymentConfig(version=version, use_worker_versioning=True,
            default_versioning_behavior=VersioningBehavior.PINNED))

    async def pump():
        while True:
            await publisher.tick()
            await executor.tick()
            await asyncio.sleep(0.01)

    async with worker:
        for i in range(30):
            try:
                await client.workflow_service.set_worker_deployment_current_version(
                    SetWorkerDeploymentCurrentVersionRequest(namespace=namespace,
                        deployment_name=version.deployment_name, build_id=version.build_id,
                        identity="rewrite-integration-test"))
                break
            except RPCError:
                if i == 29:
                    raise
                await asyncio.sleep(0.5)
        api_client = runtime_client("t")
        try:
            ref = await api_client.put_json({"document_id": "fake-doc", "mode": "restyle", "style": "hanh_chinh"})
            submission = await api_client.submit({"task_type": "rewrite/v1", "submission_key": uid, "input": ref})
            work = asyncio.create_task(pump())
            try:
                for _ in range(900):
                    status = await api_client.get_run(submission["run_id"])
                    if status["state"] != "RUNNING":
                        break
                    await asyncio.sleep(0.05)
                assert status["state"] == "SUCCEEDED", status
                result = await api_client.read_json(status["result"])
                assert result["content"] == source + "\n"
                assert result["segments_rewritten"] >= 2
                assert result["segments_failed"] == 0
                assert len(engine.calls) == result["llm_calls"]
                assert engine.timeouts == [13] * result["llm_calls"]
                before = len(engine.calls)
                handle = client.get_workflow_handle(submission["run_id"])
                await asyncio.wait_for(handle.result(), timeout=10)
                history = await handle.fetch_history()
                assert peak_children(history) == 1
                await Replayer(workflows=WORKFLOWS).replay_workflow(history)
                for event in history.events:
                    if event.HasField("child_workflow_execution_started_event_attributes"):
                        child_id = event.child_workflow_execution_started_event_attributes.workflow_execution.workflow_id
                        child_history = await client.get_workflow_handle(child_id).fetch_history()
                        await Replayer(workflows=WORKFLOWS).replay_workflow(child_history)
                assert len(engine.calls) == before
            finally:
                with suppress(RPCError):
                    handle = client.get_workflow_handle(submission["run_id"])
                    if (await handle.describe()).close_time is None:
                        await handle.terminate("disposable rewrite test cleanup")
                work.cancel()
                with suppress(asyncio.CancelledError):
                    await work
        finally:
            await api_client.close()
            await sizing_client.aclose()
