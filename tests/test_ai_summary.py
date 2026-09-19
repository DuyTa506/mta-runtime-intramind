"""Actual Temporal summary decisions, fake inference, and publication failure replay."""

import asyncio
import os
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest
from conftest import pool, temporal_test_client
from fakes import MemoryArtifacts
from feature_harness import peak_children
from temporalio.api.workflowservice.v1 import SetWorkerDeploymentCurrentVersionRequest
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


@pytest.mark.parametrize("mode, expected_calls", [("abstractive", 3), ("chapter", 2)])
async def test_summary_children_retry_publication_without_repeating_inference(
    store, monkeypatch, mode, expected_calls
):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment and opt-in required")
    from api.background.summary.activities import SummaryActivities
    from api.background.summary.workflows import WORKFLOWS
    from tools.summary import SummaryTool

    client = await temporal_test_client()
    namespace, uid = client.namespace, uuid4().hex
    queue, token = "summary-test-" + uid, "summary-test-service-token-1234567890"
    blobs = MemoryArtifacts()
    spec = pool(target=1).model_copy(update={"context_limit": 16384})
    await store.configure_pool(spec, 1)

    monkeypatch.setattr(
        SummaryTool, "_init_tokenizer", lambda self: setattr(self, "tokenizer", None)
    )
    llm = MagicMock()
    llm.get_model_info.return_value = {"model_name": "test", "provider": "fake"}
    llm.agenerate = AsyncMock(side_effect=AssertionError("activities must not infer"))
    tool = SummaryTool(llm, {"context_window": 16384, "map_concurrency": 1})
    tool._count_tokens = lambda text: len(text.split())
    tool._get_document_text = AsyncMock(return_value="Nguồn có thời hạn và ngoại lệ. " * 20)
    tool.chunker.mindmap_chunk = MagicMock(
        return_value=[
            {"content": "Phần thứ nhất có thời hạn ngày 15.", "level": 1, "title": "A"},
            {"content": "Phần thứ hai có ngoại lệ phải phê duyệt.", "level": 1, "title": "B"},
        ]
    )
    tool.chunker.last_outline_source = "test"

    async def tokenize(request):
        if request.url.path.endswith("apply-template"):
            return httpx.Response(200, json={"prompt": "fake template"})
        return httpx.Response(200, json={"tokens": [1] * 10})

    sizing = httpx.AsyncClient(base_url="http://fake/", transport=httpx.MockTransport(tokenize))
    preparer = LlamaCppPromptSizer(
        sizing,
        model="test",
        capacity_profile_id=spec.profile_id,
        context_limit=spec.context_limit,
        token_margin=2,
        expected_output_tokens=50,
    )
    app = create_app(
        store,
        blobs,
        token,
        {
            "summary/v1": {
                "deadline_seconds": 120,
                "budget_limit": 100000,
                "task_queue": queue,
            }
        },
        queue,
        {"test": preparer},
    )
    publish_attempts = []

    class PublicationClient(RuntimeClient):
        async def put_json(self, value):
            ref = await super().put_json(value)
            if "summary" in value and "metadata" in value:
                publish_attempts.append(ref)
                if len(publish_attempts) == 1:
                    raise RuntimeError("simulated acknowledgement lost after artifact write")
            return ref

    def runtime_client(tenant):
        return PublicationClient(
            "http://runtime",
            token,
            tenant,
            client=httpx.AsyncClient(
                base_url="http://runtime",
                transport=httpx.ASGITransport(app=app),
                headers={"Authorization": "Bearer " + token, "X-Tenant-ID": tenant},
            ),
        )

    class Engine:
        def __init__(self):
            self.calls = []

        async def execute(self, reservation, payload):
            self.calls.append(reservation.attempt_id)
            tool.config["map_concurrency"] = 16
            return EngineResult(
                body={
                    "choices": [
                        {"message": {"content": "Giữ thời hạn ngày 15 và ngoại lệ cần phê duyệt."}}
                    ]
                },
                input_tokens=10,
                output_tokens=20,
            )

    engine = Engine()
    executor = Executor(store, blobs, engine, "p", "summary-executor-" + uid)
    publisher = OutboxPublisher(store, client, "summary-publisher-" + uid)
    broker = BrokerActivities(store, blobs)
    features = SummaryActivities(runtime_client, model_profile="test", tool_factory=lambda: tool)
    version = WorkerDeploymentVersion("summary-test-" + uid, "build-1")
    worker = Worker(
        client,
        task_queue=queue,
        workflows=WORKFLOWS,
        activities=[broker.submit_or_attach, broker.finish, *features.registered()],
        deployment_config=WorkerDeploymentConfig(
            version=version,
            use_worker_versioning=True,
            default_versioning_behavior=VersioningBehavior.PINNED,
        ),
    )

    async def pump():
        while True:
            await publisher.tick()
            await executor.tick()
            await asyncio.sleep(0.01)

    async with worker:
        for attempt in range(30):
            try:
                await client.workflow_service.set_worker_deployment_current_version(
                    SetWorkerDeploymentCurrentVersionRequest(
                        namespace=namespace,
                        deployment_name=version.deployment_name,
                        build_id=version.build_id,
                        identity="summary-integration-test",
                    )
                )
                break
            except RPCError:
                if attempt == 29:
                    raise
                await asyncio.sleep(0.5)
        api = runtime_client("summary-tenant")
        work = asyncio.create_task(pump())
        try:
            source = await api.put_json({"document_id": "fake-document", "summary_type": mode})
            submitted = await api.submit(
                {"task_type": "summary/v1", "submission_key": uid, "input": source}
            )
            for _ in range(1200):
                status = await api.get_run(submitted["run_id"])
                if status["state"] != "RUNNING":
                    break
                await asyncio.sleep(0.05)
            assert status["state"] == "SUCCEEDED", status
            handle = client.get_workflow_handle(submitted["run_id"])
            await asyncio.wait_for(handle.result(), timeout=10)
            result = await api.read_json(status["result"])
            assert result["summary"]
            assert result["metadata"]["initial_chunks"] == 2
            assert len(engine.calls) == expected_calls
            assert len(publish_attempts) == 2
            assert publish_attempts[0] == publish_attempts[1]
            tool._get_document_text.assert_awaited_once_with("fake-document")
            llm.agenerate.assert_not_awaited()
            history = await handle.fetch_history()
            assert peak_children(history) == 1
            await Replayer(workflows=WORKFLOWS).replay_workflow(history)
            for event in history.events:
                if event.HasField("child_workflow_execution_started_event_attributes"):
                    child = event.child_workflow_execution_started_event_attributes.workflow_execution.workflow_id
                    await Replayer(workflows=WORKFLOWS).replay_workflow(
                        await client.get_workflow_handle(child).fetch_history()
                    )
            assert len(engine.calls) == expected_calls
        finally:
            work.cancel()
            with suppress(asyncio.CancelledError):
                await work
            await api.close()
            await sizing.aclose()
