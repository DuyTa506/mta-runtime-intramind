"""Disposable Temporal/Postgres feature harness with actual broker accounting."""

import asyncio
import os
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from conftest import pool
from fakes import MemoryArtifacts
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


@asynccontextmanager
async def feature_environment(
    store, *, name, workflows, build_activities, respond, allow_tool_calls=False
):
    address = os.environ.get("RUNTIME_TEST_TEMPORAL_ADDRESS", "")
    if not address.startswith("127.0.0.1:"):
        pytest.fail("disposable loopback Temporal required")
    namespace, uid = "intramind-runtime-test", uuid4().hex
    queue, token = name + "-" + uid, "test-runtime-feature-token-1234567890"
    temporal = await Client.connect(address, namespace=namespace)
    blobs, calls, runs = MemoryArtifacts(), [], []
    spec = pool(target=1).model_copy(update={"context_limit": 16384})
    await store.configure_pool(spec, 1)

    async def tokenize(request):
        return httpx.Response(200, json={"prompt": "test template", "tokens": [1] * 10})

    sizing = httpx.AsyncClient(base_url="http://fake/", transport=httpx.MockTransport(tokenize))
    preparer = LlamaCppPromptSizer(
        sizing,
        model="test",
        capacity_profile_id=spec.profile_id,
        context_limit=spec.context_limit,
        token_margin=2,
        expected_output_tokens=50,
        response_formats=frozenset({"text", "json_schema", "json_object"}),
        allow_tool_calls=allow_tool_calls,
    )
    app = create_app(
        store,
        blobs,
        token,
        {
            name + "/v1": {"deadline_seconds": 120, "budget_limit": 1000000, "task_queue": queue},
        },
        queue,
        {"test": preparer},
    )

    def runtime_client(tenant):
        return RuntimeClient(
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
        async def execute(self, reservation, payload):
            calls.append(reservation.attempt_id)
            body = await respond(reservation, payload)
            return EngineResult(body=body, input_tokens=10, output_tokens=20)

    executor = Executor(store, blobs, Engine(), "p", "feature-executor-" + uid)
    publisher = OutboxPublisher(store, temporal, "feature-publisher-" + uid)
    broker = BrokerActivities(store, blobs)
    version = WorkerDeploymentVersion("feature-" + uid, "build-1")
    worker = Worker(
        temporal,
        task_queue=queue,
        workflows=workflows,
        activities=[broker.submit_or_attach, broker.finish, *build_activities(runtime_client)],
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

    async def submit(source, *, tenant="user:test"):
        api = runtime_client(tenant)
        try:
            ref = await api.put_json(source)
            result = await api.submit(
                {"task_type": name + "/v1", "submission_key": uid, "input": ref}
            )
            runs.append(result["run_id"])
            for _ in range(2000):
                status = await api.get_run(result["run_id"])
                if status["state"] != "RUNNING":
                    break
                await asyncio.sleep(0.05)
            assert status["state"] in {"SUCCEEDED", "PARTIAL"}, status
            handle = temporal.get_workflow_handle(result["run_id"])
            await asyncio.wait_for(handle.result(), timeout=10)
            return status, await api.read_json(status["result"])
        finally:
            await api.close()

    async def replay(run_id):
        history = await temporal.get_workflow_handle(run_id).fetch_history()
        await Replayer(workflows=workflows).replay_workflow(history)
        for event in history.events:
            if event.HasField("child_workflow_execution_started_event_attributes"):
                child = event.child_workflow_execution_started_event_attributes.workflow_execution.workflow_id
                await replay(child)

    async with worker:
        for attempt in range(30):
            try:
                await temporal.workflow_service.set_worker_deployment_current_version(
                    SetWorkerDeploymentCurrentVersionRequest(
                        namespace=namespace,
                        deployment_name=version.deployment_name,
                        build_id=version.build_id,
                        identity="feature-integration-test",
                    )
                )
                break
            except RPCError:
                if attempt == 29:
                    raise
                await asyncio.sleep(0.5)
        work = asyncio.create_task(pump())
        try:
            yield SimpleNamespace(
                submit=submit,
                replay=replay,
                calls=calls,
                blobs=blobs,
                runtime_client=runtime_client,
                temporal=temporal,
            )
        finally:
            for run_id in runs:
                handle = temporal.get_workflow_handle(run_id)
                with suppress(RPCError):
                    if (await handle.describe()).close_time is None:
                        await handle.terminate("disposable feature test cleanup")
            work.cancel()
            with suppress(asyncio.CancelledError):
                await work
            await sizing.aclose()
