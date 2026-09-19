"""Real Temporal/PostgreSQL integration, including history replay."""

import asyncio
import os
from contextlib import suppress
from datetime import timedelta
from uuid import uuid4

import pytest
from conftest import pool, root
from fakes import IndependentEngine, MemoryArtifacts
from google.protobuf.duration_pb2 import Duration
from temporalio.api.workflowservice.v1 import (
    RegisterNamespaceRequest,
    SetWorkerDeploymentCurrentVersionRequest,
)
from temporalio.client import Client
from temporalio.common import VersioningBehavior, WorkerDeploymentVersion
from temporalio.service import RPCError, RPCStatusCode
from temporalio.worker import Replayer, Worker, WorkerDeploymentConfig

from intramind_runtime.admin import worker_health
from intramind_runtime.example_workflow import contract_feature
from intramind_runtime.executor import Executor
from intramind_runtime.temporal_adapter import BrokerActivities, OutboxPublisher

pytestmark = pytest.mark.integration


async def test_real_temporal_async_completion_and_replay(store):
    address = os.environ.get("RUNTIME_TEST_TEMPORAL_ADDRESS")
    if not address:
        pytest.skip("explicit disposable RUNTIME_TEST_TEMPORAL_ADDRESS required")
    if not address.startswith("127.0.0.1:"):
        pytest.fail("test requires a disposable loopback Temporal endpoint")
    namespace = "intramind-runtime-test"
    client = await Client.connect(address, namespace=namespace)
    try:
        await client.workflow_service.register_namespace(RegisterNamespaceRequest(
            namespace=namespace, workflow_execution_retention_period=Duration(seconds=86400)))
    except RPCError as exc:
        if exc.status != RPCStatusCode.ALREADY_EXISTS:
            raise
    uid = uuid4().hex
    queue = "runtime-test-"+uid
    root_spec = root(root_id=uid)
    await store.create_root(root_spec)
    await store.configure_pool(pool(target=1), 1)
    blobs = MemoryArtifacts()
    ref = await blobs.put("t", b'{"messages":[{"role":"user","content":"test"}]}')
    engine = IndependentEngine()
    engine.gate.set()
    executor = Executor(store, blobs, engine, "p", "executor")
    publisher = OutboxPublisher(store, client, "publisher")
    activities = BrokerActivities(store)
    version = WorkerDeploymentVersion("runtime-test-"+uid, "build-1")
    worker = Worker(client, task_queue=queue, workflows=[contract_feature],
        activities=[activities.submit_or_attach, activities.finish],
        deployment_config=WorkerDeploymentConfig(version=version, use_worker_versioning=True,
            default_versioning_behavior=VersioningBehavior.PINNED))

    async def pump():
        while True:
            await executor.tick()
            await publisher.tick()
            await asyncio.sleep(0.05)

    async with worker:
        for i in range(30):
            try:
                await client.workflow_service.set_worker_deployment_current_version(
                    SetWorkerDeploymentCurrentVersionRequest(namespace=namespace,
                        deployment_name=version.deployment_name, build_id=version.build_id,
                        identity="runtime-integration-test"))
                break
            except RPCError:
                if i == 29:
                    raise
                await asyncio.sleep(0.5)
        for _ in range(30):
            health = await worker_health(
                client, [(version.deployment_name, version.build_id, queue, "both")]
            )
            if health["status"] == "ready":
                break
            await asyncio.sleep(0.5)
        assert health["status"] == "ready", health
        task = asyncio.create_task(pump())
        try:
            handle = await client.start_workflow("runtime-contract/v1", {
                "root_id": uid, "tenant_id": "t", "deadline": root_spec.deadline.isoformat(),
                "control_queue": queue, "input": {"payload": ref.model_dump(mode="json")},
            }, id=uid, task_queue=queue, execution_timeout=timedelta(minutes=1))
            result = await asyncio.wait_for(handle.result(), timeout=45)
            assert result["sha256"]
            assert len(engine.calls) == 1
            assert (await store.run(uid, "t"))["state"] == "SUCCEEDED"
            history = await handle.fetch_history()
            await Replayer(workflows=[contract_feature]).replay_workflow(history)
            assert len(engine.calls) == 1
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
