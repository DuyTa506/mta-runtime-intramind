"""Real Temporal/PostgreSQL integration, including history replay."""

import asyncio
from contextlib import suppress
from datetime import timedelta
from uuid import uuid4

import pytest
from conftest import pool, root, temporal_test_client
from deadline_workflows import deadline_feature
from fakes import IndependentEngine, MemoryArtifacts, TestPermits
from feature_harness import feature_environment
from temporalio.api.workflowservice.v1 import (
    SetWorkerDeploymentCurrentVersionRequest,
)
from temporalio.common import VersioningBehavior, WorkerDeploymentVersion
from temporalio.service import RPCError
from temporalio.worker import Replayer, Worker, WorkerDeploymentConfig

from intramind_runtime.admin import worker_health
from intramind_runtime.example_workflow import contract_feature
from intramind_runtime.executor import Executor
from intramind_runtime.temporal_adapter import BrokerActivities, OutboxPublisher

pytestmark = pytest.mark.integration


async def test_phase_expiry_completes_and_replays_while_backend_cleanup_is_pending(store):
    gate, completed, jobs = asyncio.Event(), [], []

    async def respond(reservation, payload):
        async def compute():
            await gate.wait()
            completed.append(reservation.attempt_id)
            return {"choices": [{"message": {"content": "late"}}]}

        task = asyncio.create_task(compute())
        jobs.append(task)
        return await asyncio.shield(task)

    try:
        async with feature_environment(
            store, name="deadline-contract", workflows=[deadline_feature],
            build_activities=lambda factory: [], respond=respond,
        ) as env:
            status, _ = await env.submit({"messages": [{"role": "user", "content": "review"}]})
            assert status["state"] == "PARTIAL"
            assert status["terminal_reason"] == "assessment_timeout"
            assert status["cleanup_pending"] and status["reserved"] == 30
            assert len(env.calls) == 1 and completed == []
            await env.replay(status["root_id"])
            assert len(env.calls) == 1 and completed == []
            gate.set()
            await asyncio.gather(*jobs)
            await store.confirm_epoch_stopped("p", "e1", "all isolated engine jobs joined")
            state = await store.run(status["root_id"], "user:test")
            assert state["state"] == "PARTIAL"
            assert not state["cleanup_pending"] and state["spent"] == 30
    finally:
        gate.set()
        await asyncio.gather(*jobs)


async def test_real_temporal_async_completion_and_replay(store):
    client = await temporal_test_client()
    namespace = client.namespace
    uid = uuid4().hex
    queue = "runtime-test-"+uid
    root_spec = root(root_id=uid)
    await store.create_root(root_spec)
    await store.configure_pool(pool(target=1), 1)
    blobs = MemoryArtifacts()
    ref = await blobs.put("t", b'{"messages":[{"role":"user","content":"test"}]}')
    engine = IndependentEngine()
    engine.gate.set()
    executor = Executor(store, blobs, engine, "p", "executor", TestPermits())
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
