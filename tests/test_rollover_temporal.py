"""Two real Temporal histories share one inference operation and root budget."""

import asyncio
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from conftest import pool, root, temporal_test_client
from fakes import IndependentEngine, MemoryArtifacts, TestPermits
from rollover_workflows import rollover_feature
from temporalio.api.workflowservice.v1 import (
    SetWorkerDeploymentCurrentVersionRequest,
)
from temporalio.common import VersioningBehavior, WorkerDeploymentVersion
from temporalio.service import RPCError
from temporalio.worker import Replayer, Worker, WorkerDeploymentConfig

from intramind_runtime.executor import Executor
from intramind_runtime.temporal_adapter import BrokerActivities, OutboxPublisher

pytestmark = pytest.mark.integration


async def test_rollover_attaches_operation_preserves_signal_and_replays_both_histories(store):
    client = await temporal_test_client()
    namespace = client.namespace

    uid = uuid4().hex
    queue = "runtime-rollover-" + uid
    root_spec = root(root_id=uid).model_copy(
        update={
            "deadline": datetime.now(UTC) + timedelta(minutes=2),
        }
    )
    await store.create_root(root_spec)
    await store.configure_pool(pool(target=1), 1)
    blobs = MemoryArtifacts()
    ref = await blobs.put("t", b'{"messages":[{"role":"user","content":"test"}]}')
    engine = IndependentEngine()
    executor = Executor(store, blobs, engine, "p", "rollover-executor", TestPermits())
    publisher = OutboxPublisher(store, client, "rollover-publisher")
    broker = BrokerActivities(store, blobs)
    version = WorkerDeploymentVersion("runtime-rollover-" + uid, "build-1")
    worker = Worker(
        client,
        task_queue=queue,
        workflows=[rollover_feature],
        activities=[broker.submit_or_attach, broker.finish],
        deployment_config=WorkerDeploymentConfig(
            version=version,
            use_worker_versioning=True,
            default_versioning_behavior=VersioningBehavior.PINNED,
        ),
    )

    async def pump():
        while True:
            await executor.tick()
            await publisher.tick()
            await asyncio.sleep(0.01)

    async with worker:
        for attempt in range(30):
            try:
                await client.workflow_service.set_worker_deployment_current_version(
                    SetWorkerDeploymentCurrentVersionRequest(
                        namespace=namespace,
                        deployment_name=version.deployment_name,
                        build_id=version.build_id,
                        identity="runtime-rollover-test",
                    )
                )
                break
            except RPCError:
                if attempt == 29:
                    raise
                await asyncio.sleep(0.5)

        work = asyncio.create_task(pump())
        try:
            handle = await client.start_workflow(
                "runtime-rollover-contract/v1",
                {
                    "root_id": uid,
                    "tenant_id": "t",
                    "deadline": root_spec.deadline.isoformat(),
                    "control_queue": queue,
                    "input": {"payload": ref.model_dump(mode="json")},
                },
                id=uid,
                task_queue=queue,
                execution_timeout=timedelta(minutes=2),
            )
            # Both signals arrive in the first history while inference is
            # active; only "rollover" is consumed before continue-as-new.
            await asyncio.wait_for(engine.started.wait(), timeout=15)
            await handle.signal(
                "event",
                {
                    "key": "carry",
                    "value": {"approval": "accepted-before-rollover"},
                },
            )
            await handle.signal("event", {"key": "rollover", "value": True})
            engine.gate.set()
            result = await asyncio.wait_for(handle.result(), timeout=45)

            assert result["sha256"]
            assert len(engine.calls) == 1
            run = await store.run(uid, "t")
            assert run["state"] == "SUCCEEDED"
            assert run["operations"] == {"SUCCEEDED": 1}
            assert run["spent"] == 20
            assert run["reserved"] == 0
            assert run["budget_limit"] == root_spec.budget_limit
            assert run["deadline"] == root_spec.deadline

            first = await client.get_workflow_handle(
                uid, run_id=handle.result_run_id
            ).fetch_history()
            continuation = next(
                event.workflow_execution_continued_as_new_event_attributes
                for event in first.events
                if event.HasField("workflow_execution_continued_as_new_event_attributes")
            )
            checkpoint = (await client.data_converter.decode(continuation.input.payloads))[0]
            assert checkpoint["pending_events"] == {
                "carry": {"approval": "accepted-before-rollover"}
            }
            assert checkpoint["root_id"] == uid
            assert checkpoint["rollovers"] == 1
            assert checkpoint["deadline"] == root_spec.deadline.isoformat()
            # SDK Info.start_time is the first workflow task's start, whereas
            # history event zero records workflow creation. Compare the value
            # captured by the first handler with the facade's carried envelope.
            assert checkpoint["started_at"] == checkpoint["input"]["started_at"]

            second = await client.get_workflow_handle(
                uid, run_id=continuation.new_execution_run_id
            ).fetch_history()
            resumed_envelope = (
                await client.data_converter.decode(
                    second.events[0].workflow_execution_started_event_attributes.input.payloads
                )
            )[0]
            assert resumed_envelope["started_at"] == checkpoint["input"]["started_at"]
            assert datetime.fromisoformat(resumed_envelope["started_at"]) < second.events[
                0
            ].event_time.ToDatetime(tzinfo=UTC)
            assert any(
                event.HasField("workflow_execution_completed_event_attributes")
                for event in second.events
            )
            operations = []
            for history in (first, second):
                for event in history.events:
                    if event.HasField("activity_task_scheduled_event_attributes"):
                        scheduled = event.activity_task_scheduled_event_attributes
                        if scheduled.activity_type.name == "runtime.submit_or_attach_llm":
                            operation = (
                                await client.data_converter.decode(scheduled.input.payloads)
                            )[0]
                            operations.append(operation["operation_id"])
                await Replayer(workflows=[rollover_feature]).replay_workflow(history)
            assert len(operations) == 2
            assert operations[0] == operations[1]
            assert len(engine.calls) == 1
            assert (await store.run(uid, "t"))["spent"] == 20
        finally:
            engine.gate.set()
            work.cancel()
            with suppress(asyncio.CancelledError):
                await work
            await asyncio.gather(*engine.jobs, return_exceptions=True)
