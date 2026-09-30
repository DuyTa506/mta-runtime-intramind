"""Real Temporal histories for both child schedules, replayed with the gated SDK."""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from conftest import temporal_test_client
from rolling_workflows import rolling_leaf, rolling_parent
from temporalio import activity
from temporalio.api.workflowservice.v1 import SetWorkerDeploymentCurrentVersionRequest
from temporalio.client import WorkflowFailureError
from temporalio.common import VersioningBehavior, WorkerDeploymentVersion
from temporalio.service import RPCError
from temporalio.worker import (
    Replayer,
    UnsandboxedWorkflowRunner,
    Worker,
    WorkerDeploymentConfig,
)
from temporalio.workflow import NondeterminismError

from intramind_runtime import sdk

pytestmark = pytest.mark.integration

WORKFLOWS = [rolling_parent, rolling_leaf]
# Child 0 outlives the rest, so only a rolling window starts child 2 before child 0 ends.
SLOW_FIRST = [{"index": 0, "seconds": 5}] + [{"index": i, "seconds": 0.5} for i in (1, 2, 3)]
# Child 0 fails after child 1 finished; child 2 would still be running under a rolling window.
FIRST_FAILS = [
    {"index": 0, "seconds": 2, "fail": True},
    {"index": 1, "seconds": 0.5},
    {"index": 2, "seconds": 30},
    {"index": 3, "seconds": 0.5},
]


@activity.defn(name="runtime.finish_run")
async def finish_run(request: dict) -> None:
    pass


async def run_parent(client, inputs, **worker_options):
    """Run one parent to its end; returns (result or WorkflowFailureError, history)."""
    uid = uuid4().hex
    queue = "runtime-rolling-" + uid
    version = WorkerDeploymentVersion(queue, "build-1")
    worker = Worker(
        client,
        task_queue=queue,
        workflows=WORKFLOWS,
        activities=[finish_run],
        deployment_config=WorkerDeploymentConfig(
            version=version,
            use_worker_versioning=True,
            default_versioning_behavior=VersioningBehavior.PINNED,
        ),
        **worker_options,
    )
    async with worker:
        for attempt in range(30):
            try:
                await client.workflow_service.set_worker_deployment_current_version(
                    SetWorkerDeploymentCurrentVersionRequest(
                        namespace=client.namespace,
                        deployment_name=version.deployment_name,
                        build_id=version.build_id,
                        identity="runtime-rolling-test",
                    )
                )
                break
            except RPCError:
                if attempt == 29:
                    raise
                await asyncio.sleep(0.5)
        handle = await client.start_workflow(
            "runtime-rolling-contract/v1",
            {
                "root_id": uid,
                "tenant_id": "t",
                "deadline": (datetime.now(UTC) + timedelta(minutes=2)).isoformat(),
                "control_queue": queue,
                "input": inputs,
            },
            id=uid,
            task_queue=queue,
            execution_timeout=timedelta(minutes=2),
        )
        try:
            outcome = await asyncio.wait_for(handle.result(), timeout=60)
        except WorkflowFailureError as exc:
            outcome = exc
        return outcome, await handle.fetch_history()


async def child_schedule(client, history) -> list[tuple[str, int]]:
    """Child starts and terminal events by item index, and timer starts by ordinal, in history order."""
    terminals = {
        "child_workflow_execution_completed_event_attributes": "finish",
        "child_workflow_execution_failed_event_attributes": "fail",
        "child_workflow_execution_canceled_event_attributes": "cancelled",
    }
    index_by_id = {}
    schedule = []
    timers = 0
    for event in history.events:
        if event.HasField("timer_started_event_attributes"):
            timers += 1
            schedule.append(("timer", timers))
        if event.HasField("start_child_workflow_execution_initiated_event_attributes"):
            started = event.start_child_workflow_execution_initiated_event_attributes
            envelope = (await client.data_converter.decode(started.input.payloads))[0]
            index_by_id[started.workflow_id] = envelope["input"]["index"]
            schedule.append(("start", index_by_id[started.workflow_id]))
        for field, kind in terminals.items():
            if event.HasField(field):
                child = getattr(event, field).workflow_execution.workflow_id
                schedule.append((kind, index_by_id[child]))
    return schedule


def max_in_flight(schedule: list[tuple[str, int]]) -> int:
    running = peak = 0
    for kind, _ in schedule:
        running += {"start": 1, "timer": 0}.get(kind, -1)
        peak = max(peak, running)
    return peak


def patch_markers(history) -> int:
    return sum(
        event.HasField("marker_recorded_event_attributes")
        and sdk.ROLLING_CHILDREN_PATCH.encode()
        in event.marker_recorded_event_attributes.SerializeToString()
        for event in history.events
    )


async def record_before_patch(client, monkeypatch, inputs):
    with monkeypatch.context() as pre_patch:
        # The pre-patch SDK never called workflow.patched; forcing False emits its exact commands.
        pre_patch.setattr(sdk.TaskContext, "rolling_children", lambda self: False)
        return await run_parent(client, inputs, workflow_runner=UnsandboxedWorkflowRunner())


async def test_new_run_records_the_patch_rolls_the_window_and_replays():
    client = await temporal_test_client()
    result, history = await run_parent(client, {"items": SLOW_FIRST})

    assert result["key"] == "k0,k1,k2,k3"
    schedule = await child_schedule(client, history)
    assert [index for kind, index in schedule if kind == "start"] == [0, 1, 2, 3]
    assert schedule.index(("start", 2)) < schedule.index(("finish", 0))
    assert max_in_flight(schedule) == 2
    assert patch_markers(history) == 1
    await Replayer(workflows=WORKFLOWS).replay_workflow(history)


async def test_new_run_failure_cancels_the_child_in_flight_and_starts_no_more():
    client = await temporal_test_client()
    outcome, history = await run_parent(client, {"items": FIRST_FAILS})

    assert isinstance(outcome, WorkflowFailureError)
    assert await child_schedule(client, history) == [
        ("start", 0), ("start", 1), ("finish", 1), ("start", 2), ("fail", 0), ("cancelled", 2),
    ]
    await Replayer(workflows=WORKFLOWS).replay_workflow(history)


async def test_history_recorded_before_the_patch_replays_only_through_the_gate(monkeypatch):
    client = await temporal_test_client()
    result, history = await record_before_patch(
        client, monkeypatch, {"items": SLOW_FIRST, "ticks": [2.5, 0.5]}
    )

    assert result["key"] == "k0,k1,k2,k3"
    schedule = await child_schedule(client, history)
    assert schedule.index(("finish", 0)) < schedule.index(("start", 2))
    assert max_in_flight(schedule) == 2
    assert patch_markers(history) == 0
    await Replayer(workflows=WORKFLOWS).replay_workflow(history)

    # Replay matches commands by order, not by workflow task, so an earlier child
    # start only diverges when another command was recorded in between.
    assert schedule.index(("finish", 1)) < schedule.index(("timer", 2)) < schedule.index(("finish", 0))
    with monkeypatch.context() as ungated:
        ungated.setattr(sdk.TaskContext, "rolling_children", lambda self: True)
        with pytest.raises(NondeterminismError):
            await Replayer(
                workflows=WORKFLOWS, workflow_runner=UnsandboxedWorkflowRunner()
            ).replay_workflow(history)


async def test_failed_history_recorded_before_the_patch_replays_on_the_batched_path(monkeypatch):
    client = await temporal_test_client()
    outcome, history = await record_before_patch(client, monkeypatch, {"items": FIRST_FAILS})

    assert isinstance(outcome, WorkflowFailureError)
    assert await child_schedule(client, history) == [
        ("start", 0), ("start", 1), ("finish", 1), ("fail", 0),
    ]
    assert patch_markers(history) == 0
    await Replayer(workflows=WORKFLOWS).replay_workflow(history)
