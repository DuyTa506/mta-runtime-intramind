"""Feature facade identity, command guards and bounded history rollover."""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    CancelledError,
    ChildWorkflowError,
)
from temporalio.exceptions import TimeoutError as ActivityTimeoutError

from intramind_runtime import sdk


def failed_activity(cause):
    error = ActivityError(
        "model activity failed",
        scheduled_event_id=1,
        started_event_id=2,
        identity="worker",
        activity_type="llm",
        activity_id="call",
        retry_state=None,
    )
    error.__cause__ = cause
    return error


@pytest.fixture
def runtime_clock(monkeypatch):
    now = datetime(2026, 9, 19, 12, tzinfo=UTC)
    info = SimpleNamespace(task_queue="features", start_time=now)
    monkeypatch.setattr(sdk.workflow, "info", lambda: info)
    monkeypatch.setattr(sdk.workflow, "now", lambda: now)
    return now, info


def context(now: datetime, *, policy: sdk.TaskPolicy | None = None, **extra) -> sdk.TaskContext:
    envelope = {
        "root_id": "root",
        "tenant_id": "tenant",
        "deadline": (now + timedelta(hours=1)).isoformat(),
        "control_queue": "control",
    } | extra
    return sdk.TaskContext(envelope, policy or sdk.TaskPolicy(max_iterations=3))


@pytest.mark.parametrize("seconds", [None, 12.5])
async def test_attempt_policy_does_not_shorten_durable_wait_or_change_legacy_command(
    runtime_clock, monkeypatch, seconds
):
    now, _ = runtime_clock
    ctx = context(now)
    execute = AsyncMock(return_value={})
    monkeypatch.setattr(sdk.workflow, "execute_activity", execute)
    ref = {"key": "payload", "sha256": "a" * 64, "size": 5}
    await ctx.llm(
        key="call", payload=ref, model_profile="test", input_tokens_bound=10,
        max_output_tokens=20, expected_cost=15, attempt_timeout_seconds=seconds,
    )
    command = execute.await_args.args[1]
    if seconds is None:
        assert "attempt_timeout_seconds" not in command
    else:
        assert command["attempt_timeout_seconds"] == seconds
    assert execute.await_args.kwargs["start_to_close_timeout"] == timedelta(hours=1)
    assert execute.await_args.kwargs["schedule_to_close_timeout"] == timedelta(hours=1)
    assert "deadline" not in command


async def test_operation_deadline_bounds_the_phase_wait_without_changing_attempt_timeout(
    runtime_clock, monkeypatch
):
    now, _ = runtime_clock
    ctx = context(now)
    execute = AsyncMock(return_value={})
    monkeypatch.setattr(sdk.workflow, "execute_activity", execute)
    deadline = now + timedelta(seconds=90)
    await ctx.llm(
        key="call", payload={"key": "payload", "sha256": "a" * 64, "size": 5},
        model_profile="test", input_tokens_bound=10, max_output_tokens=20,
        expected_cost=15, deadline=deadline,
    )
    assert datetime.fromisoformat(execute.await_args.args[1]["deadline"]) == deadline
    assert execute.await_args.kwargs["start_to_close_timeout"] == timedelta(seconds=90)
    assert execute.await_args.kwargs["schedule_to_close_timeout"] == timedelta(seconds=90)


async def test_operation_deadline_failure_cannot_be_replayed_as_a_model_repair(
    runtime_clock, monkeypatch
):
    ctx = context(runtime_clock[0])
    failure = failed_activity(ApplicationError(
        "operation_deadline_exceeded", type="OperationFailed", non_retryable=True))
    monkeypatch.setattr(sdk.workflow, "execute_activity", AsyncMock(side_effect=failure))
    with pytest.raises(sdk.OperationDeadlineExceeded):
        await ctx.llm_outcome(
            key="call", payload={"key": "payload", "sha256": "a" * 64, "size": 5},
            model_profile="test", input_tokens_bound=10, max_output_tokens=20, expected_cost=15,
        )


async def test_unavailable_broker_cannot_extend_a_phase_or_trigger_model_repair(runtime_clock, monkeypatch):
    now, _ = runtime_clock
    ctx = context(now)

    async def execute(*args, **kwargs):
        monkeypatch.setattr(sdk.workflow, "now", lambda: now + timedelta(seconds=90))
        raise failed_activity(ActivityTimeoutError("timed out", type=None, last_heartbeat_details=[]))

    monkeypatch.setattr(sdk.workflow, "execute_activity", execute)
    with pytest.raises(sdk.OperationDeadlineExceeded):
        await ctx.llm_outcome(
            key="call", payload={"key": "payload", "sha256": "a" * 64, "size": 5},
            model_profile="test", input_tokens_bound=10, max_output_tokens=20, expected_cost=15,
            deadline=now + timedelta(seconds=90),
        )


async def test_phase_activity_uses_one_absolute_deadline_across_attempts(runtime_clock, monkeypatch):
    now, _ = runtime_clock
    ctx = context(now)
    execute = AsyncMock(return_value={})
    monkeypatch.setattr(sdk.workflow, "execute_activity", execute)
    deadline = now + timedelta(seconds=90)
    await ctx.activity("search", {}, key="evidence", timeout_seconds=300, deadline=deadline)
    assert execute.await_args.kwargs["schedule_to_close_timeout"] == timedelta(seconds=90)
    assert execute.await_args.kwargs["start_to_close_timeout"] == timedelta(seconds=90)
    monkeypatch.setattr(sdk.workflow, "now", lambda: now + timedelta(seconds=80))
    await ctx.activity("search", {}, key="evidence-2", timeout_seconds=300, deadline=deadline)
    assert execute.await_args.kwargs["schedule_to_close_timeout"] == timedelta(seconds=10)
    with pytest.raises(sdk.OperationDeadlineExceeded):
        await ctx.activity("search", {}, key="late", deadline=now)
    assert execute.await_count == 2


@pytest.mark.parametrize("elapsed", [10, 90])
async def test_activity_timeout_is_a_phase_expiry_only_after_its_original_deadline(
    runtime_clock, monkeypatch, elapsed
):
    now, _ = runtime_clock
    ctx = context(now)

    async def execute(*args, **kwargs):
        monkeypatch.setattr(sdk.workflow, "now", lambda: now + timedelta(seconds=elapsed))
        raise failed_activity(ActivityTimeoutError("timed out", type=None, last_heartbeat_details=[]))

    monkeypatch.setattr(sdk.workflow, "execute_activity", execute)
    error = sdk.OperationDeadlineExceeded if elapsed == 90 else ActivityError
    with pytest.raises(error):
        await ctx.activity("search", {}, key="search", deadline=now + timedelta(seconds=90))


async def test_configuration_survives_children_and_rollover(runtime_clock, monkeypatch):
    now, _ = runtime_clock
    ref = {"key": "config", "sha256": "a" * 64, "size": 5, "content_type": "application/json"}
    ctx = context(now, configuration=ref)
    assert ctx.configuration == ref
    execute = AsyncMock(return_value={})
    monkeypatch.setattr(sdk.workflow, "execute_child_workflow", execute)
    await ctx.run_child(task_type="child/v1", task_queue="features", key="child", inputs={})
    child = execute.await_args.args[1]
    assert child["configuration"] == ref
    assert sdk.TaskContext(child, ctx.policy).configuration == ref
    rollover = MagicMock()
    monkeypatch.setattr(sdk.workflow, "continue_as_new", rollover)
    ctx.continue_as_new({"cursor": 1})
    assert rollover.call_args.args[0]["configuration"] == ref


@pytest.mark.parametrize("kind,extra", [("speech", {}), ("embedding", {"texts_count": 2, "model_revision": "model-v1"})])
async def test_resource_command_uses_original_root_identity_and_durable_wait(runtime_clock, monkeypatch, kind, extra):
    now, _ = runtime_clock
    ctx = context(now, child_path=["clips", "3"])
    execute = AsyncMock(return_value={"key": "result"})
    monkeypatch.setattr(sdk.workflow, "execute_activity", execute)
    result = await getattr(ctx, kind)(
        key="voice", payload={"key": "payload", "sha256": "a" * 64, "size": 5},
        model_profile="voice", capacity_profile_id="voice-v1", characters_bound=30,
        expected_cost=30, attempt_timeout_seconds=25, **extra,
    )
    assert result == {"key": "result"}
    name, command = execute.await_args.args
    assert name == f"runtime.submit_or_attach_{kind}"
    assert command["root_id"] == "root" and command["tenant_id"] == "tenant"
    assert command["operation_id"] == ctx.key("voice") and command["kind"] == kind
    assert "max_output_tokens" not in command
    assert execute.await_args.kwargs["start_to_close_timeout"] == timedelta(hours=1)


@pytest.mark.parametrize("reason,allowed", [
    ("speech_backend_503", True), ("speech_backend_500", True), ("max_attempts", True),
    ("root_budget_exhausted", False), ("root_attempt_budget", False),
    ("capacity_profile_changed", False), ("deadline_exceeded", False),
    ("embedding_invalid_vectors", False), ("embedding_model_revision_changed", False),
    ("backend_epoch_stopped", False),
])
@pytest.mark.parametrize("kind", ["speech", "embedding"])
async def test_native_fallback_does_not_swallow_root_or_compatibility_failure(
    runtime_clock, monkeypatch, reason, allowed, kind
):
    reason = reason.replace("speech_backend_", kind + "_backend_")
    ctx = context(runtime_clock[0])
    failure = failed_activity(ApplicationError(reason, type="OperationFailed", non_retryable=True))
    monkeypatch.setattr(ctx, kind, AsyncMock(side_effect=failure))
    outcome = getattr(ctx, kind + "_outcome")
    if allowed:
        assert await outcome() == {"error": reason}
    else:
        with pytest.raises(ActivityError):
            await outcome()


@pytest.mark.parametrize("window,expected", [(1, 1), (3, 3), (8, 4), (None, 4)])
async def test_feature_window_bounds_materialization_and_preserves_order(
    runtime_clock, monkeypatch, window, expected
):
    now, _ = runtime_clock
    ctx = context(now, policy=sdk.TaskPolicy(max_iterations=2, child_window=4))
    active, peak = 0, 0

    async def child(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.001)
        active -= 1
        return kwargs["inputs"]

    monkeypatch.setattr(ctx, "run_child", child)
    items = [{"index": i} for i in range(9)]
    result = await ctx.map_children(
        task_type="child/v1",
        task_queue="features",
        items=items,
        item_key="index",
        key="phase",
        window=window,
    )
    assert result == items
    assert peak == expected


@pytest.mark.parametrize("window", [0, -1, True, 2.5])
async def test_invalid_feature_window_cannot_start_children(runtime_clock, monkeypatch, window):
    now, _ = runtime_clock
    ctx = context(now)
    child = AsyncMock()
    monkeypatch.setattr(ctx, "run_child", child)
    with pytest.raises(ValueError, match="window"):
        await ctx.map_children(
            task_type="child/v1",
            task_queue="features",
            items=[{"id": 1}],
            item_key="id",
            key="phase",
            window=window,
        )
    child.assert_not_called()


def test_child_identity_preserves_component_boundaries(runtime_clock):
    now, _ = runtime_clock
    nested = context(now, child_path=["section", "repair"])
    literal_separator = context(now, child_path=["section/repair"])
    other_root = context(now, root_id="root/section", child_path=["repair"])
    keys = {
        nested.key("write/v1"),
        literal_separator.key("write/v1"),
        other_root.key("write/v1"),
        context(now, child_path=["section"]).key("repair/write/v1"),
    }
    assert len(keys) == 4
    assert context(now, child_path=["section", "repair"]).key("write/v1") == nested.key("write/v1")


@pytest.mark.parametrize("child_path", [[""], [1], ["child"] * 65])
def test_invalid_child_identity_is_rejected(runtime_clock, child_path):
    now, _ = runtime_clock
    with pytest.raises(ValueError, match="child path"):
        context(now, child_path=child_path)


async def test_feature_activity_has_finite_attempts_and_original_deadline(
    runtime_clock, monkeypatch
):
    now, _ = runtime_clock
    execute = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(sdk.workflow, "execute_activity", execute)
    ctx = context(now, deadline=(now + timedelta(seconds=45)).isoformat())

    assert await ctx.activity("render", {}, key="render/v1", timeout_seconds=20) == {"ok": True}

    options = execute.await_args.kwargs
    assert options["retry_policy"].maximum_attempts == 3
    assert options["schedule_to_close_timeout"] == timedelta(seconds=45)
    assert options["start_to_close_timeout"] == timedelta(seconds=20)
    assert "heartbeat_timeout" not in options
    assert ctx._active_commands == 0


async def test_heartbeat_timeout_is_set_only_when_the_activity_pulses(runtime_clock, monkeypatch):
    now, _ = runtime_clock
    execute = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(sdk.workflow, "execute_activity", execute)

    await context(now).activity("render", {}, key="render", heartbeat_timeout_seconds=30)

    assert execute.await_args.kwargs["heartbeat_timeout"] == timedelta(seconds=30)


async def test_non_positive_heartbeat_timeout_is_rejected(runtime_clock, monkeypatch):
    now, _ = runtime_clock
    execute = AsyncMock()
    monkeypatch.setattr(sdk.workflow, "execute_activity", execute)

    with pytest.raises(ValueError, match="retry policy"):
        await context(now).activity("render", {}, key="render", heartbeat_timeout_seconds=0)

    execute.assert_not_called()


@pytest.mark.parametrize("max_attempts", [0, -1, 101])
async def test_unbounded_activity_retry_policy_is_rejected(
    runtime_clock, monkeypatch, max_attempts
):
    now, _ = runtime_clock
    execute = AsyncMock()
    monkeypatch.setattr(sdk.workflow, "execute_activity", execute)

    with pytest.raises(ValueError, match="retry policy"):
        await context(now).activity("render", {}, key="render", max_attempts=max_attempts)

    execute.assert_not_called()


async def test_activity_failure_releases_rollover_guard(runtime_clock, monkeypatch):
    now, _ = runtime_clock
    monkeypatch.setattr(
        sdk.workflow, "execute_activity", AsyncMock(side_effect=RuntimeError("render failed"))
    )
    ctx = context(now)

    with pytest.raises(RuntimeError, match="render failed"):
        await ctx.activity("render", {}, key="render")

    assert ctx._active_commands == 0


@pytest.mark.parametrize("command", ["activity", "child"])
async def test_rollover_refuses_inflight_commands(runtime_clock, monkeypatch, command):
    now, _ = runtime_clock
    started, released = asyncio.Event(), asyncio.Event()

    async def execute(*args, **kwargs):
        started.set()
        await released.wait()
        return {}

    primitive = "execute_activity" if command == "activity" else "execute_child_workflow"
    monkeypatch.setattr(sdk.workflow, primitive, execute)
    continue_as_new = MagicMock()
    monkeypatch.setattr(sdk.workflow, "continue_as_new", continue_as_new)
    ctx = context(now)
    work = (
        ctx.activity("render", {}, key="render")
        if command == "activity"
        else ctx.run_child(task_type="child/v1", task_queue="features", key="child", inputs={})
    )
    task = asyncio.create_task(work)
    await asyncio.wait_for(started.wait(), timeout=1)
    try:
        with pytest.raises(ValueError, match="while activities, children or waits are active"):
            ctx.continue_as_new({"cursor": 1})
        assert not task.done()
        continue_as_new.assert_not_called()
    finally:
        released.set()
        await task
    assert ctx._active_commands == 0


def test_rollover_keeps_root_identity_deadline_start_and_pending_signals(
    runtime_clock, monkeypatch
):
    class RolloverRequested(BaseException):
        pass

    now, info = runtime_clock
    ctx = context(now, child_path=["a/b", "c"])
    ctx._events["review"] = {"approved": True}
    continue_as_new = MagicMock(side_effect=RolloverRequested)
    monkeypatch.setattr(sdk.workflow, "continue_as_new", continue_as_new)

    with pytest.raises(RolloverRequested):
        ctx.continue_as_new({"cursor": 12, "artifact": "immutable-reference"})

    envelope = continue_as_new.call_args.args[0]
    assert envelope["root_id"] == ctx.root_id
    assert envelope["tenant_id"] == ctx.tenant_id
    assert envelope["child_path"] == ["a/b", "c"]
    assert envelope["pending_events"] == {"review": {"approved": True}}
    assert envelope["rollovers"] == 1
    assert envelope["input"]["cursor"] == 12
    info.start_time = now + timedelta(minutes=30)
    monkeypatch.setattr(sdk.workflow, "now", lambda: now + timedelta(minutes=30))
    resumed = sdk.TaskContext(envelope, ctx.policy)
    assert resumed.started_at == ctx.started_at
    assert resumed.deadline == ctx.deadline
    assert resumed.key("same-operation") == ctx.key("same-operation")
    assert resumed.elapsed_seconds() == ctx.elapsed_seconds() == 1800


def test_rollover_budget_is_enforced_across_continuations(runtime_clock, monkeypatch):
    now, _ = runtime_clock
    continue_as_new = MagicMock()
    monkeypatch.setattr(sdk.workflow, "continue_as_new", continue_as_new)
    ctx = context(now, policy=sdk.TaskPolicy(max_iterations=1, max_rollovers=2), rollovers=2)

    with pytest.raises(ValueError, match="rollover budget exhausted"):
        ctx.continue_as_new({"cursor": 1})

    continue_as_new.assert_not_called()


def test_rollover_manifest_counts_utf8_bytes_and_pending_signal_size(runtime_clock, monkeypatch):
    now, _ = runtime_clock
    continue_as_new = MagicMock()
    monkeypatch.setattr(sdk.workflow, "continue_as_new", continue_as_new)
    ctx = context(now)

    with pytest.raises(ValueError, match="64 KiB"):
        ctx.continue_as_new({"text": "ộ" * 22_000})
    ctx._events["large-signal"] = "x" * 65_536
    with pytest.raises(ValueError, match="64 KiB"):
        ctx.continue_as_new({"cursor": 1})

    continue_as_new.assert_not_called()


async def test_early_signal_overrides_carried_signal_without_losing_other_events(
    runtime_clock, monkeypatch
):
    now, _ = runtime_clock
    finish = AsyncMock()
    monkeypatch.setattr(sdk.workflow, "execute_activity", finish)

    async def wait_condition(predicate, **kwargs):
        assert predicate(), "signal disappeared before handler resumed"

    monkeypatch.setattr(sdk.workflow, "wait_condition", wait_condition)
    observed = []
    artifact = {"key": "result", "sha256": "1" * 64, "size": 1}

    @sdk.durable_task(name="sdk-signals-test", version=1, policy=sdk.TaskPolicy(max_iterations=1))
    async def signal_feature(ctx, inputs):
        observed.extend([await ctx.wait_event("same"), await ctx.wait_event("carried")])
        return artifact

    instance = signal_feature()
    instance.event({"key": "same", "value": "fresh"})
    result = await instance.run(
        {
            "root_id": "root",
            "tenant_id": "tenant",
            "control_queue": "control",
            "deadline": (now + timedelta(hours=1)).isoformat(),
            "pending_events": {"same": "old", "carried": "kept"},
            "input": {},
        }
    )

    assert observed == ["fresh", "kept"]
    assert result == artifact
    assert instance.pending_events == {}


@pytest.mark.parametrize("failure_type,reason,allows_fallback", [
    ("OperationFailed", "backend_rejected_400", True),
    ("OperationFailed", "backend_rejected_422", True),
    ("OperationFailed", "invalid_response", True),
    ("OperationFailed", "max_attempts", True),
    ("OperationFailed", "root_budget_exhausted", False),
    ("OperationFailed", "root_attempt_budget", False),
    ("OperationFailed", "resource_budget_missing", False),
    ("OperationFailed", "context_exceeds_all_pools", False),
    ("OperationFailed", "invalid_completion_payload", False),
    ("OperationFailed", "backend_rejected_401", False),
    ("OperationFailed", "backend_rejected_403", False),
    ("OperationFailed", "backend_rejected_404", False),
    ("OperationFailed", "backend_status_503_unconfirmed", False),
    ("OperationFailed", "unknown_future_error", False),
    ("AdmissionRejected", "invalid_response", False),
    ("Timeout", "invalid_response", False),
])
async def test_only_confirmed_terminal_inference_failure_allows_fallback(
    runtime_clock, failure_type, reason, allows_fallback
):
    now, _ = runtime_clock
    ctx = context(now)
    error = failed_activity(ApplicationError(reason, type=failure_type, non_retryable=True))
    ctx.llm = AsyncMock(side_effect=error)
    if allows_fallback:
        assert await ctx.llm_outcome(key="call") == {"error": "OperationFailed"}
    else:
        with pytest.raises(ActivityError):
            await ctx.llm_outcome(key="call")


async def test_model_step_passes_committed_references_and_distinct_call_keys(runtime_clock):
    now, _ = runtime_clock
    ctx = context(now)
    final = {"key": "final", "sha256": "f" * 64, "size": 2, "content_type": "application/json"}
    first = {"key": "response", "sha256": "e" * 64, "size": 2}
    snapshots = []

    async def plan(name, inputs, **kwargs):
        snapshots.append([dict(record) for record in inputs["records"]])
        if len(snapshots) <= 2:
            return {"done": False, "request_digest": str(len(snapshots)) * 64, "request": {}}
        return {"done": True, "result": final}

    ctx.activity = AsyncMock(side_effect=plan)
    ctx.llm_outcome = AsyncMock(side_effect=[{"result": first}, {"error": "OperationFailed"}])
    assert await ctx.model_step(key="leaf", planner="leaf.plan/v1", inputs={}) == final
    assert snapshots == [
        [],
        [{"request_digest": "1" * 64, "result": first}],
        [
            {"request_digest": "1" * 64, "result": first},
            {"request_digest": "2" * 64, "error": "OperationFailed"},
        ],
    ]
    assert [call.kwargs["key"] for call in ctx.llm_outcome.await_args_list] == [
        "leaf/call/0",
        "leaf/call/1",
    ]
    assert ctx._active_commands == 0


async def test_model_step_bound_does_not_send_extra_inference(runtime_clock):
    now, _ = runtime_clock
    ctx = context(now)
    ctx.activity = AsyncMock(
        return_value={"done": False, "request_digest": "a" * 64, "request": {}}
    )
    ctx.llm_outcome = AsyncMock(return_value={"error": "OperationFailed"})
    with pytest.raises(ValueError, match="call budget exhausted"):
        await ctx.model_step(key="leaf", planner="leaf.plan/v1", inputs={}, max_model_calls=1)
    ctx.llm_outcome.assert_awaited_once()
    assert ctx._active_commands == 0


def child_failure(cause):
    error = ChildWorkflowError(
        "child failed",
        namespace="intramind",
        workflow_id="child",
        run_id="run",
        workflow_type="feature/v1",
        initiated_event_id=1,
        started_event_id=2,
        retry_state=None,
    )
    error.__cause__ = cause
    return error


def test_terminal_reason_unwraps_child_workflow_to_the_ledger_cause():
    inner = ApplicationError("engine_epoch_stopped", type="OperationFailed", non_retryable=True)
    wrapped = ApplicationError("feature failed", type="ActivityError", non_retryable=True)
    wrapped.__cause__ = failed_activity(inner)

    assert sdk.terminal_reason(child_failure(wrapped)) == "engine_epoch_stopped"
    assert sdk.is_cancellation(failed_activity(CancelledError("stopped")))


def _envelope(now):
    return {
        "root_id": "root",
        "tenant_id": "tenant",
        "control_queue": "control",
        "deadline": (now + timedelta(hours=1)).isoformat(),
        "input": {},
    }


async def test_cancelled_activity_is_not_recorded_as_feature_failed(runtime_clock, monkeypatch):
    now, _ = runtime_clock
    finish = AsyncMock()
    monkeypatch.setattr(sdk.workflow, "execute_activity", finish)

    @sdk.durable_task(name="sdk-cancel-test", version=1, policy=sdk.TaskPolicy(max_iterations=1))
    async def feature(ctx, inputs):
        raise failed_activity(CancelledError("activity cancelled"))

    with pytest.raises(ActivityError):
        await feature().run(_envelope(now))

    payload = finish.await_args.args[1]
    assert payload["state"] == "CANCELLED"
    assert payload["reason"] == "user_cancelled"


async def test_failed_feature_records_the_operation_reason(runtime_clock, monkeypatch):
    now, _ = runtime_clock
    finish = AsyncMock()
    monkeypatch.setattr(sdk.workflow, "execute_activity", finish)

    @sdk.durable_task(name="sdk-reason-test", version=1, policy=sdk.TaskPolicy(max_iterations=1))
    async def feature(ctx, inputs):
        raise failed_activity(
            ApplicationError("engine_epoch_stopped", type="OperationFailed", non_retryable=True)
        )

    with pytest.raises(ApplicationError) as caught:
        await feature().run(_envelope(now))

    assert caught.value.type == "engine_epoch_stopped"
    payload = finish.await_args.args[1]
    assert payload["state"] == "FAILED"
    assert payload["reason"] == "engine_epoch_stopped"
