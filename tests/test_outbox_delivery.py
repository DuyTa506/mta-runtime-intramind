"""A bad delivery must not block unrelated durable results or disappear silently."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.exceptions import ApplicationError
from temporalio.service import RPCError, RPCStatusCode

from intramind_runtime.temporal_adapter import BrokerActivities, OutboxPublisher


def start_event(event_id: str, *, deadline: datetime | None = None) -> dict:
    return {
        "event_id": event_id,
        "kind": "start_workflow",
        "aggregate_id": event_id,
        "deliveries": 1,
        "payload": {
            "workflow_type": "test/v1",
            "task_queue": "test",
            "input": {
                "tenant_id": "tenant",
                "deadline": (deadline or datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            },
        },
    }


def publisher_for(events: list[dict], **kwargs):
    store = MagicMock()
    store.claim_events = AsyncMock(return_value=events)
    store.delivered = AsyncMock(return_value=True)
    store.defer_event = AsyncMock(return_value=True)
    store.finish_run = AsyncMock()
    store.binding = AsyncMock()
    client = MagicMock()
    client.start_workflow = AsyncMock()
    return OutboxPublisher(store, client, "publisher", **kwargs), store, client


async def test_poison_event_is_retained_while_later_event_is_delivered(caplog):
    poison = start_event("poison") | {"kind": "unsupported", "deliveries": 100_000}
    publisher, store, client = publisher_for([poison, start_event("valid")])

    assert await publisher.tick() == 2

    client.start_workflow.assert_awaited_once()
    store.delivered.assert_awaited_once_with("valid", "publisher", 1)
    deferred = store.defer_event.await_args.args
    assert deferred[:3] == ("poison", "publisher", 100_000)
    assert 1 <= deferred[3] <= 300
    assert "event_id=poison" in caplog.text


async def test_hung_rpc_cannot_hold_other_delivery_or_lease_forever():
    publisher, store, client = publisher_for(
        [start_event("hung"), start_event("valid")], delivery_timeout_seconds=0.02
    )
    valid_completed = asyncio.Event()

    async def start(*args, **kwargs):
        if kwargs["id"] == "hung":
            await asyncio.Event().wait()
        else:
            valid_completed.set()

    client.start_workflow.side_effect = start
    await asyncio.wait_for(publisher.tick(), timeout=1)

    assert valid_completed.is_set()
    store.delivered.assert_awaited_once_with("valid", "publisher", 1)
    assert store.defer_event.await_args.args[:3] == ("hung", "publisher", 1)


async def test_failure_to_persist_retry_leaves_lease_unacknowledged(caplog):
    publisher, store, _ = publisher_for(
        [start_event("broken") | {"kind": "unsupported"}, start_event("valid")]
    )
    store.defer_event.side_effect = ConnectionError("sensitive connection details")

    await publisher.tick()

    store.delivered.assert_awaited_once_with("valid", "publisher", 1)
    assert "retry persistence failed event_id=broken" in caplog.text
    assert "sensitive connection details" not in caplog.text


async def test_stale_completion_token_is_acknowledged_only_for_its_binding():
    event = start_event("completion") | {
        "kind": "complete_activity",
        "payload": {"binding_id": "binding"},
    }
    publisher, store, client = publisher_for([event])
    store.binding.return_value = {
        "state": "SUCCEEDED",
        "result": {"key": "result"},
        "task_token": b"old-token",
    }
    handle = MagicMock()
    handle.complete = AsyncMock(side_effect=RPCError("expired token", RPCStatusCode.NOT_FOUND, b""))
    client.get_async_activity_handle.return_value = handle

    await publisher.tick()

    store.delivered.assert_awaited_once_with("completion", "publisher", 1)
    store.defer_event.assert_not_awaited()


async def test_unavailable_temporal_retries_without_acknowledgement():
    publisher, store, client = publisher_for([start_event("retry")])
    client.start_workflow.side_effect = RPCError("unavailable", RPCStatusCode.UNAVAILABLE, b"")

    await publisher.tick()

    store.delivered.assert_not_awaited()
    store.defer_event.assert_awaited_once_with("retry", "publisher", 1, 1)


async def test_expired_start_fails_root_without_starting_temporal():
    event = start_event("expired", deadline=datetime.now(UTC) - timedelta(seconds=1))
    publisher, store, client = publisher_for([event])

    await publisher.tick()

    store.finish_run.assert_awaited_once_with(
        "expired", "tenant", "FAILED", reason="deadline_exceeded"
    )
    store.delivered.assert_awaited_once_with("expired", "publisher", 1)
    client.start_workflow.assert_not_awaited()


async def test_workflow_timeout_is_bounded_by_original_deadline():
    event = start_event("valid", deadline=datetime.now(UTC) + timedelta(seconds=15))
    publisher, _, client = publisher_for([event])

    await publisher.tick()

    timeout = client.start_workflow.await_args.kwargs["execution_timeout"]
    assert timedelta(0) < timeout <= timedelta(seconds=15)


async def test_process_cancellation_keeps_delivery_recoverable():
    publisher, store, client = publisher_for([start_event("interrupted")])
    started = asyncio.Event()

    async def start(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    client.start_workflow.side_effect = start
    task = asyncio.create_task(publisher.tick())
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    store.delivered.assert_not_awaited()
    store.defer_event.assert_not_awaited()


@pytest.mark.parametrize(
    ("state", "reason"),
    [("FAILED", "deadline_exceeded"), ("CANCELLED", "user_cancelled")],
)
async def test_rejected_terminal_commit_cannot_complete_temporal_as_success(state, reason):
    store = MagicMock()
    store.finish_run = AsyncMock()
    store.run = AsyncMock(return_value={"state": state, "terminal_reason": reason})
    activities = BrokerActivities(store)

    with pytest.raises(ApplicationError, match=reason) as error:
        await activities.finish({"root_id": "run", "tenant_id": "tenant", "state": "SUCCEEDED"})

    assert error.value.non_retryable
    store.finish_run.assert_awaited_once()
