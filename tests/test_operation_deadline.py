"""A phase deadline ends its operation without erasing uncertain backend work."""

from datetime import UTC, datetime, timedelta

import pytest
from conftest import operation, pool, root
from pydantic import ValidationError

from intramind_runtime.contracts import RuntimeConflict
from intramind_runtime.store import execute, row


def test_operation_deadline_requires_timezone_and_survives_serialization():
    deadline = datetime.now(UTC) + timedelta(minutes=1)
    spec = operation(deadline=deadline)
    assert spec.deadline == deadline
    assert spec.model_validate(spec.model_dump(mode="json")) == spec
    with pytest.raises(ValidationError, match="timezone"):
        operation(deadline=datetime(2026, 9, 19))


async def setup(store, *, deadline=None):
    await store.create_root(root())
    await store.configure_pool(pool(target=1), 1)
    spec = operation(deadline=deadline or datetime.now(UTC) + timedelta(minutes=1))
    await store.submit_operation(spec)
    return spec


async def expire(store):
    async with store.engine.begin() as c:
        await execute(c, """UPDATE runtime_operations SET spec=jsonb_set(
            spec,'{deadline}',to_jsonb(CAST(:deadline AS text))) WHERE operation_id='o'""",
            deadline=(datetime.now(UTC) - timedelta(seconds=1)).isoformat())


async def assert_expired(store):
    state = await store.operation("o", "t")
    assert state["state"] == "FAILED"
    assert state["wait_reason"] == "operation_deadline_exceeded"
    assert state["result"] is None
    assert (await store.run("r", "t"))["state"] == "RUNNING"


@pytest.mark.integration
async def test_expired_submission_is_terminal_and_does_not_reserve_compute(store):
    await setup(store, deadline=datetime.now(UTC) - timedelta(seconds=1))
    await assert_expired(store)
    assert await store.reserve_next("p", "worker") is None
    assert (await store.run("r", "t"))["reserved"] == 0


@pytest.mark.integration
async def test_pending_backpressure_cannot_extend_an_expired_phase(store):
    await store.create_root(root(max_pending=1))
    await store.submit_operation(operation("existing"))
    await store.submit_operation(operation(deadline=datetime.now(UTC) - timedelta(seconds=1)))
    await assert_expired(store)
    assert (await store.operation("existing", "t"))["state"] == "READY"


@pytest.mark.integration
async def test_reserve_rechecks_deadline_without_waiting_for_reconciler(store):
    await setup(store)
    await expire(store)
    assert await store.reserve_next("p", "worker") is None
    await assert_expired(store)
    assert (await store.operation("o", "t"))["attempts"] == 0


@pytest.mark.integration
async def test_attempt_is_bounded_by_the_original_operation_deadline(store):
    spec = await setup(store)
    reservation = await store.reserve_next("p", "worker")
    assert reservation.attempt_deadline == spec.deadline


@pytest.mark.integration
async def test_expired_reserved_attempt_cannot_send_and_refunds_once_with_fresh_lease(store):
    await setup(store)
    reservation = await store.reserve_next("p", "worker")
    await expire(store)
    with pytest.raises(RuntimeConflict):
        await store.mark_send(reservation)
    await store.reconcile_expired()
    await store.reconcile_expired()
    await assert_expired(store)
    assert (await store.run("r", "t"))["reserved"] == 0
    assert not (await store.run("r", "t"))["cleanup_pending"]
    with pytest.raises(RuntimeConflict):
        await store.mark_send(reservation)


@pytest.mark.integration
@pytest.mark.parametrize("lose_lease", [False, True])
async def test_expired_sent_attempt_wakes_workflow_but_keeps_accounting(store, lose_lease):
    await setup(store)
    reservation = await store.reserve_next("p", "worker")
    await store.mark_send(reservation)
    await store.bind_completion("o", "t", b"phase-activity")
    await expire(store)
    await store.reconcile_expired()
    await assert_expired(store)
    if lose_lease:
        async with store.engine.begin() as c:
            await execute(c, "UPDATE runtime_attempts SET lease_expires_at=now()-interval '1 second'")
    else:
        await store.unknown(reservation, "read_timeout")
    await store.reconcile_expired()
    await assert_expired(store)
    state = await store.run("r", "t")
    assert state["reserved"] == 30 and state["cleanup_pending"]
    assert await store.reserve_next("p", "other") is None
    events = await store.claim_events("delivery")
    assert len([event for event in events if event["kind"] == "complete_activity"]) == 1
    await store.confirm_epoch_stopped("p", "e1", "isolated test engine stopped")
    await assert_expired(store)
    assert (await store.run("r", "t"))["spent"] == 30


@pytest.mark.integration
@pytest.mark.parametrize("reconcile_first", [False, True])
async def test_late_result_settles_once_without_overwriting_expired_operation(store, reconcile_first):
    await setup(store)
    reservation = await store.reserve_next("p", "worker")
    await store.mark_send(reservation)
    await expire(store)
    if reconcile_first:
        await store.reconcile_expired()
    await store.compute_finished(reservation)
    await store.commit_result(reservation, reservation.operation.payload, 15)
    await store.commit_result(reservation, reservation.operation.payload, 15)
    await assert_expired(store)
    state = await store.run("r", "t")
    assert state["spent"] == 15 and state["reserved"] == 0
    assert not state["cleanup_pending"]
    async with store.engine.connect() as c:
        assert (await row(c, "SELECT disposition FROM runtime_artifacts"))["disposition"] == "late_terminal"


@pytest.mark.integration
@pytest.mark.parametrize("not_sent", [False, True])
async def test_terminal_failure_cannot_retry_past_operation_deadline(store, not_sent):
    await setup(store)
    reservation = await store.reserve_next("p", "worker")
    if not not_sent:
        await store.mark_send(reservation)
        await store.compute_finished(reservation)
    await expire(store)
    await store.fail(reservation, "transient_error", not_sent=not_sent, retry=True)
    await assert_expired(store)
    assert (await store.run("r", "t"))["reserved"] == 0
    assert (await store.run("r", "t"))["spent"] == (0 if not_sent else 30)


@pytest.mark.integration
async def test_old_operation_without_deadline_still_attaches_with_unchanged_identity(store):
    await store.create_root(root())
    await store.submit_operation(operation())
    async with store.engine.begin() as c:
        await execute(c, "UPDATE runtime_operations SET spec=spec-'deadline'")
    assert await store.submit_operation(operation(deadline=None)) == "o"
    with pytest.raises(RuntimeConflict):
        await store.submit_operation(operation(deadline=datetime.now(UTC) + timedelta(minutes=1)))
