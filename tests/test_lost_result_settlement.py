"""An attempt whose inference ended but whose result can no longer be committed is settled.

Incident (tool pool, mindmap/v1): the executor finished the LLM call, stalled past its
owner lease before persisting the result, and the reconciler turned the
``BACKEND_FINISHED`` attempt into ``UNKNOWN`` with no compute held. Nothing settled that
state: the operation stayed ``RECONCILING`` and every drain timed out.
"""

import asyncio
import logging

import pytest
from conftest import operation, pool, root
from fakes import IndependentEngine, MemoryArtifacts, TestPermits
from sqlalchemy import text

from intramind_runtime.contracts import RuntimeConflict
from intramind_runtime.executor import Executor

pytestmark = pytest.mark.integration

LOST = "result_lost_after_backend_finished"
CLEAN = {"compute_held": 0, "pending_settlement": 0, "unknown_attempts": 0, "unsettled_attempts": 0}
PAYLOAD = b'{"messages":[{"role":"user","content":"test"}]}'


async def finished_attempt(store, *, target=1, ops=1, **root_kwargs):
    """Reserve, send and finish ``ops`` attempts of one root; return their reservations."""
    await store.create_root(root(**root_kwargs))
    await store.configure_pool(pool(target=target), target)
    reservations = []
    for index in range(ops):
        await store.submit_operation(operation(f"o{index}"))
        reservation = await store.reserve_next("p", f"owner-{index}")
        await store.mark_send(reservation)
        await store.compute_finished(reservation)
        reservations.append(reservation)
    return reservations[0] if ops == 1 else reservations


async def sql(store, statement, **params):
    async with store.engine.begin() as c:
        result = await c.execute(text(statement), params)
        return result.all() if result.returns_rows else []


async def expire_leases(store):
    await sql(store, "UPDATE runtime_attempts SET lease_expires_at=now()-interval '1 second'")


async def attempt(store, attempt_id):
    (found,) = await sql(store, """SELECT state,compute_held,budget_held,usage,usage_estimated,
        error_class FROM runtime_attempts WHERE attempt_id=:id""", id=attempt_id)
    return tuple(found)


def lost_attempt(error_class=LOST):
    return ("FAILED", False, False, 30, True, error_class)


async def test_expired_owner_after_backend_finished_settles_and_requeues(store):
    reservation = await finished_attempt(store)
    await expire_leases(store)
    assert await store.reconcile_expired() == 1
    assert await store.reconcile_expired() == 0
    assert await attempt(store, reservation.attempt_id) == lost_attempt()
    ledger = await store.run("r", "t")
    assert (ledger["reserved"], ledger["spent"]) == (0, 30)
    op = await store.operation("o0", "t")
    assert (op["state"], op["wait_reason"]) == ("RETRY_WAIT", LOST)
    assert await store.drain_status() == CLEAN
    # The stalled executor wakes up late: still fenced, nothing is rewritten.
    with pytest.raises(RuntimeConflict):
        await store.commit_result(reservation, reservation.operation.payload, 12)
    assert await attempt(store, reservation.attempt_id) == lost_attempt()
    assert (await store.run("r", "t"))["spent"] == 30
    assert await sql(store, "SELECT 1 FROM runtime_artifacts") == []
    retry = await store.reserve_next("p", "other")
    assert retry is not None and retry.attempt_id != reservation.attempt_id


async def test_no_attempts_left_fails_the_operation_and_wakes_the_workflow(store):
    reservation = await finished_attempt(store, max_attempts=1)
    await store.bind_completion("o0", "t", b"activity-token")
    await store.claim_events("drain-earlier-events")
    await expire_leases(store)
    assert await store.reconcile_expired() == 1
    assert await attempt(store, reservation.attempt_id) == lost_attempt()
    op = await store.operation("o0", "t")
    assert (op["state"], op["wait_reason"]) == ("FAILED", LOST)
    assert await store.drain_status() == CLEAN
    assert (await store.run("r", "t"))["reserved"] == 0
    events = [e for e in await store.claim_events("publisher") if e["kind"] == "complete_activity"]
    assert len(events) == 1


async def test_cancelled_operation_stays_cancelled_while_the_attempt_settles(store):
    reservation = await finished_attempt(store)
    await store.cancel("r", "t")
    await expire_leases(store)
    assert await store.reconcile_expired() == 1
    assert await attempt(store, reservation.attempt_id) == lost_attempt()
    assert (await store.operation("o0", "t"))["state"] == "CANCELLED"
    assert await store.drain_status() == CLEAN
    ledger = await store.run("r", "t")
    assert (ledger["reserved"], ledger["spent"]) == (0, 30)


@pytest.mark.parametrize("how", ["lease_expired_reconcile", "unknown_after_finish"])
async def test_unknown_without_compute_left_by_earlier_code_is_cleaned(store, how):
    reservation = await finished_attempt(store)
    # The executor recovered from its stall, so its owner lease is live again.
    await store.register_owner("owner-0", "boot")
    if how == "lease_expired_reconcile":
        # Exactly what the earlier reconciler wrote.
        await sql(store, """UPDATE runtime_attempts SET state='UNKNOWN',unknown_at=now(),
            error_class='lease_expired_after_send'""")
        await sql(store, """UPDATE runtime_operations SET state='RECONCILING',
            wait_reason='unknown_compute_or_result'""")
    else:
        await store.unknown(reservation, "executor_shutdown")
    assert (await attempt(store, reservation.attempt_id))[:3] == ("UNKNOWN", False, True)
    assert (await store.drain_status())["unknown_attempts"] == 1
    assert await store.reconcile_expired() == 1
    assert await store.reconcile_expired() == 0
    assert await attempt(store, reservation.attempt_id) == lost_attempt()
    op = await store.operation("o0", "t")
    assert (op["state"], op["wait_reason"]) == ("RETRY_WAIT", LOST)
    assert await store.drain_status() == CLEAN
    assert (await store.run("r", "t"))["reserved"] == 0


@pytest.mark.parametrize("backend_finished", [False, True])
async def test_unknown_that_still_holds_compute_is_not_settled(store, backend_finished):
    await store.create_root(root())
    await store.configure_pool(pool(target=1), 1)
    await store.submit_operation(operation("o0"))
    reservation = await store.reserve_next("p", "owner-0")
    await store.mark_send(reservation)
    await expire_leases(store)
    assert await store.reconcile_expired() == 1
    if backend_finished:
        await sql(store, "UPDATE runtime_attempts SET backend_finished_at=now()")
    for _ in range(2):
        assert await store.reconcile_expired() == 0
    assert (await attempt(store, reservation.attempt_id))[:3] == ("UNKNOWN", True, True)
    assert (await store.operation("o0", "t"))["state"] == "RECONCILING"
    assert (await store.run("r", "t"))["reserved"] == 30
    assert (await store.drain_status())["unknown_attempts"] == 1


@pytest.mark.parametrize("case", ["live_owner_expired_attempt_lease", "dead_owner_fresh_lease"])
async def test_backend_finished_is_left_to_an_owner_that_can_still_commit(store, case):
    reservation = await finished_attempt(store)
    if case == "live_owner_expired_attempt_lease":
        await store.register_owner("owner-0", "boot")
        await expire_leases(store)
    assert await store.reconcile_expired() == 0
    assert (await attempt(store, reservation.attempt_id))[:3] == ("BACKEND_FINISHED", False, True)
    await store.commit_result(reservation, reservation.operation.payload, 12)
    assert (await store.operation("o0", "t"))["state"] == "SUCCEEDED"
    ledger = await store.run("r", "t")
    assert (ledger["reserved"], ledger["spent"]) == (0, 12)


async def test_attempt_locked_by_a_committing_owner_is_retried_on_the_next_tick(store):
    reservation = await finished_attempt(store)
    await expire_leases(store)
    async with store.engine.begin() as c:
        await c.execute(text("SELECT 1 FROM runtime_attempts WHERE attempt_id=:id FOR UPDATE"),
                        {"id": reservation.attempt_id})
        assert await asyncio.wait_for(store.reconcile_expired(), 5) == 0
    assert (await attempt(store, reservation.attempt_id))[0] == "BACKEND_FINISHED"
    assert await store.reconcile_expired() == 1
    assert await attempt(store, reservation.attempt_id) == lost_attempt()


async def test_one_unsettleable_attempt_does_not_starve_the_others(store, monkeypatch, caplog):
    first, second = await finished_attempt(store, target=2, ops=2)
    await expire_leases(store)
    settle = store._settle_stopped_attempt

    async def poisoned(c, a, *args, **kwargs):
        if a["attempt_id"] == first.attempt_id:
            raise RuntimeConflict("attempt budget ledger missing")
        return await settle(c, a, *args, **kwargs)

    monkeypatch.setattr(store, "_settle_stopped_attempt", poisoned)
    with caplog.at_level(logging.ERROR, logger="intramind_runtime.store"):
        assert await store.reconcile_expired() == 1
    assert "lost result settlement failed" in caplog.text
    assert (await attempt(store, first.attempt_id))[0] == "BACKEND_FINISHED"
    assert await attempt(store, second.attempt_id) == lost_attempt()
    monkeypatch.setattr(store, "_settle_stopped_attempt", settle)
    assert await store.reconcile_expired() == 1
    assert await attempt(store, first.attempt_id) == lost_attempt()


async def test_unsettleable_attempts_never_fill_a_fixed_batch(store, monkeypatch):
    attempts = await finished_attempt(store, ops=70, max_pending=100)
    await expire_leases(store)
    poisoned = {a.attempt_id for a in attempts[:66]}
    settle = store._settle_stopped_attempt

    async def selective(c, a, *args, **kwargs):
        if a["attempt_id"] in poisoned:
            raise RuntimeConflict("attempt budget ledger missing")
        return await settle(c, a, *args, **kwargs)

    monkeypatch.setattr(store, "_settle_stopped_attempt", selective)
    assert await store.reconcile_expired() == 4
    for reservation in attempts[66:]:
        assert await attempt(store, reservation.attempt_id) == lost_attempt()
    assert (await attempt(store, attempts[0].attempt_id))[0] == "BACKEND_FINISHED"


async def test_executor_stalled_past_its_lease_is_fenced_and_the_operation_reruns(store):
    await store.create_root(root())
    await store.configure_pool(pool(target=1), 1)
    blobs = MemoryArtifacts()
    spec = operation()
    blobs.data[spec.payload.key] = PAYLOAD
    await store.submit_operation(spec)
    engine = IndependentEngine()
    engine.gate.set()
    executor = Executor(store, blobs, engine, "p", "executor", TestPermits())
    stalled, resume = asyncio.Event(), asyncio.Event()
    put = blobs.put

    async def stall_once(*args, **kwargs):
        if not stalled.is_set():
            stalled.set()
            await resume.wait()
        return await put(*args, **kwargs)

    blobs.put = stall_once
    first = asyncio.create_task(executor.tick())
    try:
        await asyncio.wait_for(stalled.wait(), 5)
        assert (await store.drain_status())["pending_settlement"] == 1
        await expire_leases(store)
        assert await store.reconcile_expired() == 1
        assert (await store.operation("o", "t"))["state"] == "RETRY_WAIT"
        assert await store.drain_status() == CLEAN
    finally:
        resume.set()
    assert await asyncio.wait_for(first, 5) is True
    assert (await store.operation("o", "t"))["state"] == "RETRY_WAIT"
    assert await sql(store, "SELECT 1 FROM runtime_artifacts") == []
    assert await store.drain_status() == CLEAN
    ledger = await store.run("r", "t")
    assert (ledger["reserved"], ledger["spent"]) == (0, 30)
    assert await asyncio.wait_for(executor.tick(), 10) is True
    assert len(engine.calls) == 2
    assert (await store.operation("o", "t"))["state"] == "SUCCEEDED"
    assert (await store.run("r", "t"))["spent"] == 50
    assert await store.drain_status() == CLEAN
