"""rc18 direct permits live in RAM; durable ledger still owns task recovery."""

import asyncio
from datetime import UTC, datetime, timedelta
from time import monotonic

import pytest
from conftest import operation, pool, root
from sqlalchemy import event, text

from intramind_runtime.contracts import (
    AdmissionDenied,
    EmbeddingPoolSpec,
    RerankPoolSpec,
    RuntimeConflict,
)
from intramind_runtime.direct import DirectRequest
from intramind_runtime.memory_scheduler import MemoryScheduler
from intramind_runtime.metrics import snapshot

pytestmark = pytest.mark.integration


def request(request_id="direct", **changes):
    values = dict(request_id=request_id, tenant_id="t", payload_digest="a" * 64,
                  model_profile="test", capacity_profile_id="test-v1",
                  request_bound=30, deadline=datetime.now(UTC)+timedelta(minutes=5))
    return DirectRequest(**(values | changes))


async def admit(scheduler, item, pool_id="p", owner_id="owner"):
    await scheduler.enqueue(item, pool_id, owner_id)
    return await scheduler.reserve(item, pool_id, owner_id)


async def durable(scheduler, reservation):
    return await scheduler.durable(reservation.attempt_id, reservation.pool_id,
        reservation.owner_id, reservation.engine_epoch, reservation.workload_class,
        reservation.attempt_deadline)


async def test_direct_request_has_zero_sql_statements_after_start(store):
    await store.configure_pool(pool(transport_limit=4, background_transport_limit=2), 4)
    scheduler = MemoryScheduler(store)
    await scheduler.start()
    scheduler._task.cancel()  # Periodic control refresh is unrelated to request SQL.
    await asyncio.gather(scheduler._task, return_exceptions=True)
    statements = []

    def count_sql(connection, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(store.engine.sync_engine, "before_cursor_execute", count_sql)
    try:
        for index in range(100):
            held = await admit(scheduler, request(f"zero-sql-{index}"))
            assert held is not None
            await scheduler.mark_send(held)
            await scheduler.finish(held)
    finally:
        event.remove(store.engine.sync_engine, "before_cursor_execute", count_sql)
        await scheduler.close()
    assert statements == []


async def test_qa_waiter_precedes_user_task_and_durable_background(store):
    await store.configure_pool(pool(target=1, transport_limit=1,
                                    background_transport_limit=1), 1)
    await store.create_root(root(priority="background"))
    await store.submit_operation(operation())
    scheduler = MemoryScheduler(store)
    await scheduler.start()
    try:
        task = request("tool", workload_class="user_task")
        qa = request("qa", workload_class="qa")
        await scheduler.enqueue(task, "p", "direct")
        await scheduler.enqueue(qa, "p", "direct")
        reservation = await store.reserve_next("p", "executor")
        assert reservation.workload_class == "background"
        background = asyncio.create_task(durable(scheduler, reservation))
        await asyncio.sleep(0)
        assert not background.done()
        assert await scheduler.reserve(task, "p", "direct") is None
        qa_held = await scheduler.reserve(qa, "p", "direct")
        assert qa_held is not None
        assert not background.done()
        await scheduler.finish(qa_held, evidence="not_sent")
        tool_held = await scheduler.reserve(task, "p", "direct")
        assert tool_held is not None
        assert not background.done()
        await scheduler.finish(tool_held, evidence="not_sent")
        permit = await asyncio.wait_for(background, 2)
        assert permit.attempt_id == reservation.attempt_id
        assert len(scheduler.permits) == 1
        with pytest.raises(RuntimeConflict, match="termination"):
            await scheduler.durable_release(reservation.attempt_id, reservation.owner_id)
        await store.fail(reservation, "not_sent", not_sent=True, retry=False)
        await scheduler.durable_release(reservation.attempt_id, reservation.owner_id)
        assert not scheduler.permits
    finally:
        await scheduler.close()


async def test_lower_class_cap_reserves_qa_headroom(store):
    await store.configure_pool(pool(transport_limit=3, background_transport_limit=1), 1)
    scheduler = MemoryScheduler(store)
    await scheduler.start()
    try:
        tool = await admit(scheduler, request("tool", workload_class="user_task"))
        assert tool is not None
        queued = request("background", workload_class="background")
        await scheduler.enqueue(queued, "p", "owner")
        assert await scheduler.reserve(queued, "p", "owner") is None
        question = await admit(scheduler, request("question", workload_class="qa"))
        assert question is not None
        assert len(scheduler.permits) == 2
        await scheduler.finish(tool, evidence="not_sent")
        background = await scheduler.reserve(queued, "p", "owner")
        assert background is not None, "released lower-class slot can be reused"
        await scheduler.finish(question, evidence="not_sent")
        await scheduler.finish(background, evidence="not_sent")
    finally:
        await scheduler.close()


async def test_non_llm_group_ceiling_does_not_serialize_llm_pools(store):
    embedding = EmbeddingPoolSpec(**(pool("embedding", target=1).model_dump(
        exclude={"context_limit"}) | {"character_limit": 40, "max_batch_size": 2}))
    rerank = RerankPoolSpec(**(pool("rerank", target=1).model_dump(
        exclude={"context_limit"}) | {"character_limit": 40, "max_batch_size": 2}))
    await store.configure_pool(pool(target=1), 1)
    await store.configure_pool(embedding, 1)
    await store.configure_pool(rerank, 1)
    scheduler = MemoryScheduler(store)
    await scheduler.start()
    try:
        held = await admit(scheduler, request("embedding", kind="embedding",
            request_bound=40, batch_size=2), "embedding")
        assert held is not None
        other = request("rerank", kind="rerank", request_bound=40, batch_size=2)
        await scheduler.enqueue(other, "rerank", "owner")
        assert await scheduler.reserve(other, "rerank", "owner") is None
        assert await admit(scheduler, request("llm")) is not None
        await scheduler.finish(held, evidence="not_sent")
        assert await scheduler.reserve(other, "rerank", "owner") is not None
    finally:
        await scheduler.close()


async def test_identity_ttl_and_invalid_request_do_not_consume_capacity(store):
    await store.configure_pool(pool(target=1, transport_limit=1,
                                    background_transport_limit=1), 1)
    scheduler = MemoryScheduler(store, request_ttl_seconds=1)
    await scheduler.start()
    try:
        original = request()
        await scheduler.enqueue(original, "p", "owner")
        assert await scheduler.enqueue(original, "p", "owner")
        with pytest.raises(RuntimeConflict):
            await scheduler.enqueue(original.model_copy(update={"payload_digest": "b"*64}),
                                    "p", "owner")
        with pytest.raises(RuntimeConflict):
            await scheduler.enqueue(original, "p", "other-owner")
        held = await scheduler.reserve(original, "p", "owner")
        await scheduler.mark_send(held)
        await scheduler.finish(held)
        with pytest.raises(RuntimeConflict, match="already accepted"):
            await scheduler.enqueue(original, "p", "owner")
        for changes in ({"capacity_profile_id": "old"}, {"model_profile": "foreign"},
                        {"request_bound": 1025}, {"kind": "embedding"}):
            with pytest.raises(AdmissionDenied):
                await scheduler.enqueue(request("invalid-"+str(changes), **changes), "p", "owner")
        assert not scheduler.permits
    finally:
        await scheduler.close()


async def test_unknown_direct_keeps_slot_until_epoch_is_confirmed_stopped(store):
    spec = pool(target=1, transport_limit=1, background_transport_limit=1)
    await store.configure_pool(spec, 1)
    scheduler = MemoryScheduler(store)
    await scheduler.start()
    try:
        original = request(expected_engine_epoch="e1")
        held = await admit(scheduler, original)
        await scheduler.mark_send(held)
        await scheduler.unknown(held, "transport_lost")
        next_request = request("next")
        await scheduler.enqueue(next_request, "p", "owner")
        assert await scheduler.reserve(next_request, "p", "owner") is None
        await store.quiesce_engine("p", "e1", "operator")
        await store.confirm_epoch_stopped("p", "e1", "fixture process stopped")
        await store.configure_pool(spec.model_copy(update={"engine_epoch": "e2"}), 1)
        await scheduler._refresh()
        assert scheduler.direct_attempts[held.attempt_id].state == "FAILED_RECOVERABLE"
        assert not scheduler.permits
        await scheduler.leave(next_request.request_id, "owner")
        await scheduler.enqueue(original, "p", "owner")
        replacement = await scheduler.retry_confirmed(original, "p", "owner", held.attempt_id)
        assert replacement.engine_epoch == "e2" and replacement.generation == 1
        await scheduler.finish(replacement, evidence="not_sent")
    finally:
        await scheduler.close()


async def test_restart_hydrates_durable_held_and_unknown_is_not_reclaimed(store):
    spec = pool(target=1, transport_limit=1, background_transport_limit=1)
    await store.configure_pool(spec, 1)
    await store.create_root(root())
    await store.submit_operation(operation())
    attempt = await store.reserve_next("p", "worker")
    await store.mark_send(attempt)
    await store.unknown(attempt, "lost transport")
    scheduler = MemoryScheduler(store)
    await scheduler.start()
    try:
        assert attempt.attempt_id in scheduler.permits
        item = request("blocked")
        await scheduler.enqueue(item, "p", "owner")
        assert await scheduler.reserve(item, "p", "owner") is None
        await scheduler._reconcile_durable()
        assert attempt.attempt_id in scheduler.permits
        with pytest.raises(RuntimeConflict, match="termination"):
            await scheduler.durable_release(attempt.attempt_id, attempt.owner_id)
        await store.confirm_epoch_stopped("p", "e1", "fixture process stopped")
        await scheduler._reconcile_durable()
        assert attempt.attempt_id not in scheduler.permits
    finally:
        await scheduler.close()


async def test_second_live_api_owner_is_fenced(store):
    await store.configure_pool(pool(), 1)
    first = MemoryScheduler(store)
    second = MemoryScheduler(store)
    await first.start()
    try:
        with pytest.raises(RuntimeConflict, match="another runtime-api"):
            await second.start()
    finally:
        await first.close()
    await second.start()
    await second.close()


async def test_dead_waiter_is_evicted_before_next_grant(store):
    await store.configure_pool(pool(target=1, transport_limit=1,
                                    background_transport_limit=1), 1)
    scheduler = MemoryScheduler(store)
    await scheduler.start()
    try:
        expired = request("expired", deadline=datetime.now(UTC)+timedelta(milliseconds=20))
        await scheduler.enqueue(expired, "p", "owner")
        await asyncio.sleep(0.03)
        fresh = request("fresh")
        assert await admit(scheduler, fresh) is not None
        assert "expired" not in scheduler.waiters
    finally:
        await scheduler.close()


async def test_dirty_native_pool_reopens_at_old_lease_plus_restart_drain(store, caplog):
    embedding = EmbeddingPoolSpec(**(pool("embedding", target=1).model_dump(
        exclude={"context_limit"}) | {"character_limit": 40, "max_batch_size": 2}))
    await store.configure_pool(embedding, 1)
    first = MemoryScheduler(store, attempt_timeouts={"embedding": 1800},
                            restart_drain_seconds={"embedding": 0.8})
    await first.start()
    in_flight = await admit(first, request("old-embed", kind="embedding",
        request_bound=40, batch_size=2), "embedding")
    await first.mark_send(in_flight)
    await first.close()
    async with store.engine.begin() as c:
        await c.execute(text("""UPDATE runtime_owners SET
            lease_expires_at=now()-interval '0.2 second'
            WHERE owner_id='runtime-api-pool:embedding'"""))
    restarted = MemoryScheduler(store, attempt_timeouts={"embedding": 1800},
                                restart_drain_seconds={"embedding": 0.8})
    await restarted.start()
    try:
        remaining = restarted._ready_by_pool["embedding"] - monotonic()
        assert 0.4 < remaining < 0.8
        fresh = request("new-embed", kind="embedding", request_bound=40, batch_size=2)
        await restarted.enqueue(fresh, "embedding", "owner")
        assert await restarted.reserve(fresh, "embedding", "owner") is None
        assert restarted.dirty_blocked()["embedding"] == 1
        metrics = (await snapshot(store, restarted)).decode()
        assert 'intramind_runtime_dirty_recovery_blocked{pool="embedding"} 1.0' in metrics
        await asyncio.sleep(max(0, remaining-0.1))
        assert await restarted.reserve(fresh, "embedding", "owner") is None
        await asyncio.sleep(0.12)
        assert restarted.dirty_blocked()["embedding"] == 0
        assert await restarted.reserve(fresh, "embedding", "owner") is not None
        assert "blocks capacity after unclean runtime-api exit" in caplog.text
    finally:
        await restarted.close()


async def test_idle_native_pool_crash_uses_configured_drain_not_durable_timeout(store):
    embedding = EmbeddingPoolSpec(**(pool("embedding", target=1).model_dump(
        exclude={"context_limit"}) | {"character_limit": 40, "max_batch_size": 2}))
    await store.configure_pool(embedding, 1)
    first = MemoryScheduler(store, attempt_timeouts={"embedding": 1800})
    await first.start()
    # An unclean exit leaves the owner row, even though no request was active.
    first._task.cancel()
    await asyncio.gather(first._task, return_exceptions=True)
    async with store.engine.begin() as c:
        await c.execute(text("""UPDATE runtime_owners SET
            lease_expires_at=now()-interval '0.2 second'
            WHERE owner_id='runtime-api-pool:embedding'"""))
    restarted = MemoryScheduler(store, attempt_timeouts={"embedding": 1800},
                                restart_drain_seconds={"embedding": 0.4})
    await restarted.start()
    try:
        fresh = request("after-idle-crash", kind="embedding", request_bound=40,
                        batch_size=2)
        await restarted.enqueue(fresh, "embedding", "owner")
        assert await restarted.reserve(fresh, "embedding", "owner") is None
        assert restarted.dirty_blocked()["embedding"] == 1
        await asyncio.sleep(0.24)
        assert restarted.dirty_blocked()["embedding"] == 0
        assert await restarted.reserve(fresh, "embedding", "owner") is not None
    finally:
        await restarted.close()
        await first.close()


def test_restart_drain_rejects_invalid_values():
    for value in (0, -1, float("inf"), float("nan"), True, "30"):
        with pytest.raises(ValueError, match="restart_drain_seconds"):
            MemoryScheduler(None, restart_drain_seconds={"embedding": value})


async def test_restart_requeues_reserved_durable_without_oversubscribing(store):
    await store.configure_pool(pool(target=1, transport_limit=1,
                                    background_transport_limit=1), 1)
    reservations = []
    for index in range(5):
        root_id = f"root-{index}"
        await store.create_root(root(root_id, priority="background"))
        await store.submit_operation(operation(f"op-{index}", root_id))
        reservations.append(await store.reserve_next("p", f"worker-{index}"))
    active, pending = reservations[0], reservations[1:]
    await store.mark_send(active)
    scheduler = MemoryScheduler(store)
    await scheduler.start()
    tasks = [asyncio.create_task(durable(scheduler, item)) for item in pending]
    try:
        assert set(scheduler.permits) == {active.attempt_id}
        await asyncio.sleep(0.05)
        assert all(not task.done() for task in tasks)
        assert len(scheduler.permits) == 1
        await store.compute_finished(active)
        await scheduler._reconcile_durable()
        for _ in pending:
            done, _ = await asyncio.wait(tasks, timeout=2,
                                         return_when=asyncio.FIRST_COMPLETED)
            assert len(done) == 1
            task = done.pop()
            tasks.remove(task)
            permit = task.result()
            assert len(scheduler.permits) == 1
            reservation = next(item for item in pending if item.attempt_id == permit.attempt_id)
            await store.fail(reservation, "not_sent", not_sent=True, retry=False)
            await scheduler.durable_release(permit.attempt_id, permit.owner_id)
        assert not scheduler.permits
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await scheduler.close()


async def test_epoch_mismatch_removes_durable_head_waiter_immediately(store):
    await store.configure_pool(pool(target=1, transport_limit=1,
                                    background_transport_limit=1), 1)
    await store.create_root(root(priority="background"))
    await store.submit_operation(operation())
    scheduler = MemoryScheduler(store)
    await scheduler.start()
    old_direct = await admit(scheduler, request("occupy"))
    reservation = await store.reserve_next("p", "worker")
    waiting = asyncio.create_task(durable(scheduler, reservation))
    try:
        async with asyncio.timeout(2):
            while reservation.attempt_id not in scheduler.waiters:
                await asyncio.sleep(0.01)
        scheduler.pools["p"].epoch = "e2"
        scheduler._notify()
        with pytest.raises(RuntimeConflict, match="engine epoch changed"):
            await asyncio.wait_for(waiting, 2)
        assert reservation.attempt_id not in scheduler.waiters
        await scheduler.finish(old_direct, evidence="not_sent")
        fresh = await admit(scheduler, request("fresh-after-epoch"))
        assert fresh is not None and fresh.engine_epoch == "e2"
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await scheduler.close()


async def test_cancelled_duplicate_acquire_keeps_other_waiter_live(store):
    await store.configure_pool(pool(target=1, transport_limit=1,
                                    background_transport_limit=1), 1)
    await store.create_root(root())
    await store.submit_operation(operation())
    scheduler = MemoryScheduler(store)
    await scheduler.start()
    reservation = await store.reserve_next("p", "worker")
    scheduler.pools["p"].target = 0
    first = asyncio.create_task(durable(scheduler, reservation))
    second = asyncio.create_task(durable(scheduler, reservation))
    try:
        async with asyncio.timeout(2):
            while scheduler._durable_waiter_refs[reservation.attempt_id] != 2:
                await asyncio.sleep(0.01)
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        assert reservation.attempt_id in scheduler.waiters
        scheduler.pools["p"].target = 1
        scheduler._notify()
        assert (await asyncio.wait_for(second, 2)).attempt_id == reservation.attempt_id
        assert reservation.attempt_id not in scheduler.waiters
        assert reservation.attempt_id not in scheduler._durable_waiter_refs
    finally:
        for task in (first, second):
            task.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
        await scheduler.close()


async def test_cancelled_durable_operation_cannot_block_queue_head(store):
    await store.configure_pool(pool(target=1, transport_limit=1,
                                    background_transport_limit=1), 1)
    await store.create_root(root())
    await store.submit_operation(operation())
    scheduler = MemoryScheduler(store)
    await scheduler.start()
    reservation = await store.reserve_next("p", "worker")
    scheduler.pools["p"].target = 0
    waiting = asyncio.create_task(durable(scheduler, reservation))
    try:
        async with asyncio.timeout(2):
            while reservation.attempt_id not in scheduler.waiters:
                await asyncio.sleep(0.01)
        await store.cancel("r", "t")
        with pytest.raises(RuntimeConflict, match="no longer dispatchable"):
            await asyncio.wait_for(waiting, 2)
        assert reservation.attempt_id not in scheduler.waiters
        scheduler.pools["p"].target = 1
        assert await admit(scheduler, request("qa-after-cancel")) is not None
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await scheduler.close()
