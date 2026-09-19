"""PostgreSQL retry persistence, deadline races and delivery fencing."""

import pytest
from conftest import operation, pool, root
from sqlalchemy import text

pytestmark = pytest.mark.integration


async def expire_root(store, root_id: str = "r") -> None:
    async with store.engine.begin() as connection:
        await connection.execute(
            text("UPDATE runtime_roots SET deadline=now()-interval '1 second' WHERE root_id=:id"),
            {"id": root_id},
        )


async def test_late_success_is_deadline_failure_without_publishing_result(store):
    await store.create_root(root())
    await expire_root(store)

    await store.finish_run("r", "t", "SUCCEEDED", result=operation().payload)

    run = await store.run("r", "t")
    assert run["state"] == "FAILED"
    assert run["terminal_reason"] == "deadline_exceeded"
    assert run["result"] is None
    assert run["finished_at"] is not None


async def test_deadline_completion_cancels_pending_but_preserves_unknown_compute(store):
    await store.create_root(root())
    await store.configure_pool(pool(target=1), 1)
    await store.submit_operation(operation())
    reservation = await store.reserve_next("p", "executor")
    await store.mark_send(reservation)
    await expire_root(store)

    await store.finish_run("r", "t", "SUCCEEDED", result=operation().payload)

    run = await store.run("r", "t")
    assert run["state"] == "FAILED"
    assert run["reserved"] == 30
    assert run["cleanup_pending"]
    assert (await store.operation("o", "t"))["state"] == "CANCELLED"
    assert await store.reserve_next("p", "next") is None


async def test_late_redelivery_does_not_rewrite_previously_committed_success(store):
    await store.create_root(root())
    result = operation().payload
    await store.finish_run("r", "t", "SUCCEEDED", result=result)
    original = await store.run("r", "t")
    await expire_root(store)

    await store.finish_run("r", "t", "SUCCEEDED", result=result)

    replayed = await store.run("r", "t")
    assert replayed["state"] == "SUCCEEDED"
    assert replayed["result"] == original["result"]
    assert replayed["finished_at"] == original["finished_at"]


async def test_outbox_retry_is_durable_and_old_claim_cannot_change_new_claim(store):
    async with store.transaction() as connection:
        await store._event(connection, "test", "unsupported", "run", {})
    first = (await store.claim_events("same-owner"))[0]

    assert await store.defer_event("test", "same-owner", first["deliveries"], 60)
    assert await store.claim_events("competitor") == []
    async with store.engine.begin() as connection:
        await connection.execute(text("UPDATE runtime_outbox SET available_at=now()"))
    second = (await store.claim_events("same-owner"))[0]
    assert second["deliveries"] == first["deliveries"] + 1

    assert not await store.delivered("test", "same-owner", first["deliveries"])
    assert not await store.defer_event("test", "same-owner", first["deliveries"], 60)
    assert await store.delivered("test", "same-owner", second["deliveries"])
    assert not await store.delivered("test", "same-owner", second["deliveries"])
    assert await store.claim_events("competitor") == []
