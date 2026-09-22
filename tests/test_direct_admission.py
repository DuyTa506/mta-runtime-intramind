import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from conftest import operation, pool, root
from sqlalchemy import text

from intramind_runtime.contracts import AdmissionDenied, EmbeddingPoolSpec, RuntimeConflict
from intramind_runtime.direct import DirectAdmissions, DirectRequest
from intramind_runtime.metrics import snapshot

pytestmark = pytest.mark.integration


def request(request_id="direct", **changes):
    values = dict(
        request_id=request_id,
        tenant_id="t",
        payload_digest="a" * 64,
        model_profile="test",
        capacity_profile_id="test-v1",
        request_bound=30,
        deadline=datetime.now(UTC) + timedelta(minutes=5),
    )
    return DirectRequest(**(values | changes))


async def expire(store, attempt_id):
    async with store.engine.begin() as connection:
        await connection.execute(text("""UPDATE runtime_direct_attempts
            SET lease_expires_at=now()-interval '1 second'
            WHERE attempt_id=:id"""), {"id": attempt_id})


async def test_direct_and_background_race_for_the_same_last_permit(store):
    await store.configure_pool(pool(target=1), 1)
    await store.create_root(root())
    await store.submit_operation(operation())
    direct = DirectAdmissions(store)
    results = await asyncio.gather(
        store.reserve_next("p", "background-owner"),
        *(direct.reserve(request(f"direct-{i}"), "p", f"direct-owner-{i}") for i in range(8)),
    )
    assert sum(result is not None for result in results) == 1
    assert (await store.drain_status())["compute_held"] == 1
    assert await store.reserve_next("p", "another-background-owner") is None
    assert await direct.reserve(request("another-direct"), "p", "another-direct-owner") is None


async def test_direct_compute_blocks_another_pool_on_the_same_gpu(store):
    await store.configure_pool(pool(target=2), 1)
    await store.configure_pool(pool("p2", target=2), 1)
    await store.create_root(root())
    await store.submit_operation(operation())
    direct = DirectAdmissions(store)
    held = await direct.reserve(request(), "p", "owner")
    assert held is not None
    assert await store.reserve_next("p2", "background-owner") is None
    assert await direct.reserve(request("other"), "p2", "other-owner") is None
    await direct.finish(held, evidence="not_sent")
    assert await store.reserve_next("p2", "background-owner") is not None


async def test_unsent_expiry_releases_once_and_fences_the_old_worker(store):
    await store.configure_pool(pool(target=1), 1)
    direct = DirectAdmissions(store)
    held = await direct.reserve(request(), "p", "owner")
    await expire(store, held.attempt_id)
    assert not await direct.heartbeat(held)
    await store.reconcile_expired()
    await store.reconcile_expired()
    assert (await store.drain_status())["compute_held"] == 0
    assert not await direct.heartbeat(held)
    with pytest.raises(RuntimeConflict):
        await direct.mark_send(held)
    assert await direct.reserve(request("next"), "p", "next-owner") is not None


async def test_sent_expiry_is_unknown_until_backend_termination_is_confirmed(store):
    await store.configure_pool(pool(target=1), 1)
    direct = DirectAdmissions(store)
    held = await direct.reserve(request(), "p", "owner")
    await direct.mark_send(held)
    await expire(store, held.attempt_id)
    assert not await direct.heartbeat(held)
    await store.reconcile_expired()
    status = await store.drain_status()
    assert status["compute_held"] == status["unknown_attempts"] == status["unsettled_attempts"] == 1
    assert not await direct.heartbeat(held)
    assert await direct.reserve(request("next"), "p", "next-owner") is None
    metrics = (await snapshot(store)).decode()
    assert 'intramind_runtime_compute_held{pool="p"} 1.0' in metrics
    assert await store.confirm_epoch_stopped("p", "e1", "fixture process exited") == 1
    assert await store.confirm_epoch_stopped("p", "e1", "fixture process exited") == 0
    assert (await store.drain_status())["unsettled_attempts"] == 0


async def test_identity_owner_and_tenant_are_fenced_without_consuming_more_capacity(store):
    await store.configure_pool(pool(target=2), 2)
    direct = DirectAdmissions(store)
    original = request()
    held = await direct.reserve(original, "p", "owner")
    again = await direct.reserve(original, "p", "owner")
    assert again.attempt_id == held.attempt_id
    with pytest.raises(RuntimeConflict):
        await direct.reserve(original.model_copy(update={"payload_digest": "b" * 64}), "p", "owner")
    with pytest.raises(RuntimeConflict):
        await direct.reserve(original, "p", "other-owner")
    impostor = held.model_copy(update={"owner_id": "other-owner"})
    with pytest.raises(RuntimeConflict):
        await direct.heartbeat(impostor)
    with pytest.raises(RuntimeConflict):
        await direct.mark_send(impostor)
    with pytest.raises(RuntimeConflict):
        await direct.finish(impostor, evidence="not_sent")
    foreign = held.model_copy(update={
        "request": original.model_copy(update={"tenant_id": "other-tenant"})
    })
    with pytest.raises(RuntimeConflict):
        await direct.mark_send(foreign)
    with pytest.raises(RuntimeConflict):
        await direct.finish(foreign, evidence="not_sent")
    assert await direct.heartbeat(held)
    assert (await store.drain_status())["compute_held"] == 1
    other_tenant = await direct.reserve(
        original.model_copy(update={"tenant_id": "other-tenant"}), "p", "other-owner"
    )
    assert other_tenant.attempt_id != held.attempt_id


async def test_completed_request_is_not_sent_twice_and_double_finish_keeps_other_permit(store):
    await store.configure_pool(pool(target=2), 2)
    direct = DirectAdmissions(store)
    original = request()
    held = await direct.reserve(original, "p", "owner")
    other = await direct.reserve(request("other"), "p", "other-owner")
    await direct.mark_send(held)
    await direct.finish(held, evidence="completed_response")
    await direct.finish(held, evidence="completed_response")
    assert await direct.reserve(original, "p", "owner") is None
    assert (await store.drain_status())["compute_held"] == 1
    assert await direct.heartbeat(other)
    assert await direct.reserve(request("third"), "p", "third-owner") is not None
    assert await direct.reserve(request("fourth"), "p", "fourth-owner") is None


@pytest.mark.parametrize("changes", [
    {"capacity_profile_id": "old-profile"},
    {"model_profile": "other-model"},
    {"request_bound": 1025},
    {"kind": "embedding"},
])
async def test_invalid_request_fit_does_not_hold_capacity(store, changes):
    await store.configure_pool(pool(target=1), 1)
    direct = DirectAdmissions(store)
    with pytest.raises(AdmissionDenied):
        await direct.reserve(request(**changes), "p", "owner")
    assert (await store.drain_status())["compute_held"] == 0
    assert await direct.reserve(request("valid"), "p", "owner") is not None


async def test_expired_request_and_stopped_target_cannot_be_admitted(store):
    await store.configure_pool(pool(target=1), 1)
    direct = DirectAdmissions(store)
    assert await direct.reserve(
        request(deadline=datetime.now(UTC) - timedelta(seconds=1)), "p", "owner"
    ) is None
    await store.update_target("p", 0, 1, "operator drain")
    assert await direct.reserve(request("fresh"), "p", "owner") is None
    assert (await store.drain_status())["compute_held"] == 0


async def test_embedding_batch_and_character_limits_are_checked_before_shared_reservation(store):
    await store.configure_pool(pool(target=1), 1)
    embedding_pool = EmbeddingPoolSpec(**(pool("embedding", target=1).model_dump(
        exclude={"context_limit"}
    ) | {"character_limit": 40, "max_batch_size": 2}))
    await store.configure_pool(embedding_pool, 1)
    await store.create_root(root())
    await store.submit_operation(operation())
    direct = DirectAdmissions(store)
    for changes in ({"request_bound": 41}, {"batch_size": 3}):
        with pytest.raises(AdmissionDenied):
            await direct.reserve(request(kind="embedding", **changes), "embedding", "owner")
    held = await direct.reserve(
        request("embedding-valid", kind="embedding", request_bound=40, batch_size=2),
        "embedding", "owner",
    )
    assert held is not None
    assert await store.reserve_next("p", "llm-owner") is None
    assert (await store.run("r", "t"))["reserved"] == 0


async def test_epoch_change_waits_for_direct_compute_and_fences_late_release(store):
    original_pool = pool(target=1)
    await store.configure_pool(original_pool, 1)
    direct = DirectAdmissions(store)
    held = await direct.reserve(request(), "p", "owner")
    await direct.mark_send(held)
    await direct.unknown(held, "response_lost")
    changed_pool = original_pool.model_copy(update={"engine_epoch": "e2"})
    with pytest.raises(RuntimeConflict):
        await store.configure_pool(changed_pool, 1)
    await store.confirm_epoch_stopped("p", "e1", "fixture engine stopped")
    await store.configure_pool(changed_pool, 1)
    fresh = await direct.reserve(request("new-epoch"), "p", "next-owner")
    assert fresh.engine_epoch == "e2"
    with pytest.raises(RuntimeConflict):
        await direct.mark_send(held)
    await direct.finish(held, evidence="completed_response")
    assert (await store.drain_status())["compute_held"] == 1
    assert await direct.heartbeat(fresh)


async def test_every_fifth_shared_grant_prefers_eligible_background(store):
    await store.configure_pool(pool(target=1), 1)
    await store.create_root(root(priority="background"))
    await store.submit_operation(operation())
    direct = DirectAdmissions(store)
    for index in range(4):
        held = await direct.reserve(request(f"interactive-{index}"), "p", "owner")
        assert held is not None
        await direct.mark_send(held)
        await direct.finish(held, evidence="completed_response")
    assert await direct.reserve(request("fifth"), "p", "owner") is None
    background = await store.reserve_next("p", "background-owner")
    assert background.operation.operation_id == "o"
    await store.fail(background, "fixture_connect_failure", not_sent=True, retry=False)
    assert await direct.reserve(request("sixth"), "p", "owner") is not None


async def test_ineligible_background_does_not_permanently_block_the_fifth_direct_grant(store):
    await store.configure_pool(pool(target=1), 1)
    await store.create_root(root(priority="background", budget_limit=1))
    await store.submit_operation(operation())
    direct = DirectAdmissions(store)
    for index in range(5):
        held = await direct.reserve(request(f"interactive-{index}"), "p", "owner")
        assert held is not None
        await direct.mark_send(held)
        await direct.finish(held, evidence="completed_response")
