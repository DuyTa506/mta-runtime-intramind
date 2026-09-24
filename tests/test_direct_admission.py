import asyncio
import json
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


async def admit(direct, request, pool_id, owner_id):
    """Keep direct unit grants behind the same live waiter contract as the proxy."""
    await direct.enqueue(request, pool_id, owner_id)
    try:
        reservation = await direct.reserve(request, pool_id, owner_id)
    except BaseException:
        await direct.leave(request.request_id, owner_id)
        raise
    if reservation is None:
        await direct.leave(request.request_id, owner_id)
    return reservation


async def expire(store, attempt_id):
    async with store.engine.begin() as connection:
        await connection.execute(text("""UPDATE runtime_direct_attempts
            SET lease_expires_at=now()-interval '1 second'
            WHERE attempt_id=:id"""), {"id": attempt_id})


async def test_direct_and_background_race_for_endpoint_transport_bound(store):
    await store.configure_pool(pool(target=1, transport_limit=2,
                                    background_transport_limit=2), 1)
    await store.create_root(root())
    await store.submit_operation(operation())
    direct = DirectAdmissions(store)
    results = await asyncio.gather(
        store.reserve_next("p", "background-owner"),
        *(admit(direct, request(f"direct-{i}"), "p", f"direct-owner-{i}") for i in range(8)),
    )
    assert sum(result is not None for result in results) == 2
    assert (await store.drain_status())["compute_held"] == 2
    assert await store.reserve_next("p", "another-background-owner") is None
    assert await admit(direct, request("another-direct"), "p", "another-direct-owner") is None


async def test_independent_llm_endpoints_are_not_serialized_by_gpu_group(store):
    await store.configure_pool(pool(target=2), 1)
    await store.configure_pool(pool("p2", target=2), 1)
    await store.create_root(root())
    await store.submit_operation(operation())
    direct = DirectAdmissions(store)
    held = await admit(direct, request(), "p", "owner")
    assert held is not None
    assert await store.reserve_next("p2", "background-owner") is not None
    assert await admit(direct, request("other"), "p2", "other-owner") is not None
    await direct.finish(held, evidence="not_sent")


async def test_unsent_expiry_releases_once_and_fences_the_old_worker(store):
    await store.configure_pool(pool(target=1), 1)
    direct = DirectAdmissions(store)
    held = await admit(direct, request(), "p", "owner")
    await expire(store, held.attempt_id)
    assert not await direct.heartbeat(held)
    await store.reconcile_expired()
    await store.reconcile_expired()
    assert (await store.drain_status())["compute_held"] == 0
    assert not await direct.heartbeat(held)
    with pytest.raises(RuntimeConflict):
        await direct.mark_send(held)
    assert await admit(direct, request("next"), "p", "next-owner") is not None


async def test_sent_expiry_is_unknown_until_backend_termination_is_confirmed(store):
    await store.configure_pool(pool(target=1, transport_limit=1,
                                    background_transport_limit=1), 1)
    direct = DirectAdmissions(store)
    held = await admit(direct, request(), "p", "owner")
    await direct.mark_send(held)
    await expire(store, held.attempt_id)
    assert not await direct.heartbeat(held)
    await store.reconcile_expired()
    status = await store.drain_status()
    assert status["compute_held"] == status["unknown_attempts"] == status["unsettled_attempts"] == 1
    assert not await direct.heartbeat(held)
    assert await admit(direct, request("next"), "p", "next-owner") is None
    metrics = (await snapshot(store)).decode()
    assert 'intramind_runtime_compute_held{pool="p"} 1.0' in metrics
    assert await store.confirm_epoch_stopped("p", "e1", "fixture process exited") == 1
    assert await store.confirm_epoch_stopped("p", "e1", "fixture process exited") == 0
    assert (await store.drain_status())["unsettled_attempts"] == 0


async def test_identity_owner_and_tenant_are_fenced_without_consuming_more_capacity(store):
    await store.configure_pool(pool(target=2), 2)
    direct = DirectAdmissions(store)
    original = request()
    held = await admit(direct, original, "p", "owner")
    again = await admit(direct, original, "p", "owner")
    assert again.attempt_id == held.attempt_id
    with pytest.raises(RuntimeConflict):
        await admit(direct, original.model_copy(update={"payload_digest": "b" * 64}), "p", "owner")
    with pytest.raises(RuntimeConflict):
        await admit(direct, original, "p", "other-owner")
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
    other_tenant = await admit(direct,
        original.model_copy(update={"tenant_id": "other-tenant"}), "p", "other-owner"
    )
    assert other_tenant.attempt_id != held.attempt_id


async def test_completed_request_is_not_sent_twice_and_double_finish_keeps_other_permit(store):
    await store.configure_pool(pool(target=2, transport_limit=2,
                                    background_transport_limit=2), 2)
    direct = DirectAdmissions(store)
    original = request()
    held = await admit(direct, original, "p", "owner")
    other = await admit(direct, request("other"), "p", "other-owner")
    await direct.mark_send(held)
    await direct.finish(held, evidence="completed_response")
    await direct.finish(held, evidence="completed_response")
    assert await admit(direct, original, "p", "owner") is None
    assert (await store.drain_status())["compute_held"] == 1
    assert await direct.heartbeat(other)
    assert await admit(direct, request("third"), "p", "third-owner") is not None
    assert await admit(direct, request("fourth"), "p", "fourth-owner") is None


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
        await admit(direct, request(**changes), "p", "owner")
    assert (await store.drain_status())["compute_held"] == 0
    assert await admit(direct, request("valid"), "p", "owner") is not None


async def test_expired_request_and_stopped_target_cannot_be_admitted(store):
    await store.configure_pool(pool(target=1), 1)
    direct = DirectAdmissions(store)
    assert await admit(direct,
        request(deadline=datetime.now(UTC) - timedelta(seconds=1)), "p", "owner"
    ) is None
    await store.update_target("p", 0, 1, "operator drain")
    assert await admit(direct, request("fresh"), "p", "owner") is None
    assert (await store.drain_status())["compute_held"] == 0


async def test_embedding_limits_hold_cpu_capacity_without_serializing_llm(store):
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
            await admit(direct, request(kind="embedding", **changes), "embedding", "owner")
    held = await admit(direct,
        request("embedding-valid", kind="embedding", request_bound=40, batch_size=2),
        "embedding", "owner",
    )
    assert held is not None
    assert await store.reserve_next("p", "llm-owner") is not None
    assert (await store.run("r", "t"))["reserved"] > 0


async def test_epoch_change_waits_for_direct_compute_and_fences_late_release(store):
    original_pool = pool(target=1)
    await store.configure_pool(original_pool, 1)
    direct = DirectAdmissions(store)
    held = await admit(direct, request(), "p", "owner")
    await direct.mark_send(held)
    await direct.unknown(held, "response_lost")
    changed_pool = original_pool.model_copy(update={"engine_epoch": "e2"})
    with pytest.raises(RuntimeConflict):
        await store.configure_pool(changed_pool, 1)
    await store.confirm_epoch_stopped("p", "e1", "fixture engine stopped")
    await store.configure_pool(changed_pool, 1)
    fresh = await admit(direct, request("new-epoch"), "p", "next-owner")
    assert fresh.engine_epoch == "e2"
    with pytest.raises(RuntimeConflict):
        await direct.mark_send(held)
    await direct.finish(held, evidence="completed_response")
    assert (await store.drain_status())["compute_held"] == 1
    assert await direct.heartbeat(fresh)


async def test_waiting_qa_claims_next_endpoint_opening_before_background(store):
    await store.configure_pool(pool(target=1, transport_limit=2,
                                    background_transport_limit=2), 1)
    await store.create_root(root(priority="background"))
    await store.submit_operation(operation())
    direct = DirectAdmissions(store)
    qa = request("qa", workload_class="qa")
    await direct.enqueue(qa, "p", "owner")
    assert await store.reserve_next("p", "background-owner") is None
    held = await admit(direct, qa, "p", "owner")
    assert held is not None
    background = await store.reserve_next("p", "background-owner")
    assert background.operation.operation_id == "o"


async def test_qa_waiter_precedes_user_task_without_model_ratio(store):
    await store.configure_pool(pool(target=1, transport_limit=2,
                                    background_transport_limit=2), 1)
    direct = DirectAdmissions(store)
    tool = request("tool", workload_class="user_task")
    qa = request("qa", workload_class="qa")
    await direct.enqueue(tool, "p", "owner")
    await direct.enqueue(qa, "p", "owner")
    assert await admit(direct, tool, "p", "owner") is None
    assert await admit(direct, qa, "p", "owner") is not None
    assert await admit(direct, tool, "p", "owner") is not None


async def test_direct_user_task_and_durable_background_share_lower_class_cap(store):
    await store.configure_pool(pool(transport_limit=3, background_transport_limit=1), 1)
    await store.create_root(root(priority="background"))
    await store.submit_operation(operation())
    direct = DirectAdmissions(store)
    held = await admit(direct, request("tool", workload_class="user_task"), "p", "tool-owner")
    assert held is not None
    assert await admit(direct, request("next", workload_class="background"), "p", "next-owner") is None
    assert await store.reserve_next("p", "background-owner") is None
    qa = await admit(direct, request("question", workload_class="qa"), "p", "qa-owner")
    assert qa is not None, "the lower-class safety cap must leave room for QA"
    await direct.finish(held, evidence="not_sent")
    assert await store.reserve_next("p", "background-owner") is not None


async def test_cross_pool_reservation_rechecks_shared_root_budget(store):
    await store.configure_pool(pool(transport_limit=2, background_transport_limit=2), 1)
    await store.configure_pool(pool("other", transport_limit=2, background_transport_limit=2), 1)
    await store.create_root(root(budget_limit=30))
    await store.submit_operation(operation("one"))
    await store.submit_operation(operation("two"))
    granted = await asyncio.gather(store.reserve_next("p", "one-owner"),
                                   store.reserve_next("other", "other-owner"))
    assert sum(item is not None for item in granted) == 1
    assert (await store.run("r", "t"))["reserved"] == 30


async def test_watchdog_quiesce_token_and_replacement_proof_are_fenced(store):
    await store.configure_pool(pool(), 1)
    await store.quiesce_engine("p", "e1", "owner-a")
    await store.quiesce_engine("p", "e1", "owner-a")
    assert (await store.inspect_engine("p"))["quiesce_owned"] is True
    with pytest.raises(RuntimeConflict):
        await store.quiesce_engine("p", "e1", "owner-b")
    with pytest.raises(RuntimeConflict):
        await store.resume_engine("p", "e1", "owner-b")
    invalid = {"pool_id": "p", "engine_epoch": "e1",
        "verification": "container_stopped_or_replaced", "old_container_id": None,
        "old_started_at": "2026-09-24T01:00:00Z", "observed_container_id": "new",
        "observed_started_at": "2026-09-24T01:01:00Z", "model_pid": None}
    with pytest.raises(ValueError):
        await store.confirm_epoch_stopped("p", "e1", json.dumps(invalid), recover=True)
    await store.resume_engine("p", "e1", "owner-a")
    assert (await store.inspect_engine("p"))["health"] == "HEALTHY"
    await store.quiesce_engine("p", "e1", "owner-a")
    # Docker stop/start retains the container ID but advances StartedAt.
    proof = invalid | {"old_container_id": "new",
        "verified_stopped_at": "2026-09-24T01:02:00Z"}
    assert await store.confirm_epoch_stopped("p", "e1", json.dumps(proof), recover=True) == 0
    await store.configure_pool(pool().model_copy(update={"engine_epoch": "e2"}), 1)
    status = await store.inspect_engine("p")
    assert status["health"] == "HEALTHY" and status["quiesce_owned"] is False
