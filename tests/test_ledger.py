import asyncio

import pytest
from conftest import operation, pool, root
from sqlalchemy import text

from intramind_runtime.contracts import AdmissionDenied, RuntimeConflict

pytestmark = pytest.mark.integration


async def test_backend_usage_overrun_stops_root_without_rewriting_cancel(store):
    await setup(store, count=2, budget=30, target=1)
    attempt = await store.reserve_next("p", "owner")
    await store.mark_send(attempt)
    await store.compute_finished(attempt)
    await store.commit_result(attempt, attempt.operation.payload, 40)
    await store.reconcile_expired()
    state = await store.run("r", "t")
    assert state["state"] == "FAILED"
    assert state["terminal_reason"] == "observed_usage_exceeds_budget"
    assert state["operations"].get("READY", 0) == 0


async def test_late_usage_overrun_keeps_user_cancellation_terminal(store):
    await setup(store, count=1, budget=30, target=1)
    attempt = await store.reserve_next("p", "owner")
    await store.mark_send(attempt)
    await store.cancel("r", "t")
    await store.compute_finished(attempt)
    await store.commit_result(attempt, attempt.operation.payload, 40)
    state = await store.run("r", "t")
    assert state["state"] == "CANCELLED"
    assert state["spent"] == 40
    assert not state["cleanup_pending"]


async def test_prepared_request_cannot_use_a_different_tokenizer_profile(store):
    await store.create_root(root())
    await store.configure_pool(pool(), 2)
    await store.submit_operation(operation().model_copy(update={"capacity_profile_id": "different-model"}))
    assert await store.reserve_next("p", "executor") is None
    await store.reconcile_expired()
    assert (await store.operation("o", "t"))["wait_reason"] == "capacity_profile_changed"


async def test_incompatible_prefix_does_not_hide_another_eligible_root(store):
    await store.create_root(root("large", "a"))
    await store.create_root(root("small", "b"))
    await store.configure_pool(pool(), 2)
    for index in range(64):
        await store.submit_operation(operation(f"large-{index}", "large", "a").model_copy(
            update={"input_tokens_bound": 2000}))
    await store.submit_operation(operation("small-op", "small", "b"))
    attempt = await store.reserve_next("p", "executor")
    assert attempt.operation.root_id == "small"


async def setup(store, *, target=2, group_ceiling=2, budget=10000, count=3):
    await store.create_root(root(budget_limit=budget))
    await store.configure_pool(pool(target=target), group_ceiling)
    for i in range(count):
        await store.submit_operation(operation(f"o{i}"))


async def test_last_permit_and_budget_are_atomic(store):
    await setup(store, target=4, group_ceiling=4, budget=30)
    reservations = await asyncio.gather(*(store.reserve_next("p", str(i)) for i in range(12)))
    assert len([r for r in reservations if r]) == 1
    assert (await store.run("r", "t"))["reserved"] == 30


async def test_idempotency_and_conflicting_input(store):
    await setup(store, count=1)
    assert await store.submit_operation(operation("o0")) == "o0"
    with pytest.raises(RuntimeConflict):
        await store.submit_operation(operation("o0").model_copy(update={"max_output_tokens": 21}))
    with pytest.raises(AdmissionDenied):
        await store.create_root(root("expired").model_copy(update={"deadline": __import__("datetime").datetime(2000,1,1,tzinfo=__import__("datetime").UTC)}))


async def test_unknown_retains_budget_and_compute_after_lease(store):
    await setup(store, target=1)
    await store.configure_pool(pool(target=1, transport_limit=1,
                                    background_transport_limit=1), 1)
    r = await store.reserve_next("p", "a")
    await store.mark_send(r)
    async with store.engine.begin() as c:
        await c.execute(text("UPDATE runtime_attempts SET lease_expires_at=now()-interval '1 second'"))
    await store.reconcile_expired()
    assert await store.reserve_next("p", "b") is None
    assert (await store.operation("o0", "t"))["state"] == "RECONCILING"
    assert (await store.run("r", "t"))["reserved"] == 30


async def test_expired_before_send_refunds_once_and_fences_worker(store):
    await setup(store, target=1)
    r = await store.reserve_next("p", "a")
    async with store.engine.begin() as c:
        await c.execute(text("UPDATE runtime_attempts SET lease_expires_at=now()-interval '1 second'"))
    await store.reconcile_expired()
    await store.reconcile_expired()
    assert (await store.run("r", "t"))["reserved"] == 0
    with pytest.raises(RuntimeConflict):
        await store.mark_send(r)
    assert await store.reserve_next("p", "b")


async def test_compute_release_separate_from_idempotent_result_settle(store):
    await setup(store, target=1)
    r = await store.reserve_next("p", "a")
    await store.mark_send(r)
    await store.compute_finished(r)
    assert await store.reserve_next("p", "b")
    assert (await store.run("r", "t"))["reserved"] == 60
    await store.commit_result(r, r.operation.payload, 12)
    await store.commit_result(r, r.operation.payload, 12)
    ledger = await store.run("r", "t")
    assert (ledger["reserved"], ledger["spent"]) == (30, 12)
    assert (await store.operation("o0", "t"))["state"] == "SUCCEEDED"


async def test_cancel_does_not_refund_or_publish_late_result(store):
    await setup(store, target=1)
    r = await store.reserve_next("p", "a")
    await store.mark_send(r)
    await store.cancel("r", "t")
    assert (await store.run("r", "t"))["reserved"] == 30
    assert await store.reserve_next("p", "b") is None
    await store.compute_finished(r)
    await store.commit_result(r, r.operation.payload, None)
    result = await store.operation("o0", "t")
    assert result["state"] == "CANCELLED" and result["result"] is None
    assert (await store.run("r", "t"))["spent"] == 30


async def test_two_llm_pools_keep_independent_transport_capacity(store):
    await setup(store, group_ceiling=1)
    await store.configure_pool(pool("p2"), 1)
    results = await asyncio.gather(store.reserve_next("p", "a"), store.reserve_next("p2", "b"))
    assert len([r for r in results if r]) == 2


async def test_target_drop_drains_and_rejects_stale_envelope(store):
    await setup(store)
    await store.reserve_next("p", "a")
    await store.reserve_next("p", "b")
    await store.update_target("p", 0, 1, "test_congestion")
    assert await store.reserve_next("p", "c") is None
    with pytest.raises(RuntimeConflict):
        await store.update_target("p", 2, 1, "stale")


async def test_bounded_materialization(store):
    await store.create_root(root(max_pending=1))
    await store.submit_operation(operation("first"))
    with pytest.raises(AdmissionDenied):
        await store.submit_operation(operation("second"))


async def test_retry_attaches_and_no_new_inference_after_commit(store):
    await setup(store, count=1)
    r = await store.reserve_next("p", "a")
    await store.fail(r, "connect_failure", not_sent=True, retry=True)
    second = await store.reserve_next("p", "b")
    assert second.operation.operation_id == r.operation.operation_id
    assert second.attempt_id != r.attempt_id
    await store.mark_send(second)
    await store.compute_finished(second)
    await store.commit_result(second, second.operation.payload, 11)
    await store.submit_operation(operation("o0"))
    assert await store.reserve_next("p", "c") is None


async def test_completion_loss_and_rebinding_are_durable(store):
    await setup(store, count=1)
    await store.bind_completion("o0", "t", b"old-token")
    r = await store.reserve_next("p", "a")
    await store.mark_send(r)
    await store.compute_finished(r)
    await store.commit_result(r, r.operation.payload, 10)
    await store.bind_completion("o0", "t", b"fresh-token")
    events = await store.claim_events("publisher")
    assert len(events) == 2
    assert all(e["kind"] == "complete_activity" for e in events)
    assert await store.claim_events("competitor") == []


async def test_root_fairness_under_fanout(store):
    await setup(store, target=4, group_ceiling=4)
    await store.create_root(root("small", "other"))
    await store.submit_operation(operation("small-op", "small", "other"))
    results = [await store.reserve_next("p", str(i)) for i in range(2)]
    assert {r.operation.root_id for r in results} == {"r", "small"}


async def test_confirmed_epoch_stop_settles_uncertainty_once(store):
    await setup(store, target=1)
    reservation = await store.reserve_next("p", "a")
    await store.mark_send(reservation)
    await store.unknown(reservation, "read_timeout")
    with pytest.raises(ValueError):
        await store.confirm_epoch_stopped("p", "e1", "")
    assert await store.confirm_epoch_stopped("p", "e1", "isolated backend process joined") == 1
    assert await store.confirm_epoch_stopped("p", "e1", "same evidence") == 0
    ledger = await store.run("r", "t")
    assert ledger["reserved"] == 0 and ledger["spent"] == 30
    assert await store.reserve_next("p", "b") is None


async def test_impossible_context_is_terminal(store):
    await setup(store, count=0)
    await store.submit_operation(operation().model_copy(update={"input_tokens_bound": 2000}))
    assert await store.reserve_next("p", "a") is None
    await store.reconcile_expired()
    op = await store.operation("o", "t")
    assert op["state"] == "FAILED" and op["wait_reason"] == "context_exceeds_all_pools"


async def test_budget_exhaustion_is_not_an_infinite_queue(store):
    await setup(store, budget=10, count=1)
    assert await store.reserve_next("p", "a") is None
    assert (await store.operation("o0", "t"))["wait_reason"] == "root_budget_exhausted"


async def test_identical_artifact_bytes_have_independent_operation_commits(store):
    await setup(store, count=2)
    first = await store.reserve_next("p", "a")
    second = await store.reserve_next("p", "b")
    for reservation in (first, second):
        await store.mark_send(reservation)
        await store.compute_finished(reservation)
        await store.commit_result(reservation, reservation.operation.payload, 12)
    assert (await store.run("r", "t"))["spent"] == 24
    assert (await store.operation("o1", "t"))["state"] == "SUCCEEDED"
