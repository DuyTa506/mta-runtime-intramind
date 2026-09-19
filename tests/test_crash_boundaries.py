"""Repeat the two dispatch uncertainty boundaries against PostgreSQL."""

import pytest
from conftest import operation, pool, root
from sqlalchemy import text

from intramind_runtime.contracts import RuntimeConflict

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("sent", [False, True])
async def test_100_lease_recovery_cycles_keep_accounting_and_fencing(store, sent):
    await store.configure_pool(pool(target=1), 1)
    for i in range(100):
        root_id, op_id = f"root-{i}", f"op-{i}"
        await store.create_root(root(root_id=root_id))
        await store.submit_operation(operation(op_id, root_id))
        reservation = await store.reserve_next("p", "old-executor")
        assert reservation is not None
        if sent:
            await store.mark_send(reservation)
        async with store.engine.begin() as c:
            await c.execute(text("""UPDATE runtime_attempts SET lease_expires_at=now()-interval '1 second'
                WHERE attempt_id=:id"""), {"id": reservation.attempt_id})
        assert await store.reconcile_expired() == 1
        assert await store.reconcile_expired() == 0
        if sent:
            assert await store.reserve_next("p", "new-executor") is None
            assert (await store.run(root_id, "t"))["reserved"] == 30
            # Inject independent evidence that the old backend finished, then
            # commit its late output. Lease expiry itself gave no such evidence.
            await store.compute_finished(reservation)
            await store.commit_result(reservation, reservation.operation.payload, 12)
            await store.finish_run(root_id, "t", "SUCCEEDED")
            expected_usage = 12
        else:
            with pytest.raises(RuntimeConflict):
                await store.mark_send(reservation)
            replacement = await store.reserve_next("p", "new-executor")
            assert replacement and replacement.attempt_id != reservation.attempt_id
            await store.fail(replacement, "test_cleanup_before_send", not_sent=True)
            await store.finish_run(root_id, "t", "FAILED")
            expected_usage = 0
        ledger = await store.run(root_id, "t")
        assert ledger["reserved"] == 0 and ledger["spent"] == expected_usage
