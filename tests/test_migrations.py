import asyncio
import os
import sys
from pathlib import Path

import pytest
from conftest import operation, pool, root
from sqlalchemy import text
from test_speech_ledger import speech, speech_pool

from intramind_runtime.direct import DirectRequest

pytestmark = pytest.mark.integration


async def test_packaged_alembic_upgrade_is_repeatable_and_preserves_data(store):
    # The store fixture already validates the disposable database name,
    # loopback port and explicit reset opt-in before this destructive reset.
    async with store.engine.begin() as connection:
        await connection.execute(text("DROP SCHEMA public CASCADE"))
        await connection.execute(text("CREATE SCHEMA public"))
    env = os.environ | {"RUNTIME_DATABASE_URL": os.environ["RUNTIME_TEST_DATABASE_URL"]}
    directory = Path(__file__).resolve().parents[1]

    async def migrate(target="head"):
        process = await asyncio.create_subprocess_exec(sys.executable, "-m", "alembic", "upgrade", target,
            cwd=directory, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        output, _ = await asyncio.wait_for(process.communicate(), timeout=30)
        assert process.returncode == 0, output.decode()

    async def legacy_insert_operation(spec):
        # Seed old schemas with their own wire shape. The current Store must not
        # depend on new admission tables before migration 0006 is installed.
        async with store.engine.begin() as connection:
            await connection.execute(text("""INSERT INTO runtime_operations
                (operation_id,root_id,tenant_id,spec)
                VALUES (:id,:root,:tenant,CAST(:spec AS jsonb))"""), {
                "id": spec.operation_id, "root": spec.root_id,
                "tenant": spec.tenant_id, "spec": spec.model_dump_json(),
            })
            await connection.execute(text("""UPDATE runtime_roots
                SET operation_count=operation_count+1 WHERE root_id=:root"""),
                {"root": spec.root_id})

    async def legacy_configure_pool(spec, group_ceiling):
        async with store.engine.begin() as connection:
            await connection.execute(text("""INSERT INTO runtime_groups(group_id,hard_ceiling)
                VALUES (:group,:ceiling) ON CONFLICT(group_id) DO UPDATE
                SET hard_ceiling=EXCLUDED.hard_ceiling"""), {
                "group": spec.group_id, "ceiling": group_ceiling,
            })
            await connection.execute(text("""INSERT INTO runtime_pools
                (pool_id,group_id,spec,engine_epoch,target,hard_ceiling,context_limit,
                 model_profile,valid_until)
                VALUES (:id,:group,CAST(:spec AS jsonb),:epoch,:target,:ceiling,:context,
                    :model,:until)"""), {
                "id": spec.pool_id, "group": spec.group_id, "spec": spec.model_dump_json(),
                "epoch": spec.engine_epoch, "target": spec.target,
                "ceiling": spec.hard_ceiling, "context": getattr(spec, "context_limit", 0),
                "model": spec.model_profile, "until": spec.valid_until,
            })

    await migrate("0001")
    await store.create_root(root("migration-preserved"))
    await legacy_configure_pool(pool(target=1), 1)
    await legacy_insert_operation(operation("legacy", "migration-preserved"))
    async with store.engine.begin() as connection:
        await connection.execute(text("""UPDATE runtime_roots SET reserved=30,attempts=1,
            spec=spec-'resource_budgets' WHERE root_id='migration-preserved'"""))
        await connection.execute(text("""INSERT INTO runtime_attempts
            (attempt_id,operation_id,pool_id,engine_epoch,attempt_number,owner_id,lease_epoch,
             lease_expires_at,state,budget_bound)
            VALUES ('old-attempt','legacy','p','e1',1,'old-worker',1,now()+interval '1 hour',
                    'SEND_INTENT',30)"""))
        await connection.execute(text("""UPDATE runtime_operations SET state='EXECUTING',
            active_attempt='old-attempt',attempts=1 WHERE operation_id='legacy'"""))
    await migrate("0003")
    await store.create_root(root("speech-preserved", resource_budgets={"speech_characters": 12}))
    await legacy_configure_pool(speech_pool(), 4)
    await legacy_insert_operation(speech().model_copy(update={"root_id": "speech-preserved"}))
    async with store.engine.begin() as connection:
        await connection.execute(text("""UPDATE runtime_roots SET attempts=1
            WHERE root_id='speech-preserved'"""))
        await connection.execute(text("""UPDATE runtime_resource_budgets SET reserved=12
            WHERE root_id='speech-preserved' AND unit='speech_characters'"""))
        await connection.execute(text("""INSERT INTO runtime_attempts
            (attempt_id,operation_id,pool_id,engine_epoch,attempt_number,owner_id,lease_epoch,
             lease_expires_at,state,budget_bound,budget_unit)
            VALUES ('old-speech-attempt','speech','voice','e1',1,'old-speech-worker',1,
                    now()+interval '1 hour','SEND_INTENT',12,'speech_characters')"""))
        await connection.execute(text("""UPDATE runtime_operations SET state='EXECUTING',
            active_attempt='old-speech-attempt',attempts=1 WHERE operation_id='speech'"""))
    await migrate("0006")
    # Recreate the rc17 column constraint before the additive rc18 upgrade.
    async with store.engine.begin() as connection:
        await connection.execute(text(
            "ALTER TABLE runtime_pools ALTER COLUMN valid_until SET NOT NULL"))
    await store.configure_pool(pool("direct-after-upgrade", target=1), 4)
    direct_request = DirectRequest(
        request_id="upgrade-preserved", tenant_id="t", payload_digest="a" * 64,
        model_profile="test", capacity_profile_id="test-v1", request_bound=30,
        deadline=root().deadline,
    )
    # Keep a legacy rc17 row to prove rc18's additive rollout and rollback.
    async with store.engine.begin() as connection:
        await connection.execute(text("""INSERT INTO runtime_direct_attempts
            (attempt_id,request_id,tenant_id,spec,pool_id,engine_epoch,owner_id,
             state,lease_expires_at,deadline,unknown_at)
            VALUES ('old-direct','upgrade-preserved','t',CAST(:spec AS jsonb),
                'direct-after-upgrade','e1','direct-owner','UNKNOWN',
                now()+interval '1 hour',:deadline,now())"""),
            {"spec": direct_request.model_dump_json(), "deadline": direct_request.deadline})
    await migrate()
    state = await store.run("migration-preserved", "t")
    assert state["state"] == "RUNNING" and state["reserved"] == 30
    assert state["resource_budgets"] == {} and state["cleanup_pending"]
    assert (await store.run("speech-preserved", "t"))["resource_budgets"]["speech_characters"] == {
        "limit": 12, "reserved": 12, "spent": 0,
    }
    async with store.engine.connect() as connection:
        assert (await connection.execute(text("SELECT version_num FROM alembic_version"))).scalar_one() == "0007"
        assert (await connection.execute(text("""SELECT is_nullable FROM information_schema.columns
            WHERE table_name='runtime_pools' AND column_name='valid_until'"""))).scalar_one() == "YES"
        assert (await connection.execute(text("SELECT count(*) FROM runtime_buffer_items"))).scalar_one() == 0
        assert (await connection.execute(text("SELECT budget_unit FROM runtime_attempts WHERE attempt_id='old-attempt'"))).scalar_one() == "tokens"
        assert (await connection.execute(text("""SELECT count(*) FROM runtime_inflight_attempts
            WHERE compute_held"""))).scalar_one() == 3
    status = await store.drain_status()
    assert status["compute_held"] == 3 and status["unknown_attempts"] == 1
    await store.confirm_epoch_stopped("p", "e1", "test legacy engine stopped")
    state = await store.run("migration-preserved", "t")
    assert state["spent"] == 30 and state["reserved"] == 0 and not state["cleanup_pending"]
    await store.confirm_epoch_stopped("voice", "e1", "test speech engine stopped")
    assert (await store.run("speech-preserved", "t"))["resource_budgets"]["speech_characters"] == {
        "limit": 12, "reserved": 0, "spent": 12,
    }
    assert (await store.drain_status())["compute_held"] == 1
    await store.confirm_epoch_stopped("direct-after-upgrade", "e1", "test direct engine stopped")
    assert (await store.drain_status())["unsettled_attempts"] == 0
    await store.configure_pool(pool("without-review-date", valid_until=None), 1)
