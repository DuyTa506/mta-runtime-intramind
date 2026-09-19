import asyncio
import os
import sys
from pathlib import Path

import pytest
from conftest import operation, pool, root
from sqlalchemy import text

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

    await migrate("0001")
    await store.create_root(root("migration-preserved"))
    await store.configure_pool(pool(target=1), 1)
    await store.submit_operation(operation("legacy", "migration-preserved"))
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
    await migrate()
    await migrate()
    state = await store.run("migration-preserved", "t")
    assert state["state"] == "RUNNING" and state["reserved"] == 30
    assert state["resource_budgets"] == {} and state["cleanup_pending"]
    async with store.engine.connect() as connection:
        assert (await connection.execute(text("SELECT version_num FROM alembic_version"))).scalar_one() == "0003"
        assert (await connection.execute(text("SELECT count(*) FROM runtime_buffer_items"))).scalar_one() == 0
        assert (await connection.execute(text("SELECT budget_unit FROM runtime_attempts"))).scalar_one() == "tokens"
    await store.confirm_epoch_stopped("p", "e1", "test legacy engine stopped")
    state = await store.run("migration-preserved", "t")
    assert state["spent"] == 30 and state["reserved"] == 0 and not state["cleanup_pending"]
