import asyncio
import os
import sys
from pathlib import Path

import pytest
from conftest import root
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

    async def migrate():
        process = await asyncio.create_subprocess_exec(sys.executable, "-m", "alembic", "upgrade", "head",
            cwd=directory, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        output, _ = await asyncio.wait_for(process.communicate(), timeout=30)
        assert process.returncode == 0, output.decode()

    await migrate()
    await store.create_root(root("migration-preserved"))
    await migrate()
    assert (await store.run("migration-preserved", "t"))["state"] == "RUNNING"
    async with store.engine.connect() as connection:
        assert (await connection.execute(text("SELECT version_num FROM alembic_version"))).scalar_one() == "0001"
