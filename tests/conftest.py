import os
from datetime import UTC, datetime, timedelta
from importlib.resources import files

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url

from intramind_runtime.artifacts import tenant_prefix
from intramind_runtime.contracts import Artifact, OperationSpec, PoolSpec, RootSpec, digest
from intramind_runtime.store import Store


@pytest.fixture
async def store():
    url = os.environ.get("RUNTIME_TEST_DATABASE_URL")
    if not url:
        pytest.skip("explicit disposable RUNTIME_TEST_DATABASE_URL required")
    parsed = make_url(url)
    if (parsed.database != "runtime_test" or parsed.host not in {"127.0.0.1", "localhost"}
        or parsed.port not in {55439, 55440} or os.environ.get("RUNTIME_TEST_ALLOW_RESET") != "yes"):
        pytest.fail("refusing to reset a database without the runtime_test name and explicit reset opt-in")
    store = Store(url)
    async with store.engine.begin() as c:
        await c.execute(text("DROP SCHEMA public CASCADE"))
        await c.execute(text("CREATE SCHEMA public"))
        for statement in files("intramind_runtime").joinpath("schema.sql").read_text().split(";"):
            if statement.strip():
                await c.execute(text(statement))
    yield store
    await store.close()


def root(root_id="r", tenant_id="t", **kwargs):
    return RootSpec(root_id=root_id, tenant_id=tenant_id, deadline=datetime.now(UTC)+timedelta(hours=1),
                    budget_limit=kwargs.pop("budget_limit", 10000), **kwargs)


def operation(op="o", root_id="r", tenant_id="t", **kwargs):
    data = b'{"messages":[{"role":"user","content":"test"}]}'
    return OperationSpec(operation_id=op, root_id=root_id, tenant_id=tenant_id,
        payload=Artifact(key=tenant_prefix(tenant_id)+digest(data), sha256=digest(data), size=len(data)),
        model_profile="test", input_tokens_bound=10, max_output_tokens=20, expected_cost=10, **kwargs)


def pool(pool_id="p", **kwargs):
    return PoolSpec(pool_id=pool_id, group_id="gpu", engine_epoch="e1", profile_id="test-v1",
        model_profile="test", model_revision="model-1", hard_ceiling=4, target=kwargs.pop("target", 2),
        context_limit=1024, valid_until=datetime.now(UTC)+timedelta(hours=1), **kwargs)
