import os
import re
from datetime import UTC, datetime, timedelta
from importlib.resources import files
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url

from intramind_runtime.artifacts import tenant_prefix
from intramind_runtime.contracts import Artifact, OperationSpec, PoolSpec, RootSpec, digest
from intramind_runtime.store import Store

_TEMPORAL_NAMESPACE = "intramind-runtime-test-" + uuid4().hex


async def temporal_test_client():
    """Isolate qualification runs without deleting the histories of earlier test sessions."""
    from google.protobuf.duration_pb2 import Duration
    from temporalio.api.workflowservice.v1 import RegisterNamespaceRequest
    from temporalio.client import Client
    from temporalio.service import RPCError, RPCStatusCode

    address = os.environ.get("RUNTIME_TEST_TEMPORAL_ADDRESS")
    if not address:
        pytest.skip("explicit disposable RUNTIME_TEST_TEMPORAL_ADDRESS required")
    if address not in {"127.0.0.1:17233", "127.0.0.1:17234"}:
        pytest.fail("disposable loopback Temporal test port required")
    namespace = os.environ.get("RUNTIME_TEST_TEMPORAL_NAMESPACE", _TEMPORAL_NAMESPACE)
    if not re.fullmatch(r"intramind-runtime-test(?:-[a-z0-9-]+)?", namespace):
        pytest.fail("refusing to use a namespace outside intramind-runtime-test")
    client = await Client.connect(address, namespace=namespace)
    try:
        await client.workflow_service.register_namespace(RegisterNamespaceRequest(
            namespace=namespace, workflow_execution_retention_period=Duration(seconds=86400)))
    except RPCError as exc:
        if exc.status != RPCStatusCode.ALREADY_EXISTS:
            raise
    return client


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
        for name in ("schema.sql", "speech_budget.sql", "buffering.sql", "embedding_budget.sql",
                     "direct.sql", "admission_v2.sql"):
            for statement in files("intramind_runtime").joinpath(name).read_text().split(";"):
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
        context_limit=1024,
        valid_until=kwargs.pop("valid_until", datetime.now(UTC)+timedelta(hours=1)), **kwargs)
