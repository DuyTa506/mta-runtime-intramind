"""A registered version is not ready without recent pollers and explicit promotion."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.protobuf.timestamp_pb2 import Timestamp
from temporalio.api.workflowservice.v1 import (
    DescribeTaskQueueResponse,
    DescribeWorkerDeploymentResponse,
    DescribeWorkerDeploymentVersionResponse,
)
from temporalio.service import RPCError, RPCStatusCode

from intramind_runtime.admin import worker_health

NOW = datetime(2026, 9, 19, tzinfo=UTC)


def client(*, poller_build="build-a", current_build="build-a", age=0, types=(1, 2)):
    access = Timestamp()
    access.FromDatetime(NOW - timedelta(seconds=age))

    def describe_queue(request):
        # Match the current server's DEFAULT response. Versioned workers do
        # not populate the deprecated ENHANCED versions_info map.
        assert request.api_mode == 0
        assert request.task_queue_type in (1, 2)
        return DescribeTaskQueueResponse(
            pollers=[
                {
                    "last_access_time": access,
                    "deployment_options": {
                        "deployment_name": "ai",
                        "build_id": poller_build,
                    },
                }
            ]
            if request.task_queue_type in types
            else []
        )

    routing = DescribeWorkerDeploymentResponse(
        worker_deployment_info={
            "routing_config": {
                "current_deployment_version": {"deployment_name": "ai", "build_id": current_build}
            }
        }
    )
    service = SimpleNamespace(
        describe_namespace=AsyncMock(),
        describe_worker_deployment_version=AsyncMock(
            return_value=DescribeWorkerDeploymentVersionResponse()
        ),
        describe_worker_deployment=AsyncMock(return_value=routing),
        describe_task_queue=AsyncMock(side_effect=describe_queue),
    )
    return SimpleNamespace(namespace="intramind-test", workflow_service=service)


@pytest.mark.asyncio
async def test_health_requires_both_activity_and_workflow_pollers():
    runtime = client(types=(1,))
    report = await worker_health(runtime, [("ai", "build-a", "rewrite", "both")], now=NOW)
    assert report["status"] == "unavailable"
    assert report["workers"][0]["status"] == "no_recent_poller"


@pytest.mark.asyncio
@pytest.mark.parametrize("poller_build,age", [("other-build", 0), ("build-a", 121)])
async def test_registered_but_wrong_or_stale_poller_is_not_ready(poller_build, age):
    report = await worker_health(
        client(poller_build=poller_build, age=age), [("ai", "build-a", "rewrite", "both")], now=NOW
    )
    assert report["status"] == "unavailable"


@pytest.mark.asyncio
async def test_unpromoted_healthy_worker_is_reported_without_changing_routing():
    runtime = client(current_build="old-build")
    report = await worker_health(runtime, [("ai", "build-a", "rewrite", "both")], now=NOW)
    assert report["status"] == "pending_promotion"
    assert report["workers"][0]["polling"] is True
    assert report["workers"][0]["current"] is False
    runtime.workflow_service.describe_namespace.assert_awaited_once()


@pytest.mark.asyncio
async def test_current_worker_with_recent_matching_pollers_is_ready():
    runtime = client()
    report = await worker_health(runtime, [("ai", "build-a", "rewrite", "both")], now=NOW)
    assert report["status"] == "ready"
    request = runtime.workflow_service.describe_worker_deployment_version.await_args.args[0]
    assert request.deployment_version.build_id == "build-a"
    assert request.deployment_version.deployment_name == "ai"
    assert {
        call.args[0].task_queue_type
        for call in runtime.workflow_service.describe_task_queue.await_args_list
    } == {1, 2}


@pytest.mark.asyncio
async def test_missing_version_is_unavailable():
    runtime = client()
    runtime.workflow_service.describe_worker_deployment_version.side_effect = RPCError(
        "missing", RPCStatusCode.NOT_FOUND, b""
    )
    report = await worker_health(runtime, [("ai", "build-a", "rewrite", "activity")], now=NOW)
    assert report["workers"][0]["status"] == "not_registered"
    assert report["status"] == "unavailable"
    runtime.workflow_service.describe_task_queue.assert_not_awaited()


@pytest.mark.asyncio
async def test_namespace_failure_cannot_be_reported_as_healthy():
    runtime = client()
    runtime.workflow_service.describe_namespace.side_effect = RPCError(
        "unavailable", RPCStatusCode.UNAVAILABLE, b""
    )
    with pytest.raises(RPCError):
        await worker_health(runtime, [("ai", "build-a", "rewrite", "activity")], now=NOW)
