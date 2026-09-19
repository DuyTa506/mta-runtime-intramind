"""Explicit Temporal namespace and worker-version administration."""

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime

from google.protobuf.duration_pb2 import Duration
from temporalio.api.deployment.v1 import WorkerDeploymentVersion
from temporalio.api.enums.v1 import TaskQueueType
from temporalio.api.taskqueue.v1 import TaskQueue
from temporalio.api.workflowservice.v1 import (
    DescribeNamespaceRequest,
    DescribeTaskQueueRequest,
    DescribeWorkerDeploymentRequest,
    DescribeWorkerDeploymentVersionRequest,
    RegisterNamespaceRequest,
    SetWorkerDeploymentCurrentVersionRequest,
)
from temporalio.client import Client
from temporalio.service import RPCError, RPCStatusCode


async def worker_health(client, workers, *, max_poller_age=120, now=None):
    """Check exact deployments and recent queue pollers without changing routing."""
    now = now or datetime.now(UTC)
    await client.workflow_service.describe_namespace(
        DescribeNamespaceRequest(namespace=client.namespace)
    )
    observations = []
    unavailable = False
    pending_promotion = False
    for deployment, build, queue, kind in workers:
        observation = {"deployment": deployment, "build_id": build, "task_queue": queue}
        observations.append(observation)
        try:
            await client.workflow_service.describe_worker_deployment_version(
                DescribeWorkerDeploymentVersionRequest(
                    namespace=client.namespace,
                    deployment_version=WorkerDeploymentVersion(
                        deployment_name=deployment, build_id=build
                    ),
                )
            )
            described = await client.workflow_service.describe_worker_deployment(
                DescribeWorkerDeploymentRequest(
                    namespace=client.namespace, deployment_name=deployment
                )
            )
        except RPCError as exc:
            if exc.status != RPCStatusCode.NOT_FOUND:
                raise
            observation["status"] = "not_registered"
            unavailable = True
            continue

        types = {
            "activity": [TaskQueueType.TASK_QUEUE_TYPE_ACTIVITY],
            "workflow": [TaskQueueType.TASK_QUEUE_TYPE_WORKFLOW],
            "both": [
                TaskQueueType.TASK_QUEUE_TYPE_ACTIVITY,
                TaskQueueType.TASK_QUEUE_TYPE_WORKFLOW,
            ],
        }[kind]
        polled_types = set()
        for queue_type in types:
            # DEFAULT describes current Worker Deployment pollers. ENHANCED's
            # versions_info belongs to the deprecated build-ID routing API.
            described_queue = await client.workflow_service.describe_task_queue(
                DescribeTaskQueueRequest(
                    namespace=client.namespace,
                    task_queue=TaskQueue(name=queue),
                    task_queue_type=queue_type,
                )
            )
            for poller in described_queue.pollers:
                options = poller.deployment_options
                age = (now - poller.last_access_time.ToDatetime(tzinfo=UTC)).total_seconds()
                if (
                    options.deployment_name == deployment
                    and options.build_id == build
                    and -5 <= age <= max_poller_age
                ):
                    polled_types.add(queue_type)
        observation["polling"] = all(queue_type in polled_types for queue_type in types)
        routing = described.worker_deployment_info.routing_config
        current = routing.current_deployment_version
        is_current = current.deployment_name == deployment and current.build_id == build
        # Older server responses may only populate the deprecated version string.
        if not current.build_id and routing.current_version:
            is_current = routing.current_version == f"{deployment}.{build}"
        observation["current"] = is_current
        if not observation["polling"]:
            observation["status"] = "no_recent_poller"
            unavailable = True
        elif not is_current:
            observation["status"] = "pending_promotion"
            pending_promotion = True
        else:
            observation["status"] = "ready"
    status = "unavailable" if unavailable else "pending_promotion" if pending_promotion else "ready"
    return {"namespace": client.namespace, "status": status, "workers": observations}


async def execute(args):
    client = await Client.connect(args.address, namespace=args.namespace)
    if args.command == "init-namespace":
        try:
            await client.workflow_service.describe_namespace(
                DescribeNamespaceRequest(namespace=args.namespace)
            )
        except RPCError as exc:
            if exc.status != RPCStatusCode.NOT_FOUND:
                raise
            await client.workflow_service.register_namespace(
                RegisterNamespaceRequest(
                    namespace=args.namespace,
                    description="Intramind durable background workflows",
                    workflow_execution_retention_period=Duration(
                        seconds=args.retention_days * 86400
                    ),
                )
            )
    elif args.command == "promote":
        await client.workflow_service.set_worker_deployment_current_version(
            SetWorkerDeploymentCurrentVersionRequest(
                namespace=args.namespace,
                deployment_name=args.deployment,
                build_id=args.build_id,
                identity="intramind-release-operator",
            )
        )
    elif args.command == "health":
        report = await worker_health(client, args.worker, max_poller_age=args.max_poller_age)
        print(json.dumps(report, sort_keys=True))
        if report["status"] == "unavailable":
            return 1
        if report["status"] == "pending_promotion" and not args.allow_unpromoted:
            return 3
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--address", default=os.environ.get("RUNTIME_TEMPORAL_ADDRESS", "temporal:7233")
    )
    parser.add_argument(
        "--namespace", default=os.environ.get("RUNTIME_TEMPORAL_NAMESPACE", "intramind")
    )
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init-namespace")
    initialize.add_argument("--retention-days", type=int, choices=range(1, 91), default=30)
    promote = commands.add_parser("promote")
    promote.add_argument("deployment")
    promote.add_argument("build_id")
    health = commands.add_parser("health")
    health.add_argument(
        "--worker",
        nargs=4,
        action="append",
        required=True,
        metavar=("DEPLOYMENT", "BUILD_ID", "QUEUE", "KIND"),
        help="Repeat per worker; KIND is activity, workflow or both",
    )
    health.add_argument("--max-poller-age", type=int, default=120)
    health.add_argument("--allow-unpromoted", action="store_true")
    args = parser.parse_args()
    if args.command == "health":
        if args.max_poller_age <= 0 or any(
            item[3] not in {"activity", "workflow", "both"} for item in args.worker
        ):
            parser.error("positive max-poller-age and valid worker KIND required")
    raise SystemExit(asyncio.run(execute(args)))


if __name__ == "__main__":
    main()
