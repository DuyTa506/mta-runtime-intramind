"""Disposable Temporal/Postgres feature harness with actual broker accounting."""

import asyncio
import json
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace
from uuid import uuid4

import httpx
from conftest import pool, temporal_test_client
from fakes import MemoryArtifacts
from temporalio.api.workflowservice.v1 import SetWorkerDeploymentCurrentVersionRequest
from temporalio.common import VersioningBehavior, WorkerDeploymentVersion
from temporalio.service import RPCError
from temporalio.worker import Replayer, Worker, WorkerDeploymentConfig

from intramind_runtime.api import create_app
from intramind_runtime.client import RuntimeClient
from intramind_runtime.contracts import SpeechPoolSpec
from intramind_runtime.drivers import OpenAICompletionDriver
from intramind_runtime.executor import Executor
from intramind_runtime.preparation import LlamaCppPromptSizer
from intramind_runtime.speech import ServingSpeechDriver, SpeechPreparer
from intramind_runtime.temporal_adapter import BrokerActivities, OutboxPublisher


def peak_children(history):
    active, peak = 0, 0
    for event in history.events:
        if event.HasField("start_child_workflow_execution_initiated_event_attributes"):
            active += 1
            peak = max(active, peak)
        elif any(
            event.HasField(f"child_workflow_execution_{state}_event_attributes")
            for state in ("completed", "failed", "canceled", "timed_out", "terminated")
        ):
            active -= 1
        elif event.HasField("start_child_workflow_execution_failed_event_attributes"):
            active -= 1
        assert active >= 0
    assert active == 0
    return peak


@asynccontextmanager
async def feature_environment(
    store, *, name, workflows, build_activities, respond, allow_tool_calls=False,
    speech_profile=None, respond_speech=None, tokenize_prompt=None,
):
    temporal = await temporal_test_client()
    namespace, uid = temporal.namespace, uuid4().hex
    queue, token = name + "-" + uid, "test-runtime-feature-token-1234567890"
    blobs, calls, runs = MemoryArtifacts(), [], []
    spec = pool(target=1).model_copy(update={"context_limit": 16384})
    await store.configure_pool(spec, 1)
    speech_driver, speech_executor, speech_preparers = None, None, {}
    if speech_profile is not None:
        assert respond_speech is not None
        await store.configure_pool(SpeechPoolSpec(
            pool_id="voice", group_id="cpu", engine_epoch="voice-e1",
            profile_id=speech_profile.capacity_profile_id,
            model_profile=speech_profile.model_profile, model_revision="voice-model-1",
            hard_ceiling=1, target=1, character_limit=speech_profile.character_limit,
            valid_until=spec.valid_until,
        ), 1)
        speech_driver = ServingSpeechDriver("http://voice/", speech_profile, client=httpx.AsyncClient(
            base_url="http://voice/", transport=httpx.MockTransport(respond_speech)))
        speech_executor = Executor(store, blobs, speech_driver, "voice", "voice-executor-" + uid)
        speech_preparers = {speech_profile.model_profile: SpeechPreparer(speech_profile)}

    async def tokenize(request):
        return httpx.Response(200, json={"prompt": "test template", "tokens": [1] * 10})

    sizing = httpx.AsyncClient(base_url="http://fake/", transport=httpx.MockTransport(tokenize_prompt or tokenize))
    preparer = LlamaCppPromptSizer(
        sizing,
        model="test",
        capacity_profile_id=spec.profile_id,
        context_limit=spec.context_limit,
        token_margin=2,
        expected_output_tokens=50,
        response_formats=frozenset({"text", "json_schema", "json_object"}),
        allow_tool_calls=allow_tool_calls,
    )
    app = create_app(
        store,
        blobs,
        token,
        {
            name + "/v1": {"deadline_seconds": 120, "budget_limit": 1000000, "task_queue": queue,
                           "resource_budgets": {"speech_characters": 100000} if speech_profile else {}},
        },
        queue,
        {"test": preparer},
        speech_preparers=speech_preparers,
    )

    def runtime_client(tenant):
        return RuntimeClient(
            "http://runtime",
            token,
            tenant,
            client=httpx.AsyncClient(
                base_url="http://runtime",
                transport=httpx.ASGITransport(app=app),
                headers={"Authorization": "Bearer " + token, "X-Tenant-ID": tenant},
            ),
        )

    reservations = {}

    async def completion_http(request):
        attempt_id = request.headers["X-Intramind-Attempt-ID"]
        payload = json.loads(request.content)
        calls.append(attempt_id)
        body = await respond(reservations[attempt_id], payload)
        body.setdefault("usage", {"prompt_tokens": 10, "completion_tokens": 20})
        return httpx.Response(200, json=body)

    class Engine(OpenAICompletionDriver):
        async def execute(self, reservation, payload):
            reservations[reservation.attempt_id] = reservation
            try:
                return await super().execute(reservation, payload)
            finally:
                reservations.pop(reservation.attempt_id)

    completion_driver = Engine("http://fake/v1/", "test-only", "test", client=httpx.AsyncClient(
        base_url="http://fake/v1/", transport=httpx.MockTransport(completion_http)))
    executor = Executor(store, blobs, completion_driver, "p", "feature-executor-" + uid)
    publisher = OutboxPublisher(store, temporal, "feature-publisher-" + uid)
    broker = BrokerActivities(store, blobs)
    version = WorkerDeploymentVersion("feature-" + uid, "build-1")
    worker = Worker(
        temporal,
        task_queue=queue,
        workflows=workflows,
        activities=[broker.submit_or_attach, broker.submit_speech, broker.finish,
                    *build_activities(runtime_client)],
        deployment_config=WorkerDeploymentConfig(
            version=version,
            use_worker_versioning=True,
            default_versioning_behavior=VersioningBehavior.PINNED,
        ),
    )

    async def pump():
        while True:
            await publisher.tick()
            await executor.tick()
            if speech_executor:
                await speech_executor.tick()
            await asyncio.sleep(0.01)

    async def submit(source, *, tenant="user:test", configuration=None):
        api = runtime_client(tenant)
        try:
            ref = await api.put_json(source)
            submission = {"task_type": name + "/v1", "submission_key": uid, "input": ref}
            if configuration is not None:
                submission["configuration"] = await api.put_json(configuration)
            result = await api.submit(submission)
            runs.append(result["run_id"])
            for _ in range(2000):
                status = await api.get_run(result["run_id"])
                if status["state"] != "RUNNING":
                    break
                await asyncio.sleep(0.05)
            assert status["state"] in {"SUCCEEDED", "PARTIAL"}, status
            handle = temporal.get_workflow_handle(result["run_id"])
            await asyncio.wait_for(handle.result(), timeout=10)
            return status, await api.read_json(status["result"])
        finally:
            await api.close()

    async def replay(run_id):
        history = await temporal.get_workflow_handle(run_id).fetch_history()
        await Replayer(workflows=workflows).replay_workflow(history)
        for event in history.events:
            if event.HasField("child_workflow_execution_started_event_attributes"):
                child = event.child_workflow_execution_started_event_attributes.workflow_execution.workflow_id
                await replay(child)

    async with worker:
        for attempt in range(30):
            try:
                await temporal.workflow_service.set_worker_deployment_current_version(
                    SetWorkerDeploymentCurrentVersionRequest(
                        namespace=namespace,
                        deployment_name=version.deployment_name,
                        build_id=version.build_id,
                        identity="feature-integration-test",
                    )
                )
                break
            except RPCError:
                if attempt == 29:
                    raise
                await asyncio.sleep(0.5)
        work = asyncio.create_task(pump())
        try:
            yield SimpleNamespace(
                submit=submit,
                replay=replay,
                calls=calls,
                blobs=blobs,
                runtime_client=runtime_client,
                temporal=temporal,
            )
        finally:
            for run_id in runs:
                handle = temporal.get_workflow_handle(run_id)
                with suppress(RPCError):
                    if (await handle.describe()).close_time is None:
                        await handle.terminate("disposable feature test cleanup")
            work.cancel()
            with suppress(asyncio.CancelledError):
                await work
            await sizing.aclose()
            await completion_driver.close()
            if speech_driver:
                await speech_driver.close()
