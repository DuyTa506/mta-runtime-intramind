"""SQLite handoff, worker death mid-run, and replayed model calls for native PPTX tasks."""

import asyncio
import os
from uuid import uuid4

import httpx
import pytest
from feature_harness import feature_environment

pytestmark = pytest.mark.integration


async def test_native_task_survives_worker_death_and_replays_its_model_call(store, monkeypatch, tmp_path):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment required")
    from unittest.mock import AsyncMock

    import llmai.openai.client as llmai_openai
    from api.background.pptx import native
    from api.background.pptx.native_workflows import native as native_workflow
    from llmai import get_client
    from llmai.shared import JSONSchemaResponse, SystemMessage, UserMessage
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlmodel import SQLModel, select
    from tools.pptx.engine.api.v1.ppt.endpoints import presentation
    from tools.pptx.engine.enums.async_task_status import AsyncTaskStatus
    from tools.pptx.engine.models.generate_presentation_request import GeneratePresentationRequest
    from tools.pptx.engine.models.sql.async_task import AsyncTaskModel
    from tools.pptx.engine.models.sql.native_task import NativeCompletion, NativeTaskOutbox
    from tools.pptx.engine.models.sql.presentation import PresentationModel
    from tools.pptx.engine.models.sql.slide import SlideModel
    from tools.pptx.engine.models.sql.user import User
    from tools.pptx.engine.services import native_completions, native_tasks
    from tools.pptx.engine.services.webhook_outbox import relay_once
    from tools.pptx.engine.utils.llm_config import get_llm_config
    from tools.pptx.engine.utils.llm_utils import generate_response, get_generate_kwargs

    for key, value in {"BACKGROUND_RUNTIME__URL": "http://runtime",
                       "BACKGROUND_RUNTIME__SERVICE_TOKEN": "x" * 32,
                       "BACKGROUND_RUNTIME__BUILD_ID": "build-1",
                       "LLM": "custom", "CUSTOM_LLM_URL": "http://model.invalid/v1",
                       "CUSTOM_MODEL": "tool", "CUSTOM_LLM_API_KEY": "k"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("BACKGROUND_RUNTIME__DIRECT__ENABLED", raising=False)
    monkeypatch.setattr(llmai_openai, "OpenAI", llmai_openai.OpenAI)
    native_completions.install()

    model_calls = []

    def model(request):
        model_calls.append(request)
        return httpx.Response(200, json={
            "id": "m", "object": "chat.completion", "created": 0, "model": "tool",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"title": "Kế hoạch"}'}}]})

    monkeypatch.setattr(native_completions.httpx, "HTTPTransport", lambda: httpx.MockTransport(model))

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/native.db")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(lambda conn: SQLModel.metadata.create_all(conn, tables=[
            User.__table__, AsyncTaskModel.__table__, NativeTaskOutbox.__table__,
            NativeCompletion.__table__, PresentationModel.__table__, SlideModel.__table__]))
    monkeypatch.setattr(native_tasks, "async_session_maker", sessions)

    presentation_id = uuid4()
    monkeypatch.setattr(presentation, "check_if_api_request_is_valid",
                        AsyncMock(return_value=(presentation_id,)))
    async with sessions() as sql:
        task = await presentation.generate_presentation_async(
            GeneratePresentationRequest(content="Nội dung.", n_slides=2), sql)

    attempts, first_attempt_running = [], asyncio.Event()

    async def runner(request, received_id, task_id, cookie):
        attempts.append(task_id)
        client = get_client(config=get_llm_config())
        answer = (await generate_response(client, **get_generate_kwargs(
            model="tool", messages=[SystemMessage(content="JSON."), UserMessage(content="Tiêu đề?")],
            max_tokens=32, response_format=JSONSchemaResponse(name="t", json_schema={
                "type": "object", "properties": {"title": {"type": "string"}}})))).content
        if len(attempts) == 1:
            first_attempt_running.set()
            await asyncio.Event().wait()
        async with sessions() as sql:
            row = await sql.get(AsyncTaskModel, task_id)
            row.status, row.data = AsyncTaskStatus.COMPLETED, {"answer": answer}
            await sql.commit()

    monkeypatch.setattr(presentation, "_run_generate_presentation_task", runner)
    limits = {"max_batch_size": 1, "max_delay_seconds": 0, "max_item_bytes": 4_194_304,
              "max_pending_per_tenant": 128, "max_pending_per_partition": 1}

    async def unexpected_inference(*args):
        pytest.fail("native task inference is foreground HTTP, not a broker reservation")

    try:
        async with feature_environment(store, name="pptx.native", workflows=[native_workflow],
                buffering=limits, respond=unexpected_inference,
                build_activities=lambda factory: [native.NativeActivities(factory).run]) as env:
            monkeypatch.setattr(native, "RuntimeClient", lambda url, token, tenant: env.runtime_client(tenant))
            api = env.runtime_client("system:pptx")
            try:
                # The committing process is gone: only the rows it committed remain.
                await relay_once(sessions, native.accept, NativeTaskOutbox, "Native task")
                async with sessions() as sql:
                    handoff = await sql.get(NativeTaskOutbox, task.id)
                assert handoff.receipt and handoff.payload is None

                await asyncio.wait_for(first_attempt_running.wait(), timeout=60)
                async with sessions() as sql:
                    assert len(list(await sql.scalars(select(NativeCompletion)))) == 1
                # Worker death mid-generation; Temporal retries on the replacement worker.
                await asyncio.wait_for(env.restart_worker(), timeout=60)

                async with asyncio.timeout(120):
                    while True:
                        buffered = await api.get_buffered(handoff.receipt)
                        if buffered.get("state") == "SUCCEEDED":
                            break
                        await asyncio.sleep(0.1)
                run_id = buffered["run_id"]
                await asyncio.wait_for(env.temporal.get_workflow_handle(run_id).result(), timeout=30)
                state = await api.get_run(run_id)
                assert state["state"] == "SUCCEEDED"
                assert await api.read_json(state["result"]) == {"task_id": task.id, "status": "completed"}
                await env.replay(run_id)

                assert attempts == [task.id, task.id]
                assert len(model_calls) == 1, "the retry must replay the recorded completion"
                async with sessions() as sql:
                    row = await sql.get(AsyncTaskModel, task.id)
                    assert row.status == AsyncTaskStatus.COMPLETED
                    assert row.data == {"answer": {"title": "Kế hoạch"}}
                    assert list(await sql.scalars(select(NativeCompletion))) == []

                # A late retry of the same handoff never generates twice.
                assert await native_tasks.execute({
                    "task_id": task.id, "kind": "presentation.generate", "tenant_id": "system:pptx",
                    "arguments": {}}) == "completed"
                assert attempts == [task.id, task.id] and len(model_calls) == 1
                assert env.calls == []
            finally:
                await api.close()
    finally:
        await engine.dispose()
