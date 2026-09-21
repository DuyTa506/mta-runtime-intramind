"""Internal authenticated API. Gateway remains the public identity authority."""

import hmac
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from hashlib import sha256

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import Field

from .artifacts import MAX_ARTIFACT_BYTES, ArtifactPort, tenant_prefix
from .buffering import BufferedSubmission, BufferedSubmissions
from .contracts import AdmissionDenied, Artifact, Contract, NotFound, RootSpec, RuntimeConflict
from .embedding import EmbeddingPrepareRequest
from .preparation import PrepareRequest
from .speech import SpeechPrepareRequest
from .store import Store, row
from .uploads import ArtifactUploads


class Submission(Contract):
    task_type: str = Field(min_length=1, max_length=120)
    submission_key: str = Field(min_length=1, max_length=200)
    input: Artifact
    configuration: Artifact | None = None


def create_app(
    store: Store,
    artifacts: ArtifactPort,
    service_token: str,
    definitions: dict[str, dict],
    control_queue="intramind-control",
    preparers=None,
    *,
    manage_lifecycle=False,
    speech_preparers=None,
    embedding_preparers=None,
    artifact_max_bytes=MAX_ARTIFACT_BYTES,
    artifact_upload_concurrency=2,
    direct_proxies=None,
) -> FastAPI:
    if len(service_token) < 32:
        raise ValueError("service token must contain at least 32 characters")

    @asynccontextmanager
    async def lifespan(app):
        try:
            if manage_lifecycle:
                await artifacts.ready()
            yield
        finally:
            for proxy in (direct_proxies or {}).values():
                await proxy.close()
            if manage_lifecycle:
                for preparer in (preparers or {}).values():
                    await preparer.client.aclose()
                await store.close()

    app = FastAPI(title="Intramind Runtime", version="0.1.0", lifespan=lifespan)
    uploads = ArtifactUploads(artifacts, max_bytes=artifact_max_bytes,
                              concurrency=artifact_upload_concurrency)
    buffers = BufferedSubmissions(store, artifacts, control_queue=control_queue)

    async def tenant(request: Request):
        provided = request.headers.get("authorization", "")
        if not hmac.compare_digest(provided.encode(), f"Bearer {service_token}".encode()):
            raise HTTPException(401, "trusted service authentication required")
        identity = request.headers.get("x-tenant-id", "")
        if not identity or len(identity) > 200:
            raise HTTPException(400, "verified tenant identity required")
        return identity

    from .direct_routes import register

    register(app, tenant, direct_proxies or {}, preparers or {})

    @app.get("/v1/llm/profiles/{model_profile}")
    async def llm_profile(model_profile: str, tenant_id=Depends(tenant)):
        """Expose configured prompt limits for planning; this grants no capacity reservation."""
        preparer = (preparers or {}).get(model_profile)
        if preparer is None:
            raise HTTPException(503, "no qualified tokenizer/template profile for this model")
        return {
            "model_profile": model_profile,
            "model": preparer.model,
            "endpoint_fingerprint": sha256(
                str(preparer.client.base_url).rstrip("/").removesuffix("/v1").encode()
            ).hexdigest(),
            "capacity_profile_id": preparer.profile_id,
            "context_limit": preparer.context_limit,
            "response_formats": sorted(preparer.response_formats),
            "allow_tool_calls": preparer.allow_tool_calls,
        }

    @app.post("/v1/requests/prepare")
    async def prepare(request: PrepareRequest, tenant_id=Depends(tenant)):
        preparer = (preparers or {}).get(request.model_profile)
        if preparer is None:
            raise HTTPException(503, "no qualified tokenizer/template profile for this model")
        try:
            return await preparer.prepare(request, artifacts, tenant_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.exception_handler(RuntimeConflict)
    async def conflict(request, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.get("/v1/speech/profiles/{model_profile}")
    async def speech_profile(model_profile: str, tenant_id=Depends(tenant)):
        preparer = (speech_preparers or {}).get(model_profile)
        if preparer is None:
            raise HTTPException(503, "no qualified speech profile")
        return preparer.profile.model_dump(mode="json")

    @app.post("/v1/speech/prepare")
    async def prepare_speech(request: SpeechPrepareRequest, tenant_id=Depends(tenant)):
        preparer = (speech_preparers or {}).get(request.model_profile)
        if preparer is None:
            raise HTTPException(503, "no qualified speech profile")
        try:
            return await preparer.prepare(request, artifacts, tenant_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/v1/embedding/profiles/{model_profile}")
    async def embedding_profile(model_profile: str, tenant_id=Depends(tenant)):
        preparer = (embedding_preparers or {}).get(model_profile)
        if preparer is None:
            raise HTTPException(503, "no qualified embedding profile")
        return preparer.profile.model_dump(mode="json")

    @app.post("/v1/embedding/prepare")
    async def prepare_embedding(request: EmbeddingPrepareRequest, tenant_id=Depends(tenant)):
        preparer = (embedding_preparers or {}).get(request.model_profile)
        if preparer is None:
            raise HTTPException(503, "no qualified embedding profile")
        try:
            return await preparer.prepare(request, artifacts, tenant_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.exception_handler(AdmissionDenied)
    async def denied(request, exc):
        return JSONResponse(
            status_code=429 if exc.retryable else 422,
            content={"detail": str(exc)},
            headers={"Retry-After": "5"} if exc.retryable else {},
        )

    @app.exception_handler(NotFound)
    async def missing(request, exc):
        return JSONResponse(status_code=404, content={"detail": "not found"})

    @app.get("/health")
    async def health():
        async with store.engine.connect() as c:
            await row(c, "SELECT id FROM runtime_authority WHERE id=1")
        return {"status": "ledger_ready"}

    @app.get("/metrics")
    async def metrics():
        from .metrics import snapshot

        return Response(await snapshot(store), media_type="text/plain; version=0.0.4")

    @app.post("/v1/artifacts", response_model=Artifact)
    async def upload(request: Request, tenant_id=Depends(tenant)):
        return await uploads.receive(request, tenant_id)

    @app.post("/v1/artifacts/read")
    async def download(ref: Artifact, tenant_id=Depends(tenant)):
        if not ref.key.startswith(tenant_prefix(tenant_id)):
            raise NotFound("artifact")
        return Response(await artifacts.get(ref), media_type=ref.content_type)

    @app.post("/v1/runs", status_code=202)
    async def submit(request: Submission, tenant_id=Depends(tenant)):
        definition = definitions.get(request.task_type)
        if definition is None:
            raise HTTPException(422, "task type/version is not registered")
        if not request.input.key.startswith(tenant_prefix(tenant_id)):
            raise NotFound("input")
        await artifacts.get(request.input)
        if request.configuration:
            if not request.configuration.key.startswith(tenant_prefix(tenant_id)):
                raise NotFound("configuration")
            await artifacts.get(request.configuration)
        root_id = sha256(f"{tenant_id}\x00{request.submission_key}".encode()).hexdigest()
        deadline = datetime.now(UTC) + timedelta(seconds=definition["deadline_seconds"])
        root = RootSpec(
            root_id=root_id,
            tenant_id=tenant_id,
            deadline=deadline,
            budget_limit=definition["budget_limit"],
            resource_budgets=definition.get("resource_budgets", {}),
            priority=definition.get("priority", "background"),
        )
        # Stable submit intent excludes wall-clock deadline; retries attach to
        # the original root and cannot extend its lifetime.
        spec = {
            "workflow_type": request.task_type,
            "task_queue": definition["task_queue"],
            "input": {
                "root_id": root_id,
                "tenant_id": tenant_id,
                "control_queue": control_queue,
                "input": request.input.model_dump(mode="json"),
            },
        }
        if request.configuration:
            spec["input"]["configuration"] = request.configuration.model_dump(mode="json")
        run_id = await store.submit_run(root, request.submission_key, request.input.sha256, spec)
        return {"task_id": run_id, "run_id": run_id, "status": "submitted", "owner": "temporal"}

    @app.post("/v1/buffers", status_code=202)
    async def buffer(request: BufferedSubmission, tenant_id=Depends(tenant)):
        definition = definitions.get(request.task_type)
        if definition is None:
            raise HTTPException(422, "task type/version is not registered")
        for ref in (request.input, request.configuration):
            if not ref.key.startswith(tenant_prefix(tenant_id)):
                raise NotFound("artifact")
            await artifacts.get(ref)
        item_id = await buffers.append(tenant_id, request, definition)
        return {"item_id": item_id, "status": "buffered", "owner": "temporal"}

    @app.get("/v1/buffers/{item_id}")
    async def buffer_status(item_id: str, tenant_id=Depends(tenant)):
        return await buffers.status(tenant_id, item_id)

    @app.get("/v1/runs/{run_id}")
    async def status(run_id: str, tenant_id=Depends(tenant)):
        return await store.run(run_id, tenant_id)

    @app.post("/v1/runs/{run_id}/cancel", status_code=202)
    async def cancel(run_id: str, tenant_id=Depends(tenant)):
        await store.cancel(run_id, tenant_id)
        return {"run_id": run_id, "cancellation": "requested"}

    @app.get("/v1/operations/{operation_id}")
    async def operation(operation_id: str, tenant_id=Depends(tenant)):
        return await store.operation(operation_id, tenant_id)

    return app
