"""HTTP caller binding; it does not submit workflows or retry inference."""

import json
import re
from collections.abc import AsyncIterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

_tenant: ContextVar[str | None] = ContextVar("direct_inference_tenant", default=None)
_workload: ContextVar[str] = ContextVar("direct_inference_workload", default="background")
_logical_request: ContextVar[str | None] = ContextVar("direct_inference_request", default=None)
_deadline_seconds: ContextVar[int | None] = ContextVar("direct_inference_deadline", default=None)
WORKLOAD_CLASSES = frozenset({"qa", "user_task", "background", "maintenance"})


@contextmanager
def inference_scope(tenant_id: str | None, *, workload_class: str = "background",
                    logical_request_id: str | None = None,
                    deadline_seconds: int | None = None):
    """Bind an identity verified by the application, including across asyncio.to_thread."""
    if tenant_id is not None and (
        not tenant_id or len(tenant_id) > 240 or not tenant_id.isascii()
        or any(ord(character) <= 32 or ord(character) == 127 for character in tenant_id)
    ):
        raise ValueError("invalid inference identity")
    if workload_class not in WORKLOAD_CLASSES:
        raise ValueError("invalid inference workload")
    if logical_request_id is not None and (
        not logical_request_id or len(logical_request_id) > 200
        or not re.fullmatch(r"[A-Za-z0-9._:-]+", logical_request_id)
    ):
        raise ValueError("invalid logical inference request")
    if deadline_seconds is not None and (type(deadline_seconds) is not int or deadline_seconds <= 0):
        raise ValueError("inference deadline must be positive seconds")
    token = _tenant.set(tenant_id)
    workload_token = _workload.set(workload_class)
    request_token = _logical_request.set(logical_request_id)
    deadline_token = _deadline_seconds.set(deadline_seconds)
    try:
        yield
    finally:
        _deadline_seconds.reset(deadline_token)
        _logical_request.reset(request_token)
        _workload.reset(workload_token)
        _tenant.reset(token)


@dataclass(frozen=True)
class DirectStreamEvent:
    kind: Literal["token", "waiting", "resumed", "recovering", "generation_reset"]
    text: str = ""
    generation: int = 0
    reason: str | None = None


class DirectStreamError(RuntimeError):
    """A managed terminal SSE error with a stable reason for application adapters."""

    def __init__(self, message: str, *, reason: str | None = None,
                 error_type: str | None = None):
        super().__init__(message)
        self.reason = reason
        self.error_type = error_type


@dataclass(frozen=True)
class DirectBinding:
    base_url: str
    service_token: str = field(repr=False)
    model_profile: str

    def __post_init__(self):
        url = httpx.URL(self.base_url)
        if (url.scheme not in {"http", "https"} or not url.host or url.userinfo
            or url.query or url.fragment):
            raise ValueError("invalid runtime URL")
        if len(self.service_token) < 32 or not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", self.model_profile):
            raise ValueError("runtime credential and model profile are required")

    def client_kwargs(self) -> dict:
        url = self.base_url.rstrip("/") + f"/v1/direct/{self.model_profile}/"
        return {"base_url": url, "auth": _IdentityAuth(url, self.service_token),
                "follow_redirects": False}

    async def stream_events(self, payload: dict, *, client: httpx.AsyncClient | None = None
                            ) -> AsyncIterator[DirectStreamEvent]:
        """Read ordered managed stream events without hiding controls in OpenAI SDK chunks.

        A caller may share a long-lived authenticated client. The temporary-client
        branch exists for small callers and tests; inference services should pool.
        """
        if payload.get("stream") is not True:
            raise ValueError("stream_events requires stream=true")
        if client is None:
            async with httpx.AsyncClient(**self.client_kwargs(),
                timeout=httpx.Timeout(None, connect=10),
                transport=httpx.AsyncHTTPTransport(retries=0)) as owned:
                async for event in self.stream_events(payload, client=owned):
                    yield event
            return
        async with client.stream("POST", "chat/completions", json=payload) as response:
            response.raise_for_status()
            event_name = "message"
            data = []
            generation = 0
            async for line in response.aiter_lines():
                if not line:
                    if not data:
                        event_name = "message"
                        continue
                    raw = "\n".join(data)
                    data = []
                    if raw == "[DONE]":
                        return
                    frame = json.loads(raw)
                    if event_name == "intramind.error":
                        raise DirectStreamError(str(frame.get("message") or "managed inference failed"),
                            reason=frame.get("reason"), error_type=frame.get("type"))
                    if event_name == "intramind.control":
                        kind = frame.get("type")
                        if kind not in {"waiting", "resumed", "recovering", "generation_reset"}:
                            raise ValueError("invalid direct stream control event")
                        if kind == "generation_reset":
                            generation = int(frame.get("generation", generation + 1))
                        yield DirectStreamEvent(kind=kind,
                            generation=int(frame.get("generation", generation)),
                            reason=frame.get("reason"))
                    else:
                        for choice in frame.get("choices", []):
                            token = choice.get("delta", {}).get("content")
                            if token:
                                yield DirectStreamEvent(kind="token", text=token,
                                    generation=int(frame.get("intramind_generation", generation)))
                    event_name = "message"
                elif line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip())
            # HTTP 200 and a closed socket are not successful inference. llama.cpp
            # completion streams terminate only at an explicit [DONE] frame.
            raise RuntimeError("managed inference stream ended before [DONE]")


class _IdentityAuth(httpx.Auth):
    def __init__(self, base_url: str, service_token: str):
        self.base_url = httpx.URL(base_url)
        self.service_token = service_token

    def auth_flow(self, request):
        url = request.url
        if ((url.scheme, url.host, url.port) !=
            (self.base_url.scheme, self.base_url.host, self.base_url.port)
            or not url.path.startswith(self.base_url.path)):
            raise ValueError("direct inference request is outside its runtime binding")
        identity = _tenant.get()
        if identity is None:
            raise ValueError("verified inference identity is required")
        request.headers["Authorization"] = "Bearer " + self.service_token
        request.headers["X-Tenant-ID"] = identity
        request.headers["X-Intramind-Workload-Class"] = _workload.get()
        logical_request = _logical_request.get()
        if logical_request is not None:
            request.headers["X-Intramind-Logical-Request-ID"] = logical_request
        deadline_seconds = _deadline_seconds.get()
        if deadline_seconds is not None:
            request.headers["X-Intramind-Deadline-Seconds"] = str(min(deadline_seconds, 86400))
        yield request


class DirectLLMRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    base_url: str
    model: str
    model_profile: str = Field(pattern=r"^[A-Za-z0-9._-]{1,128}$")


class DirectRouting(BaseModel):
    """Deployment-owned routing; original model configs remain the source of identity."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    embedding_profile: str = ""
    rerank_profile: str = ""
    llm_routes: list[DirectLLMRoute] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_routes(self):
        keys = [(route.base_url.rstrip("/"), route.model) for route in self.llm_routes]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate direct inference backend/model route")
        return self

    def native_binding(self, kind: str, base_url: str, service_token: str) -> DirectBinding | None:
        if not self.enabled:
            return None
        profile = {"embedding": self.embedding_profile, "rerank": self.rerank_profile}.get(kind)
        if not profile:
            raise ValueError(f"qualified direct {kind} profile is required")
        return DirectBinding(base_url, service_token, profile)

    def llm_binding(self, config: dict, base_url: str, service_token: str) -> DirectBinding | None:
        if not self.enabled or not config.get("enabled", True):
            return None
        if config.get("provider", "openai") != "openai":
            raise ValueError("managed direct inference requires an OpenAI-compatible HTTP provider")
        endpoint = (config.get("base_url") or "").rstrip("/")
        model = config.get("model_name", config.get("model"))
        for route in self.llm_routes:
            if route.base_url.rstrip("/") == endpoint and route.model == model:
                return DirectBinding(base_url, service_token, route.model_profile)
        raise ValueError("qualified direct LLM route is required for this endpoint/model")
