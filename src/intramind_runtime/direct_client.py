"""HTTP caller binding; it does not submit workflows or retry inference."""

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

_tenant: ContextVar[str | None] = ContextVar("direct_inference_tenant", default=None)


@contextmanager
def inference_scope(tenant_id: str | None):
    """Bind an identity verified by the application, including across asyncio.to_thread."""
    if tenant_id is not None and (
        not tenant_id or len(tenant_id) > 240 or not tenant_id.isascii()
        or any(ord(character) <= 32 or ord(character) == 127 for character in tenant_id)
    ):
        raise ValueError("invalid inference identity")
    token = _tenant.set(tenant_id)
    try:
        yield
    finally:
        _tenant.reset(token)


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
