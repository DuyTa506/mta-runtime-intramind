"""Bounded embedding batches through shared admission, without hidden transport retry."""

import json
from typing import Annotated, Literal

import httpx
from pydantic import Field

from .contracts import (
    AttemptTimeout,
    CancelOutcome,
    Contract,
    EmbeddingOperationSpec,
    EmbeddingResult,
    Reservation,
)
from .drivers import DriverFailure

TERMINATION_CONTRACT = "termination-v1"


class EmbeddingPayload(Contract):
    texts: list[Annotated[str, Field(strict=True)]] = Field(min_length=1, max_length=256)
    input_type: Literal["document", "query"] = "document"

    @property
    def characters(self) -> int:
        return sum(map(len, self.texts))


class EmbeddingProfile(Contract):
    model_profile: str = Field(min_length=1)
    capacity_profile_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    model_revision: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:-]+$")
    dimension: int = Field(gt=0, le=16384, strict=True)
    max_batch_size: int = Field(gt=0, le=256, strict=True)
    character_limit: int = Field(gt=0, le=1_000_000, strict=True)
    max_text_characters: int = Field(gt=0, le=1_000_000, strict=True)
    max_response_bytes: int = Field(gt=0, le=16 * 1024 * 1024, strict=True)

    def validate_payload(self, payload: dict) -> EmbeddingPayload:
        request = EmbeddingPayload.model_validate(payload)
        if (len(request.texts) > self.max_batch_size
            or request.characters > self.character_limit
            or any(len(text) > self.max_text_characters for text in request.texts)):
            raise ValueError("embedding batch exceeds the qualified profile")
        return request


class EmbeddingPrepareRequest(Contract):
    model_profile: str = Field(min_length=1)
    capacity_profile_id: str = Field(min_length=1)
    payload: dict
    attempt_timeout_seconds: AttemptTimeout


class EmbeddingPreparer:
    """Freeze a batch; splitting text or changing document/query semantics belongs to the caller."""

    def __init__(self, profile: EmbeddingProfile):
        self.profile = profile

    async def prepare(self, request: EmbeddingPrepareRequest, artifacts, tenant_id: str) -> dict:
        if (request.model_profile != self.profile.model_profile
            or request.capacity_profile_id != self.profile.capacity_profile_id):
            raise ValueError("embedding profile changed; accepted work requires its pinned profile")
        payload = self.profile.validate_payload(request.payload)
        raw = json.dumps(payload.model_dump(mode="json"), ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        artifact = await artifacts.put(tenant_id, raw)
        return {
            "payload": artifact.model_dump(mode="json"),
            "model_profile": request.model_profile,
            "capacity_profile_id": request.capacity_profile_id,
            "model_revision": self.profile.model_revision,
            "characters_bound": payload.characters,
            "texts_count": len(payload.texts),
            "expected_cost": max(1, payload.characters),
            "attempt_timeout_seconds": request.attempt_timeout_seconds,
        }


class _EmbeddingResponse(Contract):
    embeddings: list[list[Annotated[float, Field(strict=True, allow_inf_nan=False)]]]
    dimension: int = Field(gt=0, strict=True)
    model: str


class ServingEmbeddingDriver:
    """Accept CPU vector results only from the pinned model and an echoed termination contract."""

    def __init__(self, base_url: str, profile: EmbeddingProfile, *, api_key: str | None = None,
                 client: httpx.AsyncClient | None = None):
        self.profile = profile
        self.client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=httpx.Timeout(None, connect=10),
            transport=httpx.AsyncHTTPTransport(retries=0), follow_redirects=False,
        )

    async def execute(self, reservation: Reservation, payload: dict) -> EmbeddingResult:
        spec = reservation.operation
        try:
            request = self.profile.validate_payload(payload)
            if (not isinstance(spec, EmbeddingOperationSpec)
                or spec.characters_bound != request.characters
                or spec.texts_count != len(request.texts)
                or spec.model_profile != self.profile.model_profile
                or spec.capacity_profile_id != self.profile.capacity_profile_id
                or spec.model_revision != self.profile.model_revision
                or reservation.model_revision != self.profile.model_revision):
                raise ValueError("embedding reservation differs from accepted payload/profile")
        except ValueError as exc:
            raise DriverFailure("invalid_embedding_payload", not_sent=True) from exc
        state = None
        try:
            async with self.client.stream("POST", "api/v1/embed", json=request.model_dump(mode="json"),
                headers={"X-Intramind-Attempt-ID": reservation.attempt_id,
                         "X-Intramind-Expected-Revision": self.profile.model_revision}) as response:
                if (response.headers.get("X-Intramind-Embedding-Contract") == TERMINATION_CONTRACT
                    and response.headers.get("X-Intramind-Attempt-ID") == reservation.attempt_id):
                    state = response.headers.get("X-Intramind-Compute-State")
                if state not in {"not_started", "terminated"}:
                    raise DriverFailure("embedding_response_unconfirmed")
                if response.status_code != 200:
                    raise DriverFailure(f"embedding_backend_{response.status_code}",
                        not_sent=state == "not_started", finished=state == "terminated",
                        retry=state == "not_started" and response.status_code == 429)
                if state != "terminated":
                    raise DriverFailure("embedding_success_without_termination")
                if response.headers.get("X-Intramind-Model-Revision") != self.profile.model_revision:
                    raise DriverFailure("embedding_model_revision_changed", finished=True)
                pieces, size = [], 0
                async for piece in response.aiter_bytes():
                    size += len(piece)
                    if size > self.profile.max_response_bytes:
                        raise DriverFailure("embedding_response_exceeds_profile", finished=True)
                    pieces.append(piece)
                try:
                    result = _EmbeddingResponse.model_validate_json(b"".join(pieces))
                    if (response.headers.get("Content-Type", "").split(";")[0] != "application/json"
                        or result.model != self.profile.model
                        or result.dimension != self.profile.dimension
                        or len(result.embeddings) != len(request.texts)
                        or any(len(vector) != result.dimension for vector in result.embeddings)):
                        raise ValueError("embedding output differs from qualified model/batch/dimension")
                except ValueError as exc:
                    raise DriverFailure("embedding_invalid_vectors", finished=True) from exc
                return EmbeddingResult(body=result.model_dump(mode="json"), characters=request.characters)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            raise DriverFailure(type(exc).__name__, not_sent=True, retry=True) from exc
        except httpx.HTTPError as exc:
            raise DriverFailure(type(exc).__name__, finished=state == "terminated",
                                retry=state == "terminated") from exc

    async def cancel(self, reservation: Reservation) -> CancelOutcome:
        return CancelOutcome.UNSUPPORTED

    async def close(self):
        await self.client.aclose()
