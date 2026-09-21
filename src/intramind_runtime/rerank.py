"""Qualified native reranking bounds preserve candidate order and raw score scale."""

from typing import Annotated

from pydantic import Field

from .contracts import Contract


class RerankPayload(Contract):
    query: str = Field(strict=True)
    documents: list[Annotated[str, Field(strict=True)]] = Field(default_factory=list)
    top_k: int | None = Field(default=None, strict=True)


class RerankProfile(Contract):
    model_profile: str = Field(min_length=1)
    capacity_profile_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:-]+$")
    character_limit: int = Field(gt=0, le=1_000_000, strict=True)
    max_batch_size: int = Field(gt=0, le=256, strict=True)
    max_response_bytes: int = Field(gt=0, le=16 * 1024 * 1024, strict=True)

    def validate_payload(self, payload: dict) -> RerankPayload:
        """Count the prefix processed by the pinned native cap; never rewrite the request."""
        request = RerankPayload.model_validate(payload)
        characters, _ = self.requirement(request)
        if characters > self.character_limit:
            raise ValueError("rerank request exceeds the qualified profile")
        return request

    def requirement(self, request: RerankPayload) -> tuple[int, int]:
        processed = request.documents[:self.max_batch_size]
        return (len(request.query) + sum(map(len, processed)), len(processed)) if processed else (0, 0)

    def validate_response(self, body: dict, payload: dict):
        request = self.validate_payload(payload)
        result = _RerankResponse.model_validate(body)
        indices = [item.index for item in result.results]
        if (len(indices) != len(set(indices))
            or any(index >= min(len(request.documents), self.max_batch_size) for index in indices)):
            raise ValueError("rerank result index differs from the qualified candidate batch")
        return result


class _RerankItem(Contract):
    index: int = Field(ge=0, strict=True)
    score: float = Field(strict=True, allow_inf_nan=False)


class _RerankResponse(Contract):
    results: list[_RerankItem]
