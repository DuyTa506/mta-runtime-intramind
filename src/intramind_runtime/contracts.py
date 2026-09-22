"""Versioned wire contracts. Policy fields are supplied by trusted services."""

from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

DEFAULT_ATTEMPT_TIMEOUT_SECONDS = 1800.0
AttemptTimeout = Annotated[float, Field(gt=0, le=86400, allow_inf_nan=False, strict=True)]


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OperationState(StrEnum):
    READY = "READY"
    EXECUTING = "EXECUTING"
    RECONCILING = "RECONCILING"
    RETRY_WAIT = "RETRY_WAIT"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Artifact(Contract):
    key: str = Field(min_length=1, max_length=1024)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size: int = Field(ge=0)
    content_type: str = "application/json"


class RootSpec(Contract):
    root_id: str = Field(min_length=1, max_length=200)
    tenant_id: str = Field(min_length=1, max_length=200)
    deadline: datetime
    budget_limit: int = Field(gt=0)
    resource_budgets: dict[
        Literal["speech_characters", "embedding_characters"], Annotated[int, Field(gt=0, strict=True)]
    ] = Field(default_factory=dict)
    max_operations: int = Field(default=10_000, gt=0, le=100_000)
    max_pending: int = Field(default=64, gt=0, le=1024)
    max_attempts: int = Field(default=30_000, gt=0)
    priority: Literal["interactive", "background"] = "background"

    @field_validator("deadline")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("deadline must include timezone")
        return value


class _OperationSpec(Contract):
    operation_id: str = Field(min_length=1, max_length=240)
    root_id: str = Field(min_length=1, max_length=200)
    tenant_id: str = Field(min_length=1, max_length=200)
    payload: Artifact
    model_profile: str = Field(min_length=1, max_length=120)
    expected_cost: int = Field(gt=0)
    max_attempts: int = Field(default=3, ge=1, le=10)
    attempt_timeout_seconds: AttemptTimeout = DEFAULT_ATTEMPT_TIMEOUT_SECONDS
    deadline: datetime | None = None
    capacity_profile_id: str | None = None
    required_capabilities: frozenset[str] = frozenset()

    @field_validator("deadline")
    @classmethod
    def aware_deadline(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("operation deadline must include timezone")
        return value

    @field_serializer("required_capabilities")
    def ordered_capabilities(self, value):
        return sorted(value)


class OperationSpec(_OperationSpec):
    input_tokens_bound: int = Field(ge=0)
    max_output_tokens: int = Field(gt=0)

    @property
    def kind(self) -> str:
        return "llm"

    @property
    def budget_unit(self) -> str:
        return "tokens"

    @property
    def budget_bound(self) -> int:
        return self.input_tokens_bound + self.max_output_tokens


class SpeechOperationSpec(_OperationSpec):
    kind: Literal["speech"] = "speech"
    characters_bound: int = Field(gt=0, strict=True)

    @property
    def budget_unit(self) -> str:
        return "speech_characters"

    @property
    def budget_bound(self) -> int:
        return self.characters_bound


class EmbeddingOperationSpec(_OperationSpec):
    kind: Literal["embedding"] = "embedding"
    model_revision: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:-]+$")
    characters_bound: int = Field(ge=0, strict=True)
    texts_count: int = Field(gt=0, strict=True)

    @property
    def budget_unit(self) -> str:
        return "embedding_characters"

    @property
    def budget_bound(self) -> int:
        # An empty text still performs inference; each batch reserves at least one unit.
        return max(1, self.characters_bound)


InferenceOperation = OperationSpec | SpeechOperationSpec | EmbeddingOperationSpec


def parse_operation(value: dict) -> InferenceOperation:
    """Keep legacy completion records valid without inventing token fields for speech."""
    model = {"speech": SpeechOperationSpec, "embedding": EmbeddingOperationSpec}.get(
        value.get("kind"), OperationSpec)
    return model.model_validate(value)


class _PoolSpec(Contract):
    pool_id: str = Field(min_length=1, max_length=120)
    group_id: str = Field(min_length=1, max_length=120)
    engine_epoch: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    model_profile: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    hard_ceiling: int = Field(gt=0)
    target: int = Field(ge=0)
    valid_until: datetime
    capabilities: frozenset[str] = frozenset()

    @field_serializer("capabilities")
    def ordered_capabilities(self, value):
        return sorted(value)

    @model_validator(mode="after")
    def check(self):
        if self.target > self.hard_ceiling:
            raise ValueError("target exceeds hard ceiling")
        if self.valid_until.tzinfo is None:
            raise ValueError("valid_until must include timezone")
        return self


class PoolSpec(_PoolSpec):
    context_limit: int = Field(gt=0)

    @property
    def kind(self) -> str:
        return "llm"

    @property
    def request_limit(self) -> int:
        return self.context_limit


class SpeechPoolSpec(_PoolSpec):
    kind: Literal["speech"] = "speech"
    character_limit: int = Field(gt=0, strict=True)

    @property
    def request_limit(self) -> int:
        return self.character_limit


class EmbeddingPoolSpec(_PoolSpec):
    kind: Literal["embedding"] = "embedding"
    character_limit: int = Field(gt=0, strict=True)
    max_batch_size: int = Field(gt=0, le=256, strict=True)

    @property
    def request_limit(self) -> int:
        return self.character_limit


class RerankPoolSpec(_PoolSpec):
    kind: Literal["rerank"] = "rerank"
    character_limit: int = Field(gt=0, strict=True)
    max_batch_size: int = Field(gt=0, le=256, strict=True)

    @property
    def request_limit(self) -> int:
        return self.character_limit


InferencePool = PoolSpec | SpeechPoolSpec | EmbeddingPoolSpec | RerankPoolSpec


def parse_pool(value: dict) -> InferencePool:
    """Read the resource class explicitly; old pool records remain completion pools."""
    model = {"speech": SpeechPoolSpec, "embedding": EmbeddingPoolSpec, "rerank": RerankPoolSpec}.get(
        value.get("kind"), PoolSpec)
    return model.model_validate(value)


class Reservation(Contract):
    attempt_id: str
    operation: InferenceOperation
    pool_id: str
    engine_epoch: str
    model_revision: str
    owner_id: str
    lease_epoch: int
    attempt_deadline: datetime

    @field_validator("attempt_deadline")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("attempt deadline must include timezone")
        return value


class EngineResult(Contract):
    body: dict[str, Any]
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class SpeechResult(Contract):
    body: dict[str, Any]
    audio: bytes = Field(exclude=True)
    characters: int = Field(gt=0, strict=True)


class EmbeddingResult(Contract):
    body: dict[str, Any]
    characters: int = Field(ge=0, strict=True)


class CancelOutcome(StrEnum):
    CONFIRMED_STOPPED = "confirmed_stopped"
    REQUESTED = "requested"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class RuntimeConflict(Exception):
    """A stable identity was reused with different content or ownership."""


class AdmissionDenied(Exception):
    """A finite budget, capacity or policy bound prevents admission."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class NotFound(Exception):
    """No object in the authenticated tenant's scope."""
