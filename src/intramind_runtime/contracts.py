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


class OperationSpec(Contract):
    operation_id: str = Field(min_length=1, max_length=240)
    root_id: str = Field(min_length=1, max_length=200)
    tenant_id: str = Field(min_length=1, max_length=200)
    payload: Artifact
    model_profile: str = Field(min_length=1, max_length=120)
    input_tokens_bound: int = Field(ge=0)
    max_output_tokens: int = Field(gt=0)
    expected_cost: int = Field(gt=0)
    max_attempts: int = Field(default=3, ge=1, le=10)
    attempt_timeout_seconds: AttemptTimeout = DEFAULT_ATTEMPT_TIMEOUT_SECONDS
    capacity_profile_id: str | None = None
    required_capabilities: frozenset[str] = frozenset()

    @field_serializer("required_capabilities")
    def ordered_capabilities(self, value):
        return sorted(value)

    @property
    def budget_bound(self) -> int:
        return self.input_tokens_bound + self.max_output_tokens


class PoolSpec(Contract):
    pool_id: str = Field(min_length=1, max_length=120)
    group_id: str = Field(min_length=1, max_length=120)
    engine_epoch: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    model_profile: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    hard_ceiling: int = Field(gt=0)
    target: int = Field(ge=0)
    context_limit: int = Field(gt=0)
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


class Reservation(Contract):
    attempt_id: str
    operation: OperationSpec
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
