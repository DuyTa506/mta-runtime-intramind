"""Contracts for foreground inference held only by runtime-api memory."""

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator

from .contracts import Contract


class DirectRequest(Contract):
    request_id: str = Field(min_length=1, max_length=200)
    tenant_id: str = Field(min_length=1, max_length=200)
    payload_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_profile: str = Field(min_length=1, max_length=120)
    model_revision: str | None = Field(default=None, min_length=1)
    expected_engine_epoch: str | None = Field(default=None, min_length=1)
    capacity_profile_id: str = Field(min_length=1, max_length=200)
    kind: Literal["llm", "embedding", "speech", "rerank"] = "llm"
    request_bound: int = Field(gt=0, strict=True)
    batch_size: int = Field(default=1, gt=0, le=256, strict=True)
    deadline: datetime
    workload_class: Literal["qa", "user_task", "background", "maintenance"] = "qa"
    logical_request_id: str | None = Field(default=None, max_length=200)
    admission_mode: Literal["wait", "try"] = "wait"
    dispatch_before: datetime | None = None
    execution_timeout_seconds: float | None = Field(default=None, gt=0, le=86400)

    @field_validator("deadline", "dispatch_before")
    @classmethod
    def aware(cls, value):
        if value is not None and value.tzinfo is None:
            raise ValueError("deadline must include timezone")
        return value


class DirectReservation(Contract):
    attempt_id: str
    request: DirectRequest
    pool_id: str
    engine_epoch: str
    owner_id: str
    generation: int = 0
