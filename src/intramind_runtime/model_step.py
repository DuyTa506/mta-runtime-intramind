"""Plan a pure, sequential model leaf using previously committed responses.

This is an adapter for existing prompt/validation functions, not a workflow
runner. A leaf must have no I/O, side effects, clocks, randomness or parallel
tasks. Its caller checkpoints each returned request/result in a durable
workflow. Fan-out and external effects belong to explicit workflow commands.
"""

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


class StepReplayError(RuntimeError):
    """The leaf no longer matches the recorded inputs or sequential contract."""


@dataclass(frozen=True)
class ModelRecord:
    request_digest: str
    result: dict[str, Any] | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-f0-9]{64}", self.request_digest):
            raise ValueError("invalid model request digest")
        if (self.result is None) == (self.error is None):
            raise ValueError("a record must have exactly one result or terminal error")
        if self.error is not None and (not isinstance(self.error, str) or not self.error):
            raise ValueError("a terminal model error must be a nonempty string")
        if self.result is not None:
            if not isinstance(self.result, dict):
                raise ValueError("a model result must be a JSON object")
            object.__setattr__(self, "result", _json_copy(self.result))


@dataclass(frozen=True)
class ModelCall:
    payload: dict[str, Any]
    max_output_tokens: int
    request_digest: str


@dataclass(frozen=True)
class ModelStep:
    done: bool
    result: Any = None
    call: ModelCall | None = None


class _AwaitModelCall(BaseException):
    """Suspend without entering a leaf's ordinary model-error fallback."""


class ModelPort:
    """Injected model port for a single invocation of a pure leaf."""

    def __init__(self, records: Sequence[ModelRecord]):
        self._records = tuple(records)
        self._owner = asyncio.current_task()
        self._ordinal = 0
        self._pending: ModelCall | None = None
        self._violation: StepReplayError | None = None
        self._closed = False

    def reject(self, message: str) -> None:
        self._violation = StepReplayError(message)
        raise self._violation

    async def invoke(self, payload: dict[str, Any], *, max_output_tokens: int) -> dict[str, Any]:
        if self._violation:
            raise self._violation
        if self._closed or asyncio.current_task() is not self._owner:
            self.reject("model leaves must be sequential; use durable children for fan-out")
        if self._pending:
            self.reject("leaf suppressed its suspension instead of awaiting a recorded response")
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            self.reject("max_output_tokens must be a positive integer")
        if not isinstance(payload, dict):
            self.reject("model payload must be a JSON object")
        try:
            frozen = _json_copy(payload)
            serialized = json.dumps(
                [frozen, max_output_tokens],
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
        except (TypeError, ValueError) as exc:
            self.reject(f"model payload is not JSON: {type(exc).__name__}")
        digest = sha256(serialized).hexdigest()
        if self._ordinal < len(self._records):
            record = self._records[self._ordinal]
            if record.request_digest != digest:
                self.reject(f"model request changed at ordinal {self._ordinal}")
            self._ordinal += 1
            if record.error is not None:
                raise RuntimeError(record.error)
            return _json_copy(record.result)
        self._pending = ModelCall(frozen, max_output_tokens, digest)
        raise _AwaitModelCall()


async def plan_model_step(
    body: Callable[[ModelPort], Awaitable[Any]], records: Sequence[ModelRecord]
) -> ModelStep:
    """Return the next request, or a JSON result after consuming all records.

    Only terminal inference failures may appear as error records. Timeouts with
    unknown compute state must remain in the broker's reconciliation lifecycle.
    """
    port = ModelPort(records)
    try:
        try:
            result = await body(port)
        except _AwaitModelCall:
            if port._violation:
                raise port._violation
            return ModelStep(done=False, call=port._pending)
        if port._violation:
            raise port._violation
        if port._pending:
            raise StepReplayError("leaf suppressed its model suspension")
        if port._ordinal != len(records):
            raise StepReplayError("unused model records: leaf control flow changed")
        return ModelStep(done=True, result=_json_copy(result))
    finally:
        port._closed = True
