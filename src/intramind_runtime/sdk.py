"""Feature-facing facade. Only this adapter imports Temporal primitives."""

import asyncio
import json
import math
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Any, NoReturn

from temporalio import activity, workflow
from temporalio.common import RetryPolicy, VersioningBehavior
from temporalio.exceptions import ActivityError, ApplicationError, CancelledError
from temporalio.exceptions import TimeoutError as ActivityTimeoutError

with workflow.unsafe.imports_passed_through():
    from .contracts import Artifact, EmbeddingOperationSpec, OperationSpec, SpeechOperationSpec


class OperationDeadlineExceeded(TimeoutError):
    """The operation has expired; backend cleanup and accounting may still be pending."""


_WRAPPER_TYPES = frozenset({"ActivityError", "ChildWorkflowError"})


def exception_chain(exc: BaseException):
    """Yield an exception and each explicit cause once."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__


def is_cancellation(exc: BaseException) -> bool:
    """True when a workflow, activity or child was cancelled rather than failed."""
    return any(
        isinstance(item, (CancelledError, asyncio.CancelledError)) for item in exception_chain(exc)
    )


def terminal_reason(exc: BaseException) -> str:
    """Public failure reason, preferring the ledger cause over a Temporal wrapper."""
    for item in reversed(list(exception_chain(exc))):
        if isinstance(item, ApplicationError):
            if item.type == "OperationFailed" and item.message and item.message != "feature failed":
                return item.message
            if item.message == "feature failed":
                continue
            if item.type and item.type not in _WRAPPER_TYPES:
                return item.type
            continue
        name = type(item).__name__
        if name not in _WRAPPER_TYPES:
            return name
    return type(exc).__name__


@dataclass(frozen=True)
class TaskPolicy:
    max_iterations: int
    deadline_seconds: int = 86400
    child_window: int = 32
    max_rollovers: int = 32

    def __post_init__(self):
        if (
            self.max_iterations <= 0
            or self.deadline_seconds <= 0
            or not 1 <= self.child_window <= 64
            or not 0 <= self.max_rollovers <= 1000
        ):
            raise ValueError("task policy must be finite and bounded")


class TaskContext:
    def __init__(self, envelope: dict, policy: TaskPolicy):
        self.root_id = envelope["root_id"]
        self.tenant_id = envelope["tenant_id"]
        self.configuration = (
            Artifact.model_validate(envelope["configuration"]).model_dump(mode="json")
            if envelope.get("configuration")
            else None
        )
        child_path = envelope.get("child_path", [])
        if not isinstance(child_path, (list, tuple)):
            raise ValueError("child path must be a sequence of identity components")
        self.path = tuple(child_path)
        if len(self.path) > 64 or not all(isinstance(part, str) and part for part in self.path):
            raise ValueError("child path must contain at most 64 nonempty identity components")
        self.deadline = datetime.fromisoformat(envelope["deadline"])
        if self.deadline.tzinfo is None:
            raise ValueError("workflow deadline must include timezone")
        self.policy = policy
        self.control_queue = envelope["control_queue"]
        self.task_queue = workflow.info().task_queue
        self.started_at = datetime.fromisoformat(
            envelope.get("started_at", workflow.info().start_time.isoformat())
        )
        if self.started_at.tzinfo is None:
            raise ValueError("workflow start time must include timezone")
        self.deadline = min(
            self.deadline, self.started_at + timedelta(seconds=policy.deadline_seconds)
        )
        self.rollovers = envelope.get("rollovers", 0)
        if type(self.rollovers) is not int or not 0 <= self.rollovers <= policy.max_rollovers:
            raise ValueError("invalid workflow rollover count")
        self._active_commands = 0
        self._events: dict[str, object] = {}

    def key(self, key: str) -> str:
        if not isinstance(key, str) or not key or len(key) > 1024:
            raise ValueError("stable operation key required")
        identity = json.dumps(
            [self.root_id, self.path, key], ensure_ascii=False, separators=(",", ":")
        )
        return sha256(identity.encode()).hexdigest()

    @contextmanager
    def _command(self) -> Iterator[None]:
        self._active_commands += 1
        try:
            yield
        finally:
            self._active_commands -= 1

    def remaining(self) -> timedelta:
        duration = self.deadline - workflow.now()
        if duration.total_seconds() <= 0:
            raise ValueError("workflow deadline exceeded")
        return duration

    async def activity(
        self,
        name: str,
        inputs: dict,
        *,
        key: str,
        timeout_seconds: int = 300,
        max_attempts: int = 3,
        task_queue: str | None = None,
        deadline: datetime | None = None,
        heartbeat_timeout_seconds: int | None = None,
    ) -> Any:
        if timeout_seconds <= 0 or not 1 <= max_attempts <= 100:
            raise ValueError("activity timeout and retry policy must be finite and positive")
        if heartbeat_timeout_seconds is not None and heartbeat_timeout_seconds <= 0:
            raise ValueError("activity timeout and retry policy must be finite and positive")
        remaining = self.remaining()
        timeout = timedelta(seconds=timeout_seconds)
        if deadline is not None:
            if deadline.tzinfo is None:
                raise ValueError("activity deadline must include timezone")
            remaining = min(remaining, deadline - workflow.now())
            if remaining.total_seconds() <= 0:
                raise OperationDeadlineExceeded("activity deadline exceeded")
            timeout = min(timeout, remaining)
        heartbeat = (
            {"heartbeat_timeout": timedelta(seconds=heartbeat_timeout_seconds)}
            if heartbeat_timeout_seconds is not None
            else {}
        )
        with self._command():
            try:
                return await workflow.execute_activity(
                    name,
                    inputs,
                    activity_id=self.key(key),
                    task_queue=task_queue,
                    start_to_close_timeout=timeout,
                    schedule_to_close_timeout=remaining,
                    retry_policy=RetryPolicy(maximum_attempts=max_attempts),
                    **heartbeat,
                )
            except ActivityError as exc:
                if (
                    deadline is not None
                    and deadline <= workflow.now()
                    and isinstance(exc.cause, ActivityTimeoutError)
                ):
                    raise OperationDeadlineExceeded("activity deadline exceeded") from exc
                raise

    async def llm(
        self,
        *,
        key: str,
        payload: dict,
        model_profile: str,
        input_tokens_bound: int,
        max_output_tokens: int,
        expected_cost: int,
        capacity_profile_id: str | None = None,
        attempt_timeout_seconds: float | None = None,
        deadline: datetime | None = None,
    ) -> dict[str, Any]:
        spec = OperationSpec(
            operation_id=self.key(key),
            root_id=self.root_id,
            tenant_id=self.tenant_id,
            payload=Artifact.model_validate(payload),
            model_profile=model_profile,
            input_tokens_bound=input_tokens_bound,
            max_output_tokens=max_output_tokens,
            expected_cost=expected_cost,
            capacity_profile_id=capacity_profile_id,
            deadline=deadline,
            **(
                {"attempt_timeout_seconds": attempt_timeout_seconds}
                if attempt_timeout_seconds is not None
                else {}
            ),
        )
        payload = spec.model_dump(mode="json")
        if attempt_timeout_seconds is None:
            # Existing histories omitted this policy field; keep that command shape.
            payload.pop("attempt_timeout_seconds")
        if deadline is None:
            payload.pop("deadline")
        return await self._inference("runtime.submit_or_attach_llm", payload)

    async def speech(
        self,
        *,
        key: str,
        payload: dict,
        model_profile: str,
        capacity_profile_id: str,
        characters_bound: int,
        expected_cost: int,
        attempt_timeout_seconds: float,
        deadline: datetime | None = None,
    ) -> dict[str, Any]:
        """A speech operation inherits root identity and waits without holding a worker slot."""
        spec = SpeechOperationSpec(
            operation_id=self.key(key),
            root_id=self.root_id,
            tenant_id=self.tenant_id,
            payload=Artifact.model_validate(payload),
            model_profile=model_profile,
            capacity_profile_id=capacity_profile_id,
            characters_bound=characters_bound,
            expected_cost=expected_cost,
            attempt_timeout_seconds=attempt_timeout_seconds,
            deadline=deadline,
        )
        payload = spec.model_dump(mode="json")
        if deadline is None:
            payload.pop("deadline")
        return await self._inference("runtime.submit_or_attach_speech", payload)

    async def embedding(
        self,
        *,
        key: str,
        payload: dict,
        model_profile: str,
        capacity_profile_id: str,
        model_revision: str,
        characters_bound: int,
        texts_count: int,
        expected_cost: int,
        attempt_timeout_seconds: float,
        deadline: datetime | None = None,
    ) -> dict[str, Any]:
        """Wait durably for one embedding batch under the same root and shared admission."""
        spec = EmbeddingOperationSpec(
            operation_id=self.key(key),
            root_id=self.root_id,
            tenant_id=self.tenant_id,
            payload=Artifact.model_validate(payload),
            model_profile=model_profile,
            capacity_profile_id=capacity_profile_id,
            characters_bound=characters_bound,
            texts_count=texts_count,
            expected_cost=expected_cost,
            model_revision=model_revision,
            attempt_timeout_seconds=attempt_timeout_seconds,
            deadline=deadline,
        )
        command = spec.model_dump(mode="json")
        if deadline is None:
            command.pop("deadline")
        return await self._inference("runtime.submit_or_attach_embedding", command)

    async def embedding_outcome(self, **request: Any) -> dict[str, Any]:
        """Allow an accepted native fallback only after a terminal backend failure."""
        try:
            return {"result": await self.embedding(**request)}
        except ActivityError as exc:
            cause = exc.cause
            if (
                isinstance(cause, ApplicationError)
                and cause.type == "OperationFailed"
                and cause.message
                in {
                    "embedding_backend_400",
                    "embedding_backend_500",
                    "embedding_backend_502",
                    "embedding_backend_503",
                    "embedding_backend_504",
                    "max_attempts",
                }
            ):
                return {"error": cause.message}
            raise

    async def speech_outcome(self, **request: Any) -> dict[str, Any]:
        """Permit the feature's voice fallback only after a terminal transport failure."""
        try:
            return {"result": await self.speech(**request)}
        except ActivityError as exc:
            cause = exc.cause
            if (
                isinstance(cause, ApplicationError)
                and cause.type == "OperationFailed"
                and cause.message
                in {
                    "speech_backend_400",
                    "speech_backend_500",
                    "speech_backend_503",
                    "max_attempts",
                }
            ):
                return {"error": cause.message}
            raise

    async def _inference(self, name: str, payload: dict) -> dict:
        deadline = datetime.fromisoformat(payload["deadline"]) if payload.get("deadline") else None
        remaining = self.remaining()
        if deadline is not None:
            remaining = min(remaining, deadline - workflow.now())
            if remaining.total_seconds() <= 0:
                raise OperationDeadlineExceeded("operation deadline exceeded")
        with self._command():
            try:
                return await workflow.execute_activity(
                    name,
                    payload,
                    activity_id=payload["operation_id"],
                    task_queue=self.control_queue,
                    start_to_close_timeout=remaining,
                    schedule_to_close_timeout=remaining,
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(seconds=2),
                        maximum_interval=timedelta(seconds=30),
                    ),
                )
            except ActivityError as exc:
                cause = exc.cause
                if (
                    isinstance(cause, ApplicationError)
                    and cause.type == "OperationFailed"
                    and cause.message == "operation_deadline_exceeded"
                ):
                    raise OperationDeadlineExceeded(cause.message) from exc
                if (
                    deadline is not None
                    and deadline <= workflow.now()
                    and isinstance(cause, ActivityTimeoutError)
                ):
                    raise OperationDeadlineExceeded("operation deadline exceeded") from exc
                raise

    async def run_child(
        self, *, task_type: str, task_queue: str, key: str, inputs: dict
    ) -> dict[str, Any]:
        if len(self.path) >= 64:
            raise ValueError("child workflow depth limit exceeded")
        with self._command():
            return await workflow.execute_child_workflow(
                task_type,
                {
                    "root_id": self.root_id,
                    "tenant_id": self.tenant_id,
                    "child_path": [*self.path, key],
                    "deadline": self.deadline.isoformat(),
                    "control_queue": self.control_queue,
                    "input": inputs,
                    **({"configuration": self.configuration} if self.configuration else {}),
                },
                id=self.key(key),
                task_queue=task_queue,
                execution_timeout=self.remaining(),
            )

    async def llm_outcome(self, **request: Any) -> dict[str, Any]:
        """Allow feature fallback only for a broker-confirmed terminal failure.

        Cancellation, deadlines, rejected budgets and ambiguous transport errors
        must propagate instead of becoming a successful fallback result.
        """
        try:
            return {"result": await self.llm(**request)}
        except ActivityError as exc:
            cause = exc.cause
            if (
                isinstance(cause, ApplicationError)
                and cause.type == "OperationFailed"
                and cause.message
                in {
                    "backend_rejected_400",
                    "backend_rejected_422",
                    "invalid_response",
                    "max_attempts",
                }
            ):
                return {"error": "OperationFailed"}
            raise

    async def model_step(
        self,
        *,
        key: str,
        planner: str,
        inputs: dict,
        max_model_calls: int = 4,
    ) -> dict[str, Any]:
        """Checkpoint a registered pure model leaf, including schema repair.

        The planning activity returns references, never raw prompts/results.
        Each inference call has its own operation identity and budget charge.
        """
        if type(max_model_calls) is not int or not 1 <= max_model_calls <= 64:
            raise ValueError("model step requires a finite call limit between 1 and 64")
        records: list[dict] = []
        with self._command():
            for ordinal in range(max_model_calls + 1):
                step = await self.activity(
                    planner,
                    {"tenant_id": self.tenant_id, "input": inputs, "records": records},
                    key=f"{key}/plan/{ordinal}",
                )
                if step["done"]:
                    return Artifact.model_validate(step["result"]).model_dump(mode="json")
                if ordinal == max_model_calls:
                    raise ValueError("model step call budget exhausted")
                outcome = await self.llm_outcome(key=f"{key}/call/{ordinal}", **step["request"])
                records.append({"request_digest": step["request_digest"], **outcome})
        raise AssertionError("model step did not return or exhaust its bound")

    async def map_children(
        self,
        *,
        task_type: str,
        task_queue: str,
        items: list[dict],
        item_key: str,
        key: str,
        window: int | None = None,
    ) -> list[dict[str, Any]]:
        if window is not None and (type(window) is not int or window <= 0):
            raise ValueError("child window must be a positive integer")
        limit = (
            min(window, self.policy.child_window)
            if window is not None
            else self.policy.child_window
        )
        identities = [str(item[item_key]) for item in items]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate child identity")
        result = []
        # Only a window of child starts is materialized in history at a time.
        for start in range(0, len(items), limit):
            result.extend(
                await asyncio.gather(
                    *(
                        self.run_child(
                            task_type=task_type,
                            task_queue=task_queue,
                            key=json.dumps([key, identities[i]], ensure_ascii=False),
                            inputs=items[i],
                        )
                        for i in range(start, min(start + limit, len(items)))
                    )
                )
            )
        return result

    async def sleep(self, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("sleep duration must be finite and positive")
        with self._command():
            await workflow.sleep(min(seconds, self.remaining().total_seconds()))
        self.remaining()

    async def wait_event(self, key: str) -> object:
        with self._command():
            await workflow.wait_condition(lambda: key in self._events, timeout=self.remaining())
        return self._events.pop(key)

    def finish_partial(self, result: dict, *, reason: str) -> dict[str, Any]:
        if not reason:
            raise ValueError("partial result requires a reason")
        return {"run_status": "PARTIAL", "terminal_reason": reason, "result": result}

    def elapsed_seconds(self) -> float:
        return (workflow.now() - self.started_at).total_seconds()

    def rollover_suggested(self) -> bool:
        return workflow.info().is_continue_as_new_suggested()

    def continue_as_new(self, checkpoint: dict[str, Any]) -> NoReturn:
        """Roll history at a completed boundary using a small artifact/cursor manifest.

        Operation keys, root budget, deadline and pending signals remain stable.
        The handler must checkpoint completed items before requesting rollover.
        """
        if self._active_commands:
            raise ValueError("cannot roll over while activities, children or waits are active")
        if self.rollovers >= self.policy.max_rollovers:
            raise ValueError("workflow rollover budget exhausted")
        self.remaining()
        envelope = {
            "root_id": self.root_id,
            "tenant_id": self.tenant_id,
            "child_path": list(self.path),
            "deadline": self.deadline.isoformat(),
            "started_at": self.started_at.isoformat(),
            "control_queue": self.control_queue,
            "rollovers": self.rollovers + 1,
            "pending_events": self._events,
            "input": checkpoint,
            **({"configuration": self.configuration} if self.configuration else {}),
        }
        if len(json.dumps(envelope, ensure_ascii=False).encode()) > 64 * 1024:
            raise ValueError("rollover manifest exceeds 64 KiB; persist large values as artifacts")
        workflow.continue_as_new(envelope)


def durable_task(*, name: str, version: int, policy: TaskPolicy):
    """Declare a replayable handler using only TaskContext durable operations."""

    def decorate(handler):
        def initialize(self):
            self.ctx = None
            self.pending_events = {}

        async def run(self, envelope: dict):
            self.ctx = TaskContext(envelope, policy)
            self.pending_events = envelope.get("pending_events", {}) | self.pending_events
            self.ctx._events = self.pending_events
            try:
                result = await handler(self.ctx, envelope["input"])
                terminal = (
                    "PARTIAL"
                    if isinstance(result, dict) and result.get("run_status") == "PARTIAL"
                    else "SUCCEEDED"
                )
                artifact = result["result"] if terminal == "PARTIAL" else result
                artifact = Artifact.model_validate(artifact).model_dump(mode="json")
            except Exception as exc:
                cancelled = is_cancellation(exc)
                reason = "user_cancelled" if cancelled else terminal_reason(exc)
                if not self.ctx.path:
                    await workflow.execute_activity(
                        "runtime.finish_run",
                        {
                            "root_id": self.ctx.root_id,
                            "tenant_id": self.ctx.tenant_id,
                            "state": "CANCELLED" if cancelled else "FAILED",
                            "reason": reason,
                        },
                        task_queue=self.ctx.control_queue,
                        start_to_close_timeout=timedelta(seconds=30),
                    )
                if cancelled:
                    raise
                raise ApplicationError("feature failed", type=reason, non_retryable=True) from exc
            if not self.ctx.path:
                await workflow.execute_activity(
                    "runtime.finish_run",
                    {
                        "root_id": self.ctx.root_id,
                        "tenant_id": self.ctx.tenant_id,
                        "state": terminal,
                        "result": artifact,
                        "reason": result.get("terminal_reason") if terminal == "PARTIAL" else None,
                    },
                    task_queue=self.ctx.control_queue,
                    start_to_close_timeout=timedelta(seconds=30),
                )
            return result

        def event(self, payload: dict):
            self.pending_events[payload["key"]] = payload["value"]

        cls_name = handler.__name__ + "Workflow"
        run.__qualname__ = f"{cls_name}.run"
        run.__module__ = handler.__module__
        event.__qualname__ = f"{cls_name}.event"
        event.__module__ = handler.__module__
        cls = type(
            cls_name,
            (),
            {
                "__module__": handler.__module__,
                "__init__": initialize,
                "run": workflow.run(run),
                "event": workflow.signal(event),
            },
        )
        cls = workflow.defn(
            name=f"{name}/v{version}", versioning_behavior=VersioningBehavior.PINNED
        )(cls)
        setattr(sys.modules[handler.__module__], cls_name, cls)
        return cls

    return decorate


def durable_activity(*, name: str):
    """Register a feature's bounded I/O/CPU function with the selected runtime."""
    return activity.defn(name=name)
