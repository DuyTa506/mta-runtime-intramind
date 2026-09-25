"""Internal one-shot admission and dispatch evidence, independent of model payloads."""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite

import httpx

CAPABILITY = "warehouse-admission-v1"
_scope = ContextVar("runtime_admission", default=None)

# Signals only: no inference permit or budget is stored in Redis. The check
# never refreshes last_busy, so polling and warehouse calls cannot extend it.
ACTIVITY_CHECK = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
if redis.call('EXISTS', KEYS[2]) == 0 or redis.call('EXISTS', KEYS[3]) == 0
   or redis.call('EXISTS', KEYS[5]) == 0 then return 2 end
if redis.call('ZCARD', KEYS[1]) > 0 then return 3 end
local last = tonumber(redis.call('GET', KEYS[4]))
if last == nil then return 2 end
if now - last < tonumber(ARGV[1]) then return 4 end
return 1
"""
ACTIVITY_REASONS = {1: None, 2: "activity_unconfirmed", 3: "foreground_active", 4: "foreground_cooldown"}


class AdmissionDeferred(Exception):
    """The runtime proved this logical inference request was never sent."""

    def __init__(self, reason, evidence=None):
        super().__init__(reason)
        self.reason = reason
        self.evidence = {**(evidence or {}), "compute_state": "not_sent", "reason": reason}


class InferenceOutcomeError(RuntimeError):
    """A failed or uncertain send must never be treated as admission deferral."""

    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


@dataclass
class AdmissionScope:
    mode: str
    dispatch_before: datetime | None
    execution_timeout_seconds: float | None
    on_event: object = None
    before_send: object = None
    evidence: dict = field(default_factory=dict)
    error: Exception | None = None

    async def observe(self, frame):
        self.evidence.update({key: frame[key] for key in (
            "attempt_id", "logical_request_id", "generation", "compute_state") if key in frame})
        if frame.get("type") == "admission_deferred" and frame.get("compute_state") == "not_sent":
            self.error = AdmissionDeferred(frame.get("reason", "inference_capacity"), frame)
        elif frame.get("type") != "started":
            self.error = InferenceOutcomeError(frame.get("message", "inference failed"),
                                                dict(self.evidence))
        if self.on_event and self.evidence.get("compute_state") in {"sent", "unknown", "terminated"}:
            await self.on_event(dict(self.evidence))
        if self.error:
            raise self.error


@contextmanager
def admission_scope(*, mode="wait", dispatch_before=None, execution_timeout_seconds=None,
                    on_event=None, before_send=None):
    """Apply trusted transport policy without changing prompts or provider validation."""
    if mode not in {"wait", "try"}:
        raise ValueError("invalid admission mode")
    if dispatch_before is not None and dispatch_before.tzinfo is None:
        raise ValueError("dispatch cutoff must be timezone aware")
    if execution_timeout_seconds is not None and (
        isinstance(execution_timeout_seconds, bool) or not isfinite(execution_timeout_seconds)
        or not 0 < execution_timeout_seconds <= 86400
    ):
        raise ValueError("invalid execution timeout")
    scope = AdmissionScope(mode, dispatch_before, execution_timeout_seconds, on_event, before_send)
    token = _scope.set(scope)
    try:
        yield scope
    finally:
        _scope.reset(token)


def apply_headers(request):
    scope = _scope.get()
    if scope is None:
        return
    request.headers["X-Intramind-Admission-Mode"] = scope.mode
    if scope.dispatch_before is not None:
        from datetime import UTC

        request.headers["X-Intramind-Dispatch-Before"] = scope.dispatch_before.astimezone(UTC).isoformat()
    if scope.execution_timeout_seconds is not None:
        request.headers["X-Intramind-Step-Timeout-Seconds"] = str(scope.execution_timeout_seconds)


class _EvidenceStream(httpx.AsyncByteStream):
    def __init__(self, stream, scope):
        self.stream, self.scope = stream, scope

    async def __aiter__(self):
        pending = b""
        async for chunk in self.stream:
            pending += chunk
            while b"\n\n" in pending:
                raw, pending = pending.split(b"\n\n", 1)
                lines = raw.splitlines()
                event = next((line[6:].strip() for line in lines if line.startswith(b"event:")), b"")
                if event in {b"intramind.control", b"intramind.error"}:
                    data = b"\n".join(line[5:].lstrip() for line in lines if line.startswith(b"data:"))
                    frame = json.loads(data)
                    if event == b"intramind.error" or frame.get("type") == "started":
                        await self.scope.observe(frame)
                    continue  # These internal envelopes are not OpenAI completion chunks.
                if any(line.startswith(b"data:") and line[5:].strip() == b"[DONE]"
                       for line in lines) and self.scope.evidence.get("attempt_id"):
                    self.scope.evidence["compute_state"] = "terminated"
                yield raw + b"\n\n"
            if len(pending) > 1024 * 1024:
                raise ValueError("admission evidence frame exceeds bound")
        if pending:
            yield pending

    async def aclose(self):
        await self.stream.aclose()


async def observe_response(response):
    """Preserve OpenAI frames; observe only runtime-owned dispatch controls."""
    scope = _scope.get()
    if scope is None:
        return
    if response.status_code == 409:
        await response.aread()
        frame = response.json().get("error", {})
        if frame.get("type") == "admission_deferred":
            await scope.observe(frame)
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        response.stream = _EvidenceStream(response.stream, scope)


async def before_request(request):
    scope = _scope.get()
    if scope is not None and scope.before_send:
        try:
            await scope.before_send()
        except AdmissionDeferred as exc:
            scope.error = exc
            raise


async def verify_capability(base_url, service_token):
    """Fail startup on an older runtime rather than silently waiting for capacity."""
    async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
        response = await client.get(base_url.rstrip("/") + "/v1/capabilities",
            headers={"Authorization": "Bearer " + service_token})
        response.raise_for_status()
        if CAPABILITY not in response.json().get("capabilities", []):
            raise RuntimeError("runtime lacks " + CAPABILITY)
