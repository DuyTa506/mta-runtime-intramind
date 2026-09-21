"""Stream foreground HTTP while an owned producer drains admitted inference."""

import asyncio
import logging
from collections import deque
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import uuid4

import httpx
from fastapi.responses import StreamingResponse

from .contracts import AdmissionDenied
from .direct import DirectAdmissions, DirectRequest
from .store import encode

logger = logging.getLogger(__name__)


class _Channel:
    def __init__(self):
        self.ready = asyncio.Condition()
        self.chunks = deque()
        self.detached = self.ended = False
        self.error = None

    async def put(self, chunk):
        async with self.ready:
            await self.ready.wait_for(lambda: self.detached or len(self.chunks) < 8)
            if not self.detached:
                self.chunks.append(chunk)
                self.ready.notify_all()

    async def detach(self):
        async with self.ready:
            self.detached = True
            self.chunks.clear()
            self.ready.notify_all()

    async def finish(self, error=None):
        async with self.ready:
            self.ended, self.error = True, error
            self.ready.notify_all()

    async def stream(self):
        try:
            while True:
                async with self.ready:
                    await self.ready.wait_for(lambda: self.chunks or self.ended)
                    if not self.chunks:
                        if self.error:
                            raise RuntimeError(self.error)
                        return
                    chunk = self.chunks.popleft()
                    self.ready.notify_all()
                yield chunk
        finally:
            await self.detach()


class _DirectResponse(StreamingResponse):
    def __init__(self, channel, **kwargs):
        super().__init__(channel.stream(), **kwargs)
        self.channel = channel

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.channel.detach()


class DirectProxy:
    """No Temporal client, inference retry, or request/response artifact persistence."""

    def __init__(self, store, *, client, pool, model=None, timeout_seconds=1800, max_response_bytes=16*1024*1024):
        self.store, self.client, self.pool = store, client, pool
        self.model = model or pool.model_revision
        self.timeout_seconds, self.max_response_bytes = timeout_seconds, max_response_bytes
        self.admission = DirectAdmissions(store)
        self.owner_id = "direct-api-" + uuid4().hex
        self.tasks = set()

    async def open(self, tenant_id, payload, *, request_bound, path="chat/completions", batch_size=1):
        raw = encode(payload).encode()
        if len(raw) > 1024*1024:
            raise AdmissionDenied("direct request exceeds transport bound")
        if self.pool.kind == "llm" and (payload.get("model") != self.model
            or type(payload.get("n", 1)) is not int or payload.get("n", 1) != 1
            or type(payload.get("stream", False)) is not bool):
            raise AdmissionDenied("direct request differs from the qualified model/completion contract")
        request = DirectRequest(request_id=uuid4().hex, tenant_id=tenant_id,
            payload_digest=sha256(raw).hexdigest(), model_profile=self.pool.model_profile,
            capacity_profile_id=self.pool.profile_id, kind=self.pool.kind,
            request_bound=request_bound, batch_size=batch_size,
            deadline=datetime.now(UTC)+timedelta(seconds=self.timeout_seconds))
        reservation = await self.admission.reserve(request, self.pool.pool_id, self.owner_id)
        if reservation is None:
            raise AdmissionDenied("inference capacity unavailable", retryable=True)
        channel = _Channel()
        headers = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(self._produce(reservation, payload, path, channel, headers))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        try:
            status, response_headers = await asyncio.shield(headers)
            return _DirectResponse(channel, status_code=status,
                headers=response_headers | {"X-Intramind-Attempt-ID": reservation.attempt_id})
        except BaseException:
            await channel.detach()
            raise

    async def _produce(self, reservation, payload, path, channel, headers):
        sent = terminated = False
        error = None

        async def heartbeat():
            while True:
                await asyncio.sleep(max(1, self.store.lease_seconds/3))
                if not await self.admission.heartbeat(reservation):
                    raise RuntimeError("direct ownership expired")

        pulse = None
        try:
            async with asyncio.timeout(self.timeout_seconds):
                if reservation.engine_epoch != self.pool.engine_epoch:
                    raise RuntimeError("direct engine epoch changed; reload the qualified pool")
                await self.admission.mark_send(reservation)
                sent = True
                pulse = asyncio.create_task(heartbeat())
                current = asyncio.current_task()

                def lost_ownership(task):
                    if not task.cancelled() and task.exception() is not None:
                        current.cancel()

                pulse.add_done_callback(lost_ownership)
                async with self.client.stream("POST", path, json=payload,
                    headers={"X-Intramind-Attempt-ID": reservation.attempt_id,
                             "X-Intramind-Expected-Revision": self.pool.model_revision}) as response:
                    streaming = self.pool.kind == "llm" and payload.get("stream", False) and response.status_code == 200
                    status = response.status_code
                    content_type = response.headers.get("Content-Type", "application/json")
                    if self.pool.kind != "llm":
                        contract = response.headers.get(f"X-Intramind-{self.pool.kind.title()}-Contract")
                        state = response.headers.get("X-Intramind-Compute-State")
                        terminated = (contract == "termination-v1"
                            and response.headers.get("X-Intramind-Attempt-ID") == reservation.attempt_id
                            and state in {"terminated", "not_started"})
                        if not terminated:
                            raise RuntimeError("direct backend termination unconfirmed")
                        if status == 200 and (state != "terminated" or response.headers.get(
                            "X-Intramind-Model-Revision") != self.pool.model_revision):
                            raise RuntimeError("direct backend model revision or termination changed")
                    forwarded = {name: value for name in (
                        "Retry-After", "X-Intramind-Model-Revision", "X-Intramind-Compute-State",
                        f"X-Intramind-{self.pool.kind.title()}-Contract")
                        if (value := response.headers.get(name)) is not None and len(value) <= 512}
                    forwarded["Content-Type"] = content_type
                    if not headers.done():
                        headers.set_result((status, forwarded))
                    size, pending = 0, b""
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > self.max_response_bytes:
                            raise ValueError("direct response exceeds bound")
                        if streaming:
                            pending += chunk
                            lines = pending.split(b"\n")
                            pending = lines.pop()
                            if len(pending) > 1024*1024:
                                raise ValueError("direct stream frame exceeds bound")
                            terminated = terminated or any(
                                line.startswith(b"data:") and line[5:].strip() == b"[DONE]" for line in lines)
                        for offset in range(0, len(chunk), 16*1024):
                            await channel.put(chunk[offset:offset+16*1024])
                    if self.pool.kind == "llm" and not streaming:
                        terminated = status in {200, 400, 401, 403, 404, 422}
                    if not terminated:
                        raise RuntimeError("direct backend termination unconfirmed")
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
            sent = False
            error = "direct backend connection failed"
        except BaseException as exc:
            error = "direct inference interrupted"
            logger.warning("Direct inference interrupted attempt_id=%s error=%s", reservation.attempt_id, type(exc).__name__)
        finally:
            if pulse:
                pulse.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await pulse
            try:
                if terminated or not sent:
                    await self.admission.finish(reservation, evidence="completed_response" if terminated else "not_sent")
                else:
                    await self.admission.unknown(reservation, error or "direct_response_unconfirmed")
            except Exception as exc:
                logger.error("Direct settlement requires reconciliation attempt_id=%s error=%s",
                             reservation.attempt_id, type(exc).__name__)
            if not headers.done():
                headers.set_result((502, {"Content-Type": "application/json"}))
                await channel.put(encode({"error": {
                    "message": error or "direct inference unavailable", "type": "upstream_error"
                }}).encode())
                error = None
            await channel.finish(error)

    async def close(self):
        """Shutdown records uncertainty rather than pretending the backend stopped."""
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.client.aclose()
