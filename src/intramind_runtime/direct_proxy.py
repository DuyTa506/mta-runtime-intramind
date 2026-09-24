"""Stream foreground HTTP while an owned producer drains admitted inference."""

import asyncio
import json
import logging
import math
from collections import deque
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import uuid4

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from .contracts import AdmissionDenied, RuntimeConflict
from .direct import DirectRequest
from .memory_scheduler import MemoryScheduler
from .store import encode

logger = logging.getLogger(__name__)

DEFAULT_WORKLOAD_DEADLINE_SECONDS = {"qa": 600, "user_task": 1800}


class DirectDeadlineExceeded(TimeoutError):
    """The logical request budget ended; it is never termination evidence."""


async def _chunks(response, buffered):
    if buffered is not None:
        yield buffered
    else:
        async for chunk in response.aiter_bytes():
            yield chunk


class _Channel:
    def __init__(self):
        self.ready = asyncio.Condition()
        self.chunks = deque()
        self.detached = self.ended = False
        self.detached_event = asyncio.Event()
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
            self.detached_event.set()
            self.chunks.clear()
            self.ready.notify_all()

    async def finish(self, error=None):
        async with self.ready:
            if self.ended:
                return
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
    """One endpoint dispatcher feeds the engine; no SQL polling per waiting user."""

    def __init__(self, store, *, client, pool, model=None, profile=None,
                 timeout_seconds=1800, max_response_bytes=16*1024*1024,
                 workload_deadline_seconds=None, scheduler=None):
        if (type(timeout_seconds) not in {int, float} or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 86400):
            raise ValueError("pool attempt timeout must be finite and positive")
        self.store, self.client, self.pool = store, client, pool
        self.model = model or pool.model_revision
        self.profile = profile
        self.timeout_seconds, self.max_response_bytes = timeout_seconds, max_response_bytes
        deadlines = {**DEFAULT_WORKLOAD_DEADLINE_SECONDS,
                     "background": timeout_seconds, "maintenance": timeout_seconds}
        if workload_deadline_seconds is not None:
            if not isinstance(workload_deadline_seconds, dict):
                raise ValueError("workload deadlines must be a mapping")
            for workload, seconds in workload_deadline_seconds.items():
                if (workload not in deadlines or type(seconds) not in {int, float}
                    or not math.isfinite(seconds) or not 0 < seconds <= 86400):
                    raise ValueError("workload deadline must be finite and positive")
                deadlines[workload] = seconds
        self.workload_deadline_seconds = deadlines
        self.admission = scheduler or MemoryScheduler(store)
        self._owns_scheduler = scheduler is None
        self.owner_id = "direct-api-" + uuid4().hex
        self.boot_generation = uuid4().hex
        self._start_lock = asyncio.Lock()
        self._pending = {}
        self._recovering = {}
        self._settlements = {}
        self._pending_signal = asyncio.Event()
        self._dispatch_task = self._owner_task = None
        self._wakeup = None
        self._started = self._closing = False
        self.tasks = set()

    async def start(self):
        async with self._start_lock:
            if self._started:
                return
            await self.admission.start()
            self._wakeup = self.admission
            self._dispatch_task = asyncio.create_task(self._dispatch())
            self._dispatch_task.add_done_callback(self._dispatch_done)
            self._started = True

    def _dispatch_done(self, task):
        if task.cancelled() or self._closing:
            return
        error = task.exception()
        logger.critical("Direct endpoint dispatcher stopped pool=%s error=%r",
                        self.pool.pool_id, error)

    async def _settle(self, reservation, evidence):
        """Retain a failed ledger settlement until the endpoint dispatcher retries it."""
        self._settlements[reservation.attempt_id] = (reservation, evidence)
        await self.admission.finish(reservation, evidence=evidence)
        self._settlements.pop(reservation.attempt_id, None)

    async def _heartbeat_owner(self):
        while not self._closing:
            await asyncio.sleep(max(1, self.store.lease_seconds/3))
            try:
                await self.store.heartbeat_owner(self.owner_id, self.boot_generation)
            except Exception:
                logger.exception("Direct endpoint owner heartbeat failed pool=%s", self.pool.pool_id)

    async def _dispatch(self):
        while not self._closing:
            try:
                await self._dispatch_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Direct endpoint dispatcher failed pool=%s; retrying", self.pool.pool_id)
                await asyncio.sleep(1)

    async def _dispatch_once(self):
        for reservation, evidence in list(self._settlements.values()):
            await self._settle(reservation, evidence)
        current_pool = self.admission.pools.get(self.pool.pool_id)
        if (current_pool is not None and current_pool.epoch != self.pool.engine_epoch
            and current_pool.spec.profile_id == self.pool.profile_id
            and current_pool.spec.model_revision == self.pool.model_revision):
            self.pool = self.pool.model_copy(update={"engine_epoch": current_pool.epoch,
                "target": current_pool.target, "valid_until": current_pool.spec.valid_until})
        generation = self._wakeup.generation
        self._pending_signal.clear()
        if self._recovering:
            try:
                statuses = await self.admission.recovery_statuses(list(self._recovering))
                for attempt_id, future in list(self._recovering.items()):
                    status = statuses.get(attempt_id)
                    if not status or future.done():
                        continue
                    if status["state"] == "FAILED_RECOVERABLE" and status["health"] == "HEALTHY" \
                            and status["current_epoch"] != status["engine_epoch"]:
                        future.set_result(status)
                        self._recovering.pop(attempt_id, None)
                    elif status["state"] == "FAILED":
                        future.set_exception(RuntimeError("inference recovery was declined"))
                        self._recovering.pop(attempt_id, None)
            except Exception:
                logger.exception("Direct recovery observation failed pool=%s", self.pool.pool_id)
        pending = sorted(self._pending.values(), key=lambda item: (
            {"qa": 0, "user_task": 1, "background": 2, "maintenance": 3}[item[0].workload_class],
            item[2], item[0].request_id))
        dispatched = 0
        for request, future, _, previous_attempt_id in pending[:8]:
            if future.done():
                self._pending.pop(request.request_id, None)
                continue
            try:
                reservation = (await self.admission.retry_confirmed(request, self.pool.pool_id,
                               self.owner_id, previous_attempt_id) if previous_attempt_id else
                               await self.admission.reserve(request, self.pool.pool_id, self.owner_id))
            except (AdmissionDenied, RuntimeConflict) as exc:
                pending_entry = self._pending.get(request.request_id)
                if pending_entry is not None and pending_entry[1] is future:
                    self._pending.pop(request.request_id, None)
                if not future.done():
                    future.set_exception(exc)
                continue
            except Exception as exc:
                logger.exception("Direct endpoint dispatch failed pool=%s error=%s",
                                 self.pool.pool_id, type(exc).__name__)
                break
            if reservation is None:
                break
            pending_entry = self._pending.get(request.request_id)
            if (future.done() or pending_entry is None or pending_entry[1] is not future):
                await self._settle(reservation, "not_sent")
                continue
            if reservation.engine_epoch != self.pool.engine_epoch:
                if previous_attempt_id is None:
                    pending_entry = self._pending.get(request.request_id)
                    if pending_entry is not None and pending_entry[1] is future:
                        self._pending.pop(request.request_id, None)
                    if not future.done():
                        future.set_exception(RuntimeConflict("proxy has an older engine epoch"))
                    await self._settle(reservation, "not_sent")
                    continue
                self.pool = self.pool.model_copy(update={"engine_epoch": reservation.engine_epoch})
            self._pending.pop(request.request_id, None)
            future.set_result(reservation)
            dispatched += 1
        if dispatched:
            await asyncio.sleep(0)
            return
        # A quiescent endpoint has no need to observe every shared scheduler
        # update from other pools. Local enqueue/recovery/settlement sets this
        # signal when it has work again.
        if not self._pending and not self._recovering and not self._settlements:
            await self._pending_signal.wait()
            return
        wake = asyncio.create_task(self._wakeup.wait(generation, timeout=5))
        local = asyncio.create_task(self._pending_signal.wait())
        try:
            done, _ = await asyncio.wait((wake, local), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in (wake, local):
                task.cancel()
            await asyncio.gather(wake, local, return_exceptions=True)

    @staticmethod
    def _control(kind: str, generation: int = 0, reason: str | None = None) -> bytes:
        body = {"type": kind, "generation": generation}
        if reason:
            body["reason"] = reason
        return b"event: intramind.control\ndata: " + encode(body).encode() + b"\n\n"

    def _deadline_seconds(self, workload_class, requested):
        try:
            limit = self.workload_deadline_seconds[workload_class]
        except KeyError:
            raise ValueError("invalid inference workload") from None
        if requested is not None:
            if type(requested) is not int or requested <= 0:
                raise ValueError("inference deadline must be a positive number of seconds")
            limit = min(limit, requested)
        return limit

    @staticmethod
    def _remaining(until):
        remaining = until-asyncio.get_running_loop().time()
        if remaining <= 0:
            raise DirectDeadlineExceeded("inference request deadline exceeded")
        return remaining

    @staticmethod
    def _error_frame(message, error_type, reason=None):
        data = {"message": message, "type": error_type}
        if reason is not None:
            data["reason"] = reason
        return b"event: intramind.error\ndata: " + encode(data).encode() + b"\n\n"

    @staticmethod
    def _error_object(message, error_type, reason=None):
        data = {"message": message, "type": error_type}
        if reason is not None:
            data["reason"] = reason
        return {"error": data}

    def _deadline_response(self, streaming):
        message = "inference request deadline exceeded"
        if not streaming:
            return JSONResponse(self._error_object(message, "admission_timeout",
                                    "deadline_exceeded"), status_code=504)

        async def expired():
            yield self._error_frame(message, "admission_timeout", "deadline_exceeded")

        return StreamingResponse(expired(), status_code=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})

    async def open(self, tenant_id, payload, *, request_bound, path="chat/completions", batch_size=1,
                   workload_class="background", logical_request_id=None, disconnected=None,
                   deadline_seconds=None, started_at_monotonic=None):
        raw = encode(payload).encode()
        if len(raw) > 1024*1024:
            raise AdmissionDenied("direct request exceeds transport bound")
        if self.pool.kind == "llm" and (payload.get("model") != self.model
            or type(payload.get("n", 1)) is not int or payload.get("n", 1) != 1
            or type(payload.get("stream", False)) is not bool):
            raise AdmissionDenied("direct request differs from the qualified model/completion contract")
        streaming = self.pool.kind == "llm" and payload.get("stream", False)
        total_seconds = self._deadline_seconds(workload_class, deadline_seconds)
        now = asyncio.get_running_loop().time()
        started = now if started_at_monotonic is None else min(started_at_monotonic, now)
        until = started+total_seconds
        if until <= now:
            return self._deadline_response(streaming)
        if not self._started:
            try:
                async with asyncio.timeout_at(until):
                    await self.start()
            except TimeoutError:
                if until <= asyncio.get_running_loop().time():
                    return self._deadline_response(streaming)
                raise
        remaining = until-asyncio.get_running_loop().time()
        if remaining <= 0:
            return self._deadline_response(streaming)
        deadline = datetime.now(UTC)+timedelta(seconds=remaining)
        request = DirectRequest(request_id=uuid4().hex, tenant_id=tenant_id,
            payload_digest=sha256(raw).hexdigest(), model_profile=self.pool.model_profile,
            model_revision=self.pool.model_revision,
            expected_engine_epoch=self.pool.engine_epoch,
            capacity_profile_id=self.pool.profile_id, kind=self.pool.kind,
            request_bound=request_bound, batch_size=batch_size,
            workload_class=workload_class, logical_request_id=logical_request_id,
            deadline=deadline)
        try:
            async with asyncio.timeout_at(until):
                queued_at = await self.admission.enqueue(request, self.pool.pool_id, self.owner_id)
        except BaseException as exc:
            # Enqueue can commit before caller cancellation is observed. No
            # producer exists yet to remove the resulting waiter.
            try:
                await asyncio.shield(self.admission.leave(request.request_id, self.owner_id))
            except Exception:
                logger.exception("Direct enqueue cleanup failed request=%s", request.request_id)
            if isinstance(exc, TimeoutError) and until <= asyncio.get_running_loop().time():
                return self._deadline_response(streaming)
            raise
        future = asyncio.get_running_loop().create_future()
        self._pending[request.request_id] = (request, future, queued_at, None)
        self._pending_signal.set()
        channel = _Channel()
        headers = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(self._produce_waiting(request, future, payload, path,
                                                          channel, headers, streaming, disconnected,
                                                          until))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        if streaming:
            return _DirectResponse(channel, status_code=200,
                headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        try:
            status, response_headers = await asyncio.shield(headers)
            return _DirectResponse(channel, status_code=status,
                headers=response_headers)
        except BaseException:
            await channel.detach()
            raise

    async def _produce_waiting(self, request, future, payload, path, channel, headers,
                               streaming, disconnected, until):
        reservation = None
        handed_off = False
        try:
            if streaming and not future.done():
                await channel.put(self._control("waiting", reason="inference_capacity"))
            detached = asyncio.create_task(channel.detached_event.wait())
            try:
                while not future.done():
                    if channel.detached or (disconnected and await disconnected()):
                        return
                    remaining = self._remaining(until)
                    done, _ = await asyncio.wait((future, detached),
                        timeout=min(remaining, 1 if disconnected else 15),
                        return_when=asyncio.FIRST_COMPLETED)
                    if detached in done:
                        return
                    if not done and streaming and not disconnected:
                        await channel.put(b": heartbeat\n\n")
            finally:
                detached.cancel()
            reservation = future.result()
            self._remaining(until)
            if channel.detached:
                return
            if streaming:
                await channel.put(self._control("resumed"))
            if channel.detached:
                return
            self._remaining(until)
            handed_off = True
            await self._produce(reservation, payload, path, channel, headers, streaming, until)
        except DirectDeadlineExceeded as exc:
            if streaming:
                await channel.put(self._error_frame(str(exc), "admission_timeout",
                                                    "deadline_exceeded"))
                await channel.finish()
            else:
                if not headers.done():
                    headers.set_result((504, {"Content-Type": "application/json"}))
                await channel.put(encode(self._error_object(str(exc), "admission_timeout",
                    "deadline_exceeded")).encode())
                await channel.finish()
        except Exception as exc:
            if streaming:
                await channel.put(b"event: intramind.error\ndata: " + encode({
                    "message": str(exc), "type": "admission_error"}).encode() + b"\n\n")
                await channel.finish()
            else:
                if not headers.done():
                    response_headers = {"Content-Type": "application/json"}
                    if reservation is not None:
                        response_headers["X-Intramind-Attempt-ID"] = reservation.attempt_id
                    headers.set_result((503, response_headers))
                await channel.put(encode({"error": {"message": str(exc),
                    "type": "admission_error"}}).encode())
                await channel.finish()
        finally:
            pending_entry = self._pending.get(request.request_id)
            if pending_entry is not None and pending_entry[1] is future:
                self._pending.pop(request.request_id, None)
            if reservation is None:
                if not future.done():
                    future.cancel()
                elif not future.cancelled():
                    try:
                        granted = future.result()
                    except Exception:
                        pass
                    else:
                        await self._settle(granted, "not_sent")
            elif not handed_off:
                await self._settle(reservation, "not_sent")
            await self.admission.leave(request.request_id, self.owner_id)
            if reservation is None and not headers.done() and not channel.ended:
                headers.set_result((499, {"Content-Type": "application/json"}))
                await channel.finish("direct request disconnected before inference")

    async def _produce(self, reservation, payload, path, channel, headers, streaming, until):
        current = reservation
        current_unsent = True
        deadline_exceeded = False
        error = None
        while True:
            if channel.detached:
                if current_unsent:
                    await self._settle(current, "not_sent")
                return
            try:
                self._remaining(until)
            except DirectDeadlineExceeded as exc:
                if current_unsent:
                    await self._settle(current, "not_sent")
                error, deadline_exceeded = str(exc), True
                break
            current_unsent = False
            outcome, error = await self._attempt(current, payload, path, channel, headers,
                                                  streaming, until)
            if outcome == "done":
                await channel.finish()
                return
            if outcome == "terminal_error":
                break
            if until <= asyncio.get_running_loop().time():
                error, deadline_exceeded = "inference request deadline exceeded", True
                break
            if outcome == "unknown":
                if self.pool.kind != "llm":
                    break
                if streaming:
                    await channel.put(self._control("waiting", current.generation,
                                                    "engine_termination_unconfirmed"))
                try:
                    recovered = await self._await_confirmed_recovery(current, channel, streaming,
                                                                     until)
                except DirectDeadlineExceeded as exc:
                    error, deadline_exceeded = str(exc), True
                    recovered = False
                except Exception as exc:
                    error = str(exc)
                    recovered = False
                if not recovered:
                    break
                if streaming:
                    await channel.put(self._control("recovering", current.generation,
                                                    "engine_epoch_stopped"))
            elif outcome != "not_sent":
                break
            if current.generation >= 2:
                error = "inference recovery limit reached"
                break
            try:
                replacement = await self._queue_retry(current, channel, streaming, until)
            except DirectDeadlineExceeded as exc:
                error, deadline_exceeded = str(exc), True
                break
            except Exception as exc:
                error = str(exc)
                break
            if replacement is None:
                return
            if channel.detached:
                await self._settle(replacement, "not_sent")
                return
            current = replacement
            current_unsent = True
            if streaming:
                await channel.put(self._control("generation_reset", current.generation,
                                                "engine_epoch_stopped" if outcome == "unknown"
                                                else "not_sent"))
                await channel.put(self._control("resumed", current.generation))
        if channel.detached:
            return
        error = error or "direct inference unavailable"
        if streaming:
            await channel.put(self._error_frame(error, "upstream_error",
                "deadline_exceeded" if deadline_exceeded else None))
        else:
            if not headers.done():
                headers.set_result((504 if deadline_exceeded else 502, {"Content-Type": "application/json",
                    "X-Intramind-Attempt-ID": current.attempt_id}))
            await channel.put(encode(self._error_object(error, "upstream_error",
                "deadline_exceeded" if deadline_exceeded else None)).encode())
        await channel.finish()

    async def _await_confirmed_recovery(self, reservation, channel, streaming, until):
        future = asyncio.get_running_loop().create_future()
        self._recovering[reservation.attempt_id] = future
        self._pending_signal.set()
        try:
            while not future.done() and not channel.detached:
                try:
                    await asyncio.wait_for(asyncio.shield(future),
                                           timeout=min(15, self._remaining(until)))
                except TimeoutError:
                    self._remaining(until)
                    if streaming:
                        await channel.put(b": heartbeat\n\n")
            if channel.detached:
                return False
            future.result()
            return True
        finally:
            self._recovering.pop(reservation.attempt_id, None)

    async def _queue_retry(self, reservation, channel, streaming, until):
        request = reservation.request
        self._remaining(until)
        try:
            async with asyncio.timeout_at(until):
                queued_at = await self.admission.enqueue(request, self.pool.pool_id, self.owner_id)
        except BaseException as exc:
            # A retry enqueue can also commit just before its deadline fires.
            try:
                await asyncio.shield(self.admission.leave(request.request_id, self.owner_id))
            except Exception:
                logger.exception("Direct retry enqueue cleanup failed request=%s", request.request_id)
            if isinstance(exc, TimeoutError) and until <= asyncio.get_running_loop().time():
                raise DirectDeadlineExceeded("inference request deadline exceeded") from exc
            raise
        future = asyncio.get_running_loop().create_future()
        replacement = None
        self._pending[request.request_id] = (request, future, queued_at, reservation.attempt_id)
        self._pending_signal.set()
        try:
            if streaming:
                await channel.put(self._control("waiting", reservation.generation,
                                                "inference_capacity"))
            detached = asyncio.create_task(channel.detached_event.wait())
            try:
                while not future.done() and not channel.detached:
                    done, _ = await asyncio.wait((future, detached),
                        timeout=min(15, self._remaining(until)),
                        return_when=asyncio.FIRST_COMPLETED)
                    if detached in done:
                        break
                    if not done and streaming:
                        await channel.put(b": heartbeat\n\n")
            finally:
                detached.cancel()
            if channel.detached:
                return None
            self._remaining(until)
            replacement = future.result()
            return replacement
        finally:
            pending_entry = self._pending.get(request.request_id)
            if pending_entry is not None and pending_entry[1] is future:
                self._pending.pop(request.request_id, None)
            if replacement is None:
                if not future.done():
                    future.cancel()
                elif not future.cancelled():
                    try:
                        granted = future.result()
                    except Exception:
                        pass
                    else:
                        await self._settle(granted, "not_sent")
            await self.admission.leave(request.request_id, self.owner_id)
            if replacement is not None and channel.detached:
                await self._settle(replacement, "not_sent")

    async def _attempt(self, reservation, payload, path, channel, headers, streaming, until):
        sent = terminated = False
        error = None
        retryable_not_sent = False
        try:
            async with asyncio.timeout(min(self.timeout_seconds, self._remaining(until))):
                await self.admission.mark_send(reservation)
                sent = True
                async with self.client.stream("POST", path, json=payload,
                    headers={"X-Intramind-Attempt-ID": reservation.attempt_id,
                             "X-Intramind-Expected-Revision": self.pool.model_revision}) as response:
                    status = response.status_code
                    content_type = response.headers.get("Content-Type", "application/json")
                    native_state = response.headers.get("X-Intramind-Compute-State")
                    if self.pool.kind != "llm":
                        contract = response.headers.get(f"X-Intramind-{self.pool.kind.title()}-Contract")
                        terminated = (contract == "termination-v1"
                            and response.headers.get("X-Intramind-Attempt-ID") == reservation.attempt_id
                            and native_state in {"terminated", "not_started"})
                        if not terminated:
                            raise RuntimeError("direct backend termination unconfirmed")
                        if status == 200 and (native_state != "terminated" or response.headers.get(
                            "X-Intramind-Model-Revision") != self.pool.model_revision):
                            raise RuntimeError("direct backend model revision or termination changed")
                    if self.pool.kind == "llm" and status == 503:
                        body = await response.aread()
                        if body.startswith(b"Loading model"):
                            retryable_not_sent = True
                            sent = False
                        else:
                            error = "engine rejected request without termination proof"
                        return ("not_sent" if retryable_not_sent else "unknown", error)
                    if self.pool.kind == "llm" and status not in {200, 400, 401, 403, 404, 422}:
                        return "unknown", "engine rejected request without termination proof"
                    if self.pool.kind == "llm" and status in {400, 401, 403, 404, 422}:
                        terminated = True
                    forwarded = {name: value for name in (
                        "Retry-After", "X-Intramind-Model-Revision", "X-Intramind-Compute-State",
                        f"X-Intramind-{self.pool.kind.title()}-Contract")
                        if (value := response.headers.get(name)) is not None and len(value) <= 512}
                    forwarded["Content-Type"] = content_type
                    size, pending = 0, b""
                    buffered = bytearray()
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > self.max_response_bytes:
                            raise ValueError("direct response exceeds bound")
                        if streaming and status == 200:
                            pending += chunk
                            lines = pending.split(b"\n")
                            pending = lines.pop()
                            if len(pending) > 1024*1024:
                                raise ValueError("direct stream frame exceeds bound")
                            terminated = terminated or any(
                                line.startswith(b"data:") and line[5:].strip() == b"[DONE]" for line in lines)
                            for offset in range(0, len(chunk), 16*1024):
                                await channel.put(chunk[offset:offset+16*1024])
                        else:
                            buffered.extend(chunk)
                    if self.pool.kind == "llm" and not streaming:
                        terminated = status in {200, 400, 401, 403, 404, 422}
                    if not terminated:
                        return "unknown", "direct backend termination unconfirmed"
                    if self.profile is not None and status == 200:
                        if content_type.split(";")[0] != "application/json":
                            raise ValueError("direct native response must be JSON")
                        self.profile.validate_response(json.loads(buffered), payload)
                    if streaming:
                        if status != 200:
                            await channel.put(b"event: intramind.error\ndata: " + encode({
                                "message": buffered.decode(errors="replace"),
                                "type": "upstream_error"}).encode() + b"\n\n")
                    else:
                        if not headers.done():
                            headers.set_result((status, forwarded | {
                                "X-Intramind-Attempt-ID": reservation.attempt_id}))
                        for offset in range(0, len(buffered), 16*1024):
                            await channel.put(bytes(buffered[offset:offset+16*1024]))
                    return "done", None
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
            sent = False
            error = "direct backend connection failed"
        except TimeoutError:
            error = ("inference request deadline exceeded"
                     if until <= asyncio.get_running_loop().time()
                     else "direct backend attempt timed out")
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            error = "direct inference interrupted"
            logger.exception("Direct inference interrupted attempt_id=%s error=%s",
                             reservation.attempt_id, type(exc).__name__)
        finally:
            try:
                if terminated or not sent:
                    await self._settle(reservation,
                        "completed_response" if terminated else "not_sent")
                else:
                    await self.admission.unknown(reservation, error or "direct_response_unconfirmed")
            except Exception as exc:
                logger.error("Direct settlement requires reconciliation attempt_id=%s error=%s",
                             reservation.attempt_id, type(exc).__name__)
        return ("terminal_error" if terminated else "not_sent" if not sent else "unknown"), error

    async def close(self):
        """Shutdown records uncertainty rather than pretending the backend stopped."""
        self._closing = True
        self._pending_signal.set()
        for request, future, _, _ in self._pending.values():
            if not future.done():
                future.cancel()
        for future in self._recovering.values():
            if not future.done():
                future.cancel()
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for task in (self._dispatch_task, self._owner_task):
            if task:
                task.cancel()
        await asyncio.gather(*(task for task in (self._dispatch_task, self._owner_task) if task),
                             return_exceptions=True)
        if self._owns_scheduler:
            await self.admission.close()
        await self.client.aclose()
