"""A ready executor owns at most one reserved inference attempt."""

import asyncio
import json
import logging
import random
from contextlib import suppress
from datetime import UTC, datetime

from .artifacts import ArtifactPort
from .contracts import EmbeddingResult, RuntimeConflict, SendBudgetUnavailable, SpeechResult
from .drivers import DriverFailure, EngineDriver
from .store import Store, encode

log = logging.getLogger(__name__)


class Executor:
    def __init__(self, store: Store, artifacts: ArtifactPort, driver: EngineDriver,
                 pool_id: str, owner_id: str, permit_client):
        self.store, self.artifacts, self.driver = store, artifacts, driver
        self.pool_id, self.owner_id = pool_id, owner_id
        if permit_client is None:
            raise ValueError("executor requires runtime-api permits")
        self.permits = permit_client

    async def _heartbeat(self, reservation, permit_ready):
        while True:
            await asyncio.sleep(self.store.lease_seconds / 3)
            try:
                active = await self.store.heartbeat(reservation)
                if permit_ready.is_set():
                    await self.permits.heartbeat(reservation)
                if not active:
                    await self.driver.cancel(reservation)
            except Exception:
                # Keep the actual transport attached so a recoverable DB outage
                # does not discard a response. Authority will retain uncertainty.
                log.exception("attempt heartbeat failed: %s", reservation.attempt_id)

    async def tick(self) -> bool:
        reservation = await self.store.reserve_next(self.pool_id, self.owner_id)
        if reservation is None:
            return False
        permit_ready = asyncio.Event()
        heartbeat = asyncio.create_task(self._heartbeat(reservation, permit_ready))
        send_marked = False
        try:
            remaining = (reservation.attempt_deadline - datetime.now(UTC)).total_seconds()
            if remaining <= 0:
                raise DriverFailure("attempt_deadline_exceeded", not_sent=True, retry=True)
            try:
                await self.permits.acquire(reservation)
                permit_ready.set()
            except TimeoutError as exc:
                raise DriverFailure("attempt_deadline_exceeded", not_sent=True,
                                    retry=True) from exc
            remaining = (reservation.attempt_deadline - datetime.now(UTC)).total_seconds()
            if remaining <= 0:
                raise DriverFailure("attempt_deadline_exceeded", not_sent=True, retry=True)
            try:
                async with asyncio.timeout(remaining) as deadline:
                    payload = json.loads(await self.artifacts.get(reservation.operation.payload))
                    # The durable SEND_INTENT commit precedes the final permit
                    # check. A runtime-api crash in this narrow window then
                    # hydrates the attempt as occupied on restart.
                    await self.store.mark_send(reservation)
                    await self.permits.heartbeat(reservation)
                    send_marked = True
                    result = await self.driver.execute(reservation, payload)
            except TimeoutError as exc:
                raise DriverFailure(
                    "attempt_deadline_exceeded" if deadline.expired() else "transport_timeout",
                    not_sent=not send_marked, retry=not send_marked,
                ) from exc
            # A completed inference is never retried to repair persistence.
            termination_recorded = False
            while True:
                try:
                    if not termination_recorded:
                        await self.store.compute_finished(reservation)
                        termination_recorded = True
                        try:
                            await self.permits.release(reservation)
                            permit_ready.clear()
                        except Exception:
                            log.exception("permit release will be reconciled: %s",
                                          reservation.attempt_id)
                    attachments = ()
                    document = result.model_dump(mode="json")
                    if isinstance(result, SpeechResult):
                        audio = await self.artifacts.put(
                            reservation.operation.tenant_id, result.audio, "audio/wav")
                        attachments = (audio,)
                        document["body"] = result.body | {"audio": audio.model_dump(mode="json")}
                        usage = result.characters
                    elif isinstance(result, EmbeddingResult):
                        usage = max(1, result.characters)
                    else:
                        usage = (result.input_tokens + result.output_tokens
                            if result.input_tokens is not None and result.output_tokens is not None else None)
                    data = encode(document).encode()
                    artifact = await self.artifacts.put(reservation.operation.tenant_id, data)
                    await self.store.commit_result(reservation, artifact, usage,
                        **({"attachments": attachments} if attachments else {}))
                    return True
                except RuntimeConflict:
                    raise
                except Exception:
                    log.exception("retrying result persistence: %s", reservation.attempt_id)
                    await asyncio.sleep(2)
        except DriverFailure as exc:
            if exc.finished:
                await self.store.compute_finished(reservation)
            if exc.finished or exc.not_sent:
                await self.store.fail(reservation, str(exc), not_sent=exc.not_sent, retry=exc.retry,
                    delay=(0 if (exc.not_sent and not permit_ready.is_set()
                                 and str(exc) == "attempt_deadline_exceeded")
                           else random.uniform(0, min(60, 2 ** (reservation.sent_attempts + 1)))))
            else:
                await self.store.unknown(reservation, str(exc))
        except asyncio.CancelledError:
            # Persist when possible; lease reconciliation is the crash fallback.
            with suppress(Exception):
                if send_marked:
                    await self.store.unknown(reservation, "executor_shutdown")
                else:
                    await self.store.fail(reservation, "shutdown_before_send", not_sent=True, retry=True)
            raise
        except SendBudgetUnavailable as exc:
            if not send_marked:
                await self.store.fail(reservation, exc.reason, not_sent=True, retry=True)
            log.info("attempt could not send under current budget: %s (%s)",
                     reservation.attempt_id, exc.reason)
        except RuntimeConflict:
            # A fenced worker may neither send nor change another owner's state.
            if not send_marked:
                with suppress(Exception):
                    await self.store.fail(reservation, "permit_fenced_before_send",
                                          not_sent=True, retry=True)
            log.warning("attempt fenced: %s", reservation.attempt_id)
        except Exception:
            with suppress(Exception):
                if send_marked:
                    await self.store.unknown(reservation, "executor_error_after_send")
                else:
                    await self.store.fail(reservation, "payload_unavailable", not_sent=True, retry=True)
            log.exception("attempt execution failed: %s", reservation.attempt_id)
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
            if permit_ready.is_set():
                try:
                    if not await self.store.attempt_compute_held(reservation.attempt_id):
                        await self.permits.release(reservation)
                except Exception:
                    log.exception("permit retained pending DB reconciliation: %s",
                                  reservation.attempt_id)
        return True
