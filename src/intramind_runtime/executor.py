"""A ready executor owns at most one reserved inference attempt."""

import asyncio
import json
import logging
import random
from contextlib import suppress

from .artifacts import ArtifactPort
from .contracts import RuntimeConflict
from .drivers import DriverFailure, EngineDriver
from .store import Store, encode

log = logging.getLogger(__name__)


class Executor:
    def __init__(self, store: Store, artifacts: ArtifactPort, driver: EngineDriver,
                 pool_id: str, owner_id: str):
        self.store, self.artifacts, self.driver = store, artifacts, driver
        self.pool_id, self.owner_id = pool_id, owner_id

    async def _heartbeat(self, reservation):
        while True:
            await asyncio.sleep(self.store.lease_seconds / 3)
            try:
                active = await self.store.heartbeat(reservation)
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
        heartbeat = asyncio.create_task(self._heartbeat(reservation))
        send_marked = False
        try:
            payload = json.loads(await self.artifacts.get(reservation.operation.payload))
            await self.store.mark_send(reservation)
            send_marked = True
            result = await self.driver.execute(reservation, payload)
            # A completed inference is never retried to repair persistence.
            data = encode(result.model_dump(mode="json")).encode()
            termination_recorded = False
            while True:
                try:
                    if not termination_recorded:
                        await self.store.compute_finished(reservation)
                        termination_recorded = True
                    artifact = await self.artifacts.put(reservation.operation.tenant_id, data)
                    usage = (result.input_tokens + result.output_tokens
                             if result.input_tokens is not None and result.output_tokens is not None else None)
                    await self.store.commit_result(reservation, artifact, usage)
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
                    delay=random.uniform(0, min(60, 2 ** reservation.lease_epoch)))
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
        except RuntimeConflict:
            # A fenced worker may neither send nor change another owner's state.
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
        return True
