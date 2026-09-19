"""Commit-coupled wakeups; durable rows remain the authority if signals are lost."""

import asyncio
import logging
from contextlib import suppress

import asyncpg
from sqlalchemy.engine import make_url

log = logging.getLogger(__name__)
CHANNEL = "intramind_runtime_wakeup"


class Wakeup:
    def __init__(self, database_url: str):
        self._dsn = (
            make_url(database_url)
            .set(drivername="postgresql")
            .render_as_string(hide_password=False)
        )
        self.generation = 0
        self._event = asyncio.Event()
        self.ready = asyncio.Event()
        self._task = None

    def _signal(self, *args):
        self.generation += 1
        self._event.set()

    async def start(self):
        self._task = asyncio.create_task(self._listen())

    async def _listen(self):
        while True:
            connection = None
            try:
                connection = await asyncpg.connect(self._dsn, timeout=5)
                disconnected = asyncio.Event()
                connection.add_termination_listener(lambda _: disconnected.set())
                await connection.add_listener(CHANNEL, self._signal)
                self.ready.set()
                self._signal()
                await disconnected.wait()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning(
                    "database wakeup unavailable: %s; bounded polling remains active",
                    type(exc).__name__,
                )
            finally:
                if connection:
                    with suppress(Exception):
                        await connection.close(timeout=2)
            await asyncio.sleep(2)

    async def wait(self, generation: int, timeout: float = 5):
        # No await between checking and clearing: a callback cannot slip into
        # that gap on this event loop. A commit during dispatch changes the
        # generation and therefore cannot be lost by clearing a stale signal.
        if self.generation != generation:
            return
        self._event.clear()
        with suppress(TimeoutError):
            await asyncio.wait_for(self._event.wait(), timeout=timeout)

    async def close(self):
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
