import asyncio
import os

import pytest
from conftest import operation, pool, root

from intramind_runtime.wakeup import Wakeup

pytestmark = pytest.mark.integration


async def test_committed_ready_work_wakes_idle_dispatch_without_poll_delay(store):
    await store.create_root(root())
    await store.configure_pool(pool(), 2)
    wakeup = Wakeup(os.environ["RUNTIME_TEST_DATABASE_URL"])
    await wakeup.start()
    try:
        await asyncio.wait_for(wakeup.ready.wait(), timeout=5)
        generation = wakeup.generation
        waiter = asyncio.create_task(wakeup.wait(generation, timeout=30))
        await store.submit_operation(operation())
        await asyncio.wait_for(waiter, timeout=2)
        assert wakeup.generation > generation
        assert await store.reserve_next("p", "executor")
        # Notification arrives while a dispatcher is busy; the next wait must
        # observe the generation change without consuming a polling interval.
        await asyncio.wait_for(wakeup.wait(generation, timeout=30), timeout=0.1)
    finally:
        await wakeup.close()
