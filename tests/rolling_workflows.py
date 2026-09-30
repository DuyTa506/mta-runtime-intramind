"""Replayable fixtures for the map_children child schedule."""

import asyncio

from intramind_runtime.sdk import TaskPolicy, durable_task


@durable_task(
    name="runtime-rolling-leaf",
    version=1,
    policy=TaskPolicy(max_iterations=1, deadline_seconds=180),
)
async def rolling_leaf(ctx, inputs):
    await ctx.sleep(inputs["seconds"])
    if inputs.get("fail"):
        raise RuntimeError(f"leaf {inputs['index']} failed")
    return {"key": f"k{inputs['index']}", "sha256": "a" * 64, "size": inputs["index"] + 1}


@durable_task(
    name="runtime-rolling-contract",
    version=1,
    policy=TaskPolicy(max_iterations=1, deadline_seconds=180, child_window=2),
)
async def rolling_parent(ctx, inputs):
    async def ticks():
        # Timers issued while children run pin where each child start sits in the command order.
        for seconds in inputs.get("ticks", []):
            await ctx.sleep(seconds)

    children, _ = await asyncio.gather(
        ctx.map_children(
            task_type="runtime-rolling-leaf/v1",
            task_queue=ctx.task_queue,
            items=inputs["items"],
            item_key="index",
            key="leaf",
            window=2,
        ),
        ticks(),
    )
    return {
        "key": ",".join(child["key"] for child in children),
        "sha256": "b" * 64,
        "size": len(children),
    }
