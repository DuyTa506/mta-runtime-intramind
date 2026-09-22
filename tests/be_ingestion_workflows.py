"""Exercise parsing as a child of a larger ingestion run with downstream retry."""

from intramind_runtime.sdk import TaskPolicy, durable_task


@durable_task(name="be-ingestion-parse-checkpoint", version=1,
              policy=TaskPolicy(max_iterations=1))
async def checkpoint_ingestion(ctx, inputs):
    if not ctx.rollovers:
        source = await ctx.activity(
            "test.ingestion-source/v1", {"tenant_id": ctx.tenant_id}, key="source"
        )
        parsed = await ctx.run_child(
            task_type="ingestion.parse/v1", task_queue=ctx.task_queue,
            key="parse", inputs=source,
        )
        ctx.continue_as_new({"parsed": parsed})
    return await ctx.activity(
        "test.ingestion-after-parse/v1",
        {"tenant_id": ctx.tenant_id, "parsed": inputs["parsed"]}, key="downstream",
    )
