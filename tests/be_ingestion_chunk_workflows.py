"""Qualify the native chunk phase inside a root with retry and history rollover."""

from intramind_runtime.sdk import TaskPolicy, durable_task


@durable_task(name="be-ingestion-chunk-checkpoint", version=1,
              policy=TaskPolicy(max_iterations=1))
async def checkpoint_ingestion_chunk(ctx, inputs):
    if not ctx.rollovers:
        source = await ctx.activity("test.ingestion-parsed-source/v1",
            {"tenant_id": ctx.tenant_id}, key="source")
        chunked = await ctx.run_child(task_type="ingestion.chunk/v1", task_queue=ctx.task_queue,
            key="chunk", inputs=source)
        ctx.continue_as_new({"chunked": chunked})
    return await ctx.activity("test.ingestion-after-chunks/v1",
        {"tenant_id": ctx.tenant_id, "chunked": inputs["chunked"]}, key="downstream")
