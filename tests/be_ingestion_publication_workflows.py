"""Exercise native BE publication as a child and retain its receipt across rollover."""

from intramind_runtime.sdk import TaskPolicy, durable_task


@durable_task(name="be-ingestion-publication-checkpoint", version=1,
              policy=TaskPolicy(max_iterations=1))
async def checkpoint_ingestion_publication(ctx, inputs):
    if not ctx.rollovers:
        prepared = await ctx.activity("test.ingestion-publication-source/v1",
            {"tenant_id": ctx.tenant_id, "root_id": ctx.root_id}, key="source")
        receipt = await ctx.run_child(task_type="ingestion.publish/v1", task_queue=ctx.task_queue,
                                      key="publish", inputs=prepared)
        ctx.continue_as_new({"receipt": receipt})
    return inputs["receipt"]
