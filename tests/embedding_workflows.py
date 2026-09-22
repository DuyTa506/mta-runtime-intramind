"""Embedding batches retain their references across rollover and downstream retries."""

from intramind_runtime.sdk import TaskPolicy, durable_task


@durable_task(name="embedding-checkpoint", version=1, policy=TaskPolicy(max_iterations=2))
async def embedding_checkpoint(ctx, inputs):
    if not ctx.rollovers:
        plan = await ctx.activity("test.embedding-prepare/v1",
            {"tenant_id": ctx.tenant_id, "input": inputs}, key="prepare")
        results = []
        for index, request in enumerate(plan):
            results.append(await ctx.embedding(key=f"batch/{index}", **request))
        ctx.continue_as_new({"results": results})
    return await ctx.activity("test.embedding-publish/v1",
        {"tenant_id": ctx.tenant_id, "results": inputs["results"]}, key="publish")
