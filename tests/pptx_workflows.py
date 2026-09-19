"""Deterministic qualification workflow for native PPTX model checkpoints only."""

from intramind_runtime.sdk import TaskContext, TaskPolicy, durable_task


@durable_task(name="pptx-model-checkpoint", version=1, policy=TaskPolicy(max_iterations=2))
async def checkpoint_models(ctx: TaskContext, inputs: dict) -> dict:
    common = {"tenant_id": ctx.tenant_id}
    prepared = await ctx.activity("test.pptx.prepare/v1", common | {"input": inputs}, key="prepare")
    condensed = await ctx.model_step(key="condense", planner="ai.model_step/v1",
                                     inputs=prepared, max_model_calls=1)
    prepared = await ctx.activity("test.pptx.brief-input/v1", common | {"input": condensed}, key="brief-input")
    brief = await ctx.model_step(key="brief", planner="ai.model_step/v1",
                                inputs=prepared, max_model_calls=1)
    return await ctx.activity("test.pptx.publish/v1", common | {"input": brief}, key="publish")
