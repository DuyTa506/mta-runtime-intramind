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


@durable_task(name="pptx-context-checkpoint", version=1, policy=TaskPolicy(max_iterations=1))
async def checkpoint_context(ctx: TaskContext, inputs: dict) -> dict:
    common = {"tenant_id": ctx.tenant_id}
    prepared = await ctx.activity("pptx.prepare/v1", common | {
        "input": inputs, "root_id": ctx.root_id, "configuration": ctx.configuration,
        "started_at": ctx.started_at.isoformat()}, key="prepare")
    context = await ctx.run_child(key="context", task_type="pptx.context/v1",
                                  task_queue=ctx.task_queue, inputs=prepared)
    return await ctx.activity("test.pptx.context-publish/v1", common | {"input": context}, key="publish")


@durable_task(name="pptx-planning-checkpoint", version=1, policy=TaskPolicy(max_iterations=1))
async def checkpoint_planning(ctx: TaskContext, inputs: dict) -> dict:
    common = {"tenant_id": ctx.tenant_id}
    prepared = await ctx.activity("pptx.prepare/v1", common | {
        "input": inputs, "root_id": ctx.root_id, "configuration": ctx.configuration,
        "started_at": ctx.started_at.isoformat()}, key="prepare")
    context = await ctx.run_child(key="context", task_type="pptx.context/v1",
                                  task_queue=ctx.task_queue, inputs=prepared)
    plan = await ctx.run_child(key="planning", task_type="pptx.planning/v1",
                              task_queue=ctx.task_queue, inputs=context)
    return await ctx.activity("test.pptx.planning-publish/v1", common | {"input": plan}, key="publish")


@durable_task(name="pptx-manuscript-checkpoint", version=1, policy=TaskPolicy(max_iterations=1))
async def checkpoint_manuscript(ctx: TaskContext, inputs: dict) -> dict:
    common = {"tenant_id": ctx.tenant_id}
    plan = await ctx.activity("test.pptx.manuscript-seed/v1", common | {
        "configuration": ctx.configuration}, key="prepare")
    manuscript = await ctx.run_child(key="manuscript", task_type="pptx.manuscript/v1",
                                    task_queue=ctx.task_queue, inputs=plan)
    return await ctx.activity("test.pptx.manuscript-publish/v1",
                              common | {"input": manuscript}, key="publish")


@durable_task(name="pptx-validation-checkpoint", version=1, policy=TaskPolicy(max_iterations=1))
async def checkpoint_validation(ctx: TaskContext, inputs: dict) -> dict:
    common = {"tenant_id": ctx.tenant_id}
    manuscript = await ctx.activity("test.pptx.validation-seed/v1", common | {
        "configuration": ctx.configuration}, key="prepare")
    result = await ctx.run_child(key="validation", task_type="pptx.validation/v1",
                                task_queue=ctx.task_queue, inputs=manuscript)
    return await ctx.activity("test.pptx.validation-publish/v1",
                              common | {"input": result}, key="publish")
