"""Trusted task registry classification; callers cannot promote their own jobs."""

USER_TASK_TYPES = frozenset({
    "summary", "mindmap", "rewrite", "draft", "enrich", "pptx", "exam",
    "teaching", "audio", "audio-overview", "translation", "directive-review",
    "directive_review",
})
MAINTENANCE_TASK_TYPES = frozenset({"conversation.compact", "conversation.title", "memory"})


def workload_for_task(task_type: str, definition: dict) -> str:
    explicit = definition.get("priority")
    if explicit is not None:
        return explicit
    base = task_type.split("/", 1)[0]
    if base in USER_TASK_TYPES:
        return "user_task"
    if base in MAINTENANCE_TASK_TYPES:
        return "maintenance"
    return "background"
