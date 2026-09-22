"""Deferred, bounded submissions. PostgreSQL owns delivery; Temporal owns execution.

Each partition runs one batch at a time. A sealed batch retains its inputs,
accepted policy and deadline through crashes before artifact/upload/submit ACKs.
No lease or TTL deletes an accepted input. The outbox process also pumps batches.
"""

import logging
from datetime import timedelta
from hashlib import sha256

from pydantic import Field

from .contracts import AdmissionDenied, Artifact, Contract, NotFound, RootSpec, RuntimeConflict
from .store import encode, execute, row, rows

logger = logging.getLogger(__name__)


class BufferedSubmission(Contract):
    task_type: str = Field(min_length=1, max_length=120)
    submission_key: str = Field(min_length=1, max_length=200)
    partition_key: str = Field(min_length=1, max_length=200)
    input: Artifact
    configuration: Artifact
    delay_seconds: int = Field(ge=0, le=86400, strict=True)
    batch_size: int = Field(ge=1, le=256, strict=True)


class BufferedSubmissions:
    def __init__(self, store, artifacts, *, control_queue="intramind-control"):
        self.store, self.artifacts = store, artifacts
        self.control_queue = control_queue

    async def append(self, tenant_id: str, request: BufferedSubmission, definition: dict) -> str:
        async with self.store.transaction() as c:
            old = await row(c, """SELECT * FROM runtime_buffer_items
                WHERE tenant_id=:tenant AND submission_key=:key""",
                tenant=tenant_id, key=request.submission_key)
            if old:
                if (old["input"] != request.input.model_dump(mode="json")
                    or old["task_type"] != request.task_type
                    or old["partition_key"] != request.partition_key):
                    raise RuntimeConflict("buffered submission key/input conflict")
                return old["item_id"]
            limits = definition.get("buffering")
            if not limits:
                raise AdmissionDenied("task does not accept buffered inputs")
            if (request.batch_size > limits["max_batch_size"]
                or request.delay_seconds > limits["max_delay_seconds"]):
                raise AdmissionDenied("buffer policy exceeds the configured bounds")
            if request.input.size > limits.get("max_item_bytes", 64 * 1024):
                raise AdmissionDenied("buffered input exceeds the configured size bound")
            definition = definition | {"control_queue": self.control_queue}
            pending = await row(c, """SELECT count(*) AS total,
                count(*) FILTER (WHERE i.tenant_id=:tenant) AS tenant,
                count(*) FILTER (WHERE i.tenant_id=:tenant AND i.task_type=:task
                    AND i.partition_key=:partition) AS partition
                FROM runtime_buffer_items i
                LEFT JOIN runtime_buffer_batches b ON b.batch_id=i.batch_id
                LEFT JOIN runtime_roots r ON r.root_id=b.batch_id
                WHERE i.batch_id IS NULL OR b.state='PREPARING'
                    OR (b.state='SUBMITTED' AND r.state='RUNNING')""",
                tenant=tenant_id, task=request.task_type, partition=request.partition_key)
            authority = await row(c, "SELECT max_pending FROM runtime_authority WHERE id=1")
            if (pending["total"] >= authority["max_pending"]
                or pending["tenant"] >= limits["max_pending_per_tenant"]
                or pending["partition"] >= limits["max_pending_per_partition"]):
                raise AdmissionDenied("buffer backlog full", retryable=True)
            item_id = sha256(f"buffer\0{tenant_id}\0{request.submission_key}".encode()).hexdigest()
            await execute(c, """INSERT INTO runtime_buffer_items
                (item_id,tenant_id,task_type,partition_key,submission_key,input,configuration,
                 definition,batch_size,due_at)
                VALUES (:id,:tenant,:task,:partition,:key,CAST(:input AS jsonb),
                    CAST(:config AS jsonb),CAST(:definition AS jsonb),:size,
                    clock_timestamp()+make_interval(secs => :delay))""",
                id=item_id, tenant=tenant_id, task=request.task_type, partition=request.partition_key,
                key=request.submission_key, input=request.input.model_dump_json(),
                config=request.configuration.model_dump_json(), definition=encode(definition),
                size=request.batch_size, delay=request.delay_seconds)
            await self.store._wake(c)
            return item_id

    async def status(self, tenant_id: str, item_id: str) -> dict:
        async with self.store.engine.connect() as c:
            result = await row(c, """SELECT i.item_id,i.due_at,i.batch_id AS run_id,
                CASE WHEN b.state='FAILED' THEN 'FAILED'
                     ELSE COALESCE(r.state,b.state,'PENDING') END AS state,
                COALESCE(r.terminal_reason,b.terminal_reason) AS terminal_reason
                FROM runtime_buffer_items i
                LEFT JOIN runtime_buffer_batches b ON b.batch_id=i.batch_id
                LEFT JOIN runtime_roots r ON r.root_id=b.batch_id
                WHERE i.item_id=:id AND i.tenant_id=:tenant""", id=item_id, tenant=tenant_id)
            if not result:
                raise NotFound("buffered input")
            return dict(result)

    async def seal(self, *, limit: int = 8) -> list[dict]:
        """Atomic bounded materialization, including unfinished seals from a prior process."""
        if not 1 <= limit <= 32:
            raise ValueError("buffer pump limit must be between 1 and 32")
        async with self.store.transaction() as c:
            batches = [dict(b) for b in await rows(c, """SELECT * FROM runtime_buffer_batches
                WHERE state='PREPARING' AND retry_at<=clock_timestamp()
                ORDER BY retry_at,batch_id LIMIT :limit""", limit=limit)]
            for _ in range(limit - len(batches)):
                first = await row(c, """SELECT i.* FROM runtime_buffer_items i
                    WHERE i.batch_id IS NULL AND i.due_at<=clock_timestamp()
                    AND NOT EXISTS (SELECT 1 FROM runtime_buffer_batches b
                        LEFT JOIN runtime_roots r ON r.root_id=b.batch_id
                        WHERE b.tenant_id=i.tenant_id AND b.task_type=i.task_type
                            AND b.partition_key=i.partition_key
                            AND (b.state='PREPARING' OR (b.state='SUBMITTED'
                                AND (r.state='RUNNING' OR r.root_id IS NULL))))
                    AND NOT EXISTS (SELECT 1 FROM runtime_buffer_items older
                        WHERE older.tenant_id=i.tenant_id AND older.task_type=i.task_type
                            AND older.partition_key=i.partition_key AND older.batch_id IS NULL
                            AND older.ordinal<i.ordinal)
                    ORDER BY i.due_at,i.ordinal LIMIT 1""")
                if not first:
                    break
                candidates = await rows(c, """SELECT * FROM runtime_buffer_items
                    WHERE tenant_id=:tenant AND task_type=:task AND partition_key=:partition
                        AND batch_id IS NULL ORDER BY ordinal LIMIT :size""",
                    tenant=first["tenant_id"], task=first["task_type"],
                    partition=first["partition_key"], size=first["batch_size"])
                selected = []
                for item in candidates:
                    if (item["configuration"] != first["configuration"]
                        or item["definition"] != first["definition"]
                        or item["batch_size"] != first["batch_size"]):
                        break
                    selected.append(item["item_id"])
                batch_id = sha256(f"batch\0{first['item_id']}".encode()).hexdigest()
                batch = await row(c, """INSERT INTO runtime_buffer_batches
                    (batch_id,tenant_id,task_type,partition_key,definition,configuration,deadline)
                    VALUES (:id,:tenant,:task,:partition,CAST(:definition AS jsonb),
                        CAST(:config AS jsonb),clock_timestamp()+make_interval(secs => :duration))
                    RETURNING *""", id=batch_id, tenant=first["tenant_id"],
                    task=first["task_type"], partition=first["partition_key"],
                    definition=encode(first["definition"]), config=encode(first["configuration"]),
                    duration=first["definition"]["deadline_seconds"])
                await execute(c, """UPDATE runtime_buffer_items SET batch_id=:batch
                    WHERE item_id=ANY(:items)""", batch=batch_id, items=selected)
                batches.append(dict(batch))
            return batches

    async def dispatch(self, batch: dict) -> None:
        async with self.store.engine.connect() as c:
            items = await rows(c, """SELECT item_id,input FROM runtime_buffer_items
                WHERE batch_id=:id ORDER BY ordinal""", id=batch["batch_id"])
        if not items:
            raise RuntimeConflict("sealed buffer batch has no inputs")
        payload = {"partition_key": batch["partition_key"],
                   "items": [dict(item) for item in items],
                   "accepted_at": batch["created_at"].isoformat()}
        ref = await self.artifacts.put(batch["tenant_id"], encode(payload).encode())
        definition = batch["definition"]
        root = RootSpec(root_id=batch["batch_id"], tenant_id=batch["tenant_id"],
                        deadline=batch["deadline"], budget_limit=definition["budget_limit"],
                        priority=definition.get("priority", "background"))
        spec = {"workflow_type": batch["task_type"], "task_queue": definition["task_queue"],
                "input": {"root_id": root.root_id, "tenant_id": root.tenant_id,
                          "control_queue": definition["control_queue"], "input": ref.model_dump(mode="json"),
                          "configuration": batch["configuration"]}}
        # A lost submit ACK reattaches to the same root, even if it has already finished.
        await self.store.submit_run(root, f"buffer:{root.root_id}", ref.sha256, spec)
        async with self.store.transaction() as c:
            await execute(c, """UPDATE runtime_buffer_batches SET state='SUBMITTED'
                WHERE batch_id=:id AND state='PREPARING'""", id=root.root_id)

    async def tick(self) -> int:
        batches = await self.seal()
        for batch in batches:
            try:
                await self.dispatch(batch)
            except Exception as exc:
                # One corrupt/unavailable artifact cannot block other namespaces.
                logger.warning("Buffered dispatch deferred batch_id=%s error=%s",
                               batch["batch_id"], type(exc).__name__)
                async with self.store.transaction() as c:
                    await execute(c, """UPDATE runtime_buffer_batches
                        SET retry_at=clock_timestamp()+:delay,
                            state=CASE WHEN deadline<=clock_timestamp()
                                AND NOT EXISTS (SELECT 1 FROM runtime_roots WHERE root_id=:id)
                                THEN 'FAILED' ELSE state END,
                            terminal_reason=CASE WHEN deadline<=clock_timestamp()
                                AND NOT EXISTS (SELECT 1 FROM runtime_roots WHERE root_id=:id)
                                THEN 'buffer_dispatch_deadline_exceeded' ELSE terminal_reason END
                        WHERE batch_id=:id AND state='PREPARING'""",
                        id=batch["batch_id"], delay=timedelta(seconds=10))
        return len(batches)
