"""PostgreSQL authority for admission and settlement.

The first deployment serializes short ledger mutations on one authority row.
No lock is held during inference, artifact I/O or Temporal RPCs. This deliberately
favors auditable accounting over sharding before the measured workload needs it.
"""

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from .contracts import (
    AdmissionDenied,
    Artifact,
    InferenceOperation,
    InferencePool,
    NotFound,
    PoolSpec,
    Reservation,
    RootSpec,
    RuntimeConflict,
    parse_operation,
    parse_pool,
)


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _submission_identity(spec: dict) -> dict:
    # Configuration is chosen by the accepting service, not part of the caller's
    # request identity. The first committed snapshot wins, even across a retry
    # after deployment or a concurrent submit handled by another service replica.
    return spec | {"input": {k: v for k, v in spec["input"].items() if k != "configuration"}}


async def row(c: AsyncConnection, sql: str, **params):
    return (await c.execute(text(sql), params)).mappings().first()


async def rows(c: AsyncConnection, sql: str, **params):
    return (await c.execute(text(sql), params)).mappings().all()


async def execute(c: AsyncConnection, sql: str, **params):
    return await c.execute(text(sql), params)


class Store:
    def __init__(self, url: str, *, lease_seconds: int = 60):
        if not url.startswith("postgresql+asyncpg://"):
            raise ValueError("The authoritative ledger requires PostgreSQL/asyncpg")
        self.engine = create_async_engine(url, pool_size=5, max_overflow=5, pool_pre_ping=True,
                                          hide_parameters=True)
        self.lease_seconds = lease_seconds

    async def close(self):
        await self.engine.dispose()

    async def drain_status(self) -> dict[str, int]:
        """Shutdown blockers; releasing compute does not make an output durable."""
        async with self.engine.connect() as c:
            status = await row(c, """SELECT
                count(*) FILTER (WHERE compute_held) AS compute_held,
                count(*) FILTER (WHERE NOT compute_held
                    AND (budget_held OR state='BACKEND_FINISHED')) AS pending_settlement,
                count(*) FILTER (WHERE state='UNKNOWN') AS unknown_attempts,
                count(*) FILTER (WHERE compute_held OR budget_held
                    OR state IN ('UNKNOWN','BACKEND_FINISHED')) AS unsettled_attempts
                FROM runtime_attempts""")
            return dict(status)

    async def _wake(self, c):
        # PostgreSQL delivers this only after commit; no prompt or identity is
        # broadcast. Repeated notifications in one transaction are coalesced.
        await execute(c, "SELECT pg_notify('intramind_runtime_wakeup','')")

    @asynccontextmanager
    async def transaction(self):
        async with self.engine.begin() as c:
            await execute(c, "SELECT id FROM runtime_authority WHERE id=1 FOR UPDATE")
            yield c

    async def _root(self, c, spec: RootSpec):
        existing = await row(c, "SELECT * FROM runtime_roots WHERE root_id=:id", id=spec.root_id)
        data = spec.model_dump(mode="json")
        if existing:
            if RootSpec.model_validate(existing["spec"]) != spec:
                raise RuntimeConflict("root identity/input conflict")
            return
        limits = await row(c, "SELECT * FROM runtime_authority WHERE id=1")
        count = await row(c, "SELECT count(*) AS n FROM runtime_roots WHERE state='RUNNING'")
        if count["n"] >= limits["max_roots"]:
            raise AdmissionDenied("root backlog full", retryable=True)
        if spec.deadline <= datetime.now(UTC):
            raise AdmissionDenied("root deadline expired")
        await execute(c, """INSERT INTO runtime_roots
            (root_id,tenant_id,spec,deadline,budget_limit,priority)
            VALUES (:id,:tenant,CAST(:spec AS jsonb),:deadline,:budget,:priority)""",
            id=spec.root_id, tenant=spec.tenant_id, spec=encode(data),
            deadline=spec.deadline, budget=spec.budget_limit, priority=spec.priority)
        for unit, limit in spec.resource_budgets.items():
            await execute(c, """INSERT INTO runtime_resource_budgets(root_id,unit,budget_limit)
                VALUES (:root,:unit,:limit)""", root=spec.root_id, unit=unit, limit=limit)

    async def create_root(self, spec: RootSpec):
        async with self.transaction() as c:
            await self._root(c, spec)

    async def submit_run(self, root: RootSpec, key: str, input_digest: str, spec: dict):
        async with self.transaction() as c:
            old = await row(c, """SELECT * FROM runtime_submissions
                WHERE tenant_id=:tenant AND submission_key=:key""", tenant=root.tenant_id, key=key)
            if old:
                if (old["input_digest"] != input_digest
                    or _submission_identity(old["spec"]) != _submission_identity(spec)):
                    raise RuntimeConflict("submission key/input conflict")
                return old["run_id"]
            await self._root(c, root)
            await execute(c, """INSERT INTO runtime_submissions VALUES
                (:run,:tenant,:key,:digest,CAST(:spec AS jsonb))""",
                run=root.root_id, tenant=root.tenant_id, key=key,
                digest=input_digest, spec=encode(spec))
            intent = spec | {"input": spec["input"] | {"deadline": root.deadline.isoformat()}}
            await self._event(c, f"start:{root.root_id}", "start_workflow", root.root_id, intent)
            return root.root_id

    async def configure_pool(self, spec: InferencePool, group_ceiling: int):
        if group_ceiling <= 0:
            raise ValueError("group ceiling must be positive")
        async with self.transaction() as c:
            old = await row(c, "SELECT * FROM runtime_pools WHERE pool_id=:id", id=spec.pool_id)
            if old and (old["engine_epoch"] != spec.engine_epoch or old["spec"] != spec.model_dump(mode="json")):
                active = await row(c, """SELECT count(*) AS n FROM runtime_attempts
                    WHERE pool_id=:id AND (compute_held OR budget_held)""", id=spec.pool_id)
                if active["n"]:
                    raise RuntimeConflict("drain/reconcile existing attempts before reconfiguration")
            await execute(c, """INSERT INTO runtime_groups(group_id,hard_ceiling) VALUES (:g,:n)
                ON CONFLICT(group_id) DO UPDATE SET hard_ceiling=EXCLUDED.hard_ceiling""",
                g=spec.group_id, n=group_ceiling)
            await execute(c, """INSERT INTO runtime_pools
                (pool_id,group_id,spec,engine_epoch,target,hard_ceiling,context_limit,model_profile,valid_until)
                VALUES (:id,:g,CAST(:spec AS jsonb),:epoch,:target,:ceiling,:context,:model,:until)
                ON CONFLICT(pool_id) DO UPDATE SET spec=EXCLUDED.spec,
                engine_epoch=EXCLUDED.engine_epoch, target=EXCLUDED.target,
                hard_ceiling=EXCLUDED.hard_ceiling, context_limit=EXCLUDED.context_limit,
                model_profile=EXCLUDED.model_profile, valid_until=EXCLUDED.valid_until,
                health=CASE WHEN runtime_pools.engine_epoch!=EXCLUDED.engine_epoch THEN 'HEALTHY'
                            ELSE runtime_pools.health END,
                group_id=EXCLUDED.group_id, envelope_version=runtime_pools.envelope_version+1""",
                id=spec.pool_id, g=spec.group_id, spec=spec.model_dump_json(), epoch=spec.engine_epoch,
                target=spec.target, ceiling=spec.hard_ceiling,
                context=spec.context_limit if isinstance(spec, PoolSpec) else 0,
                model=spec.model_profile, until=spec.valid_until)
            await self._wake(c)

    async def submit_operation(self, spec: InferenceOperation):
        from .artifacts import tenant_prefix
        if not spec.payload.key.startswith(tenant_prefix(spec.tenant_id)):
            raise NotFound("payload")
        async with self.transaction() as c:
            old = await row(c, "SELECT spec FROM runtime_operations WHERE operation_id=:id",
                            id=spec.operation_id)
            if old:
                if parse_operation(old["spec"]) != spec:
                    raise RuntimeConflict("operation identity/input conflict")
                return spec.operation_id
            root = await row(c, """SELECT *,clock_timestamp() AS observed_at FROM runtime_roots
                WHERE root_id=:id AND tenant_id=:tenant""",
                             id=spec.root_id, tenant=spec.tenant_id)
            if not root:
                raise NotFound("root")
            if root["cancel_requested"] or root["state"] != "RUNNING" or root["deadline"] <= datetime.now(UTC):
                raise AdmissionDenied("root is not eligible")
            if (spec.budget_unit != "tokens"
                and spec.budget_unit not in root["spec"].get("resource_budgets", {})):
                raise AdmissionDenied(f"root has no accepted {spec.budget_unit} budget")
            count = await row(c, """SELECT count(*) FILTER(WHERE root_id=:root) AS root_pending,
                count(*) AS total FROM runtime_operations
                WHERE state NOT IN ('SUCCEEDED','FAILED','CANCELLED')""", root=spec.root_id)
            limits = await row(c, "SELECT * FROM runtime_authority WHERE id=1")
            if root["operation_count"] >= root["spec"]["max_operations"]:
                raise AdmissionDenied("lifetime operation limit reached")
            expired = spec.deadline is not None and spec.deadline <= root["observed_at"]
            if not expired and (count["root_pending"] >= root["spec"]["max_pending"]
                or count["total"] >= limits["max_pending"]):
                raise AdmissionDenied("materialization limit reached", retryable=True)
            inserted = await row(c, """INSERT INTO runtime_operations(operation_id,root_id,tenant_id,spec)
                VALUES (:id,:root,:tenant,CAST(:spec AS jsonb)) RETURNING clock_timestamp() AS observed_at""",
                id=spec.operation_id, root=spec.root_id, tenant=spec.tenant_id, spec=spec.model_dump_json())
            await execute(c, "UPDATE runtime_roots SET operation_count=operation_count+1 WHERE root_id=:id",
                          id=spec.root_id)
            await self._expire_operation(c, spec, inserted["observed_at"])
            await self._wake(c)
            return spec.operation_id

    async def operation(self, operation_id: str, tenant_id: str):
        async with self.engine.connect() as c:
            result = await row(c, """SELECT operation_id,root_id,state,result,wait_reason,attempts
                FROM runtime_operations WHERE operation_id=:id AND tenant_id=:tenant""",
                id=operation_id, tenant=tenant_id)
            if not result:
                raise NotFound("operation")
            return dict(result)

    async def run(self, root_id: str, tenant_id: str):
        async with self.engine.connect() as c:
            root = await row(c, """SELECT r.root_id,r.state,r.deadline,r.reserved,r.spent,
                r.budget_limit,r.cancel_requested,r.result,r.terminal_reason,r.finished_at,
                s.spec->>'workflow_type' AS task_type,s.input_digest
                FROM runtime_roots r LEFT JOIN runtime_submissions s ON s.run_id=r.root_id
                WHERE r.root_id=:id AND r.tenant_id=:tenant""", id=root_id, tenant=tenant_id)
            if not root:
                raise NotFound("root")
            counts = await rows(c, """SELECT state,count(*) AS n FROM runtime_operations
                WHERE root_id=:id GROUP BY state""", id=root_id)
            cleanup = await row(c, """SELECT count(*) AS n FROM runtime_attempts a
                JOIN runtime_operations o USING(operation_id) WHERE o.root_id=:id
                AND (a.compute_held OR a.budget_held)""", id=root_id)
            budgets = await rows(c, """SELECT unit,budget_limit,reserved,spent
                FROM runtime_resource_budgets WHERE root_id=:id""", id=root_id)
            return dict(root) | {"operations": {x["state"]: x["n"] for x in counts},
                "cleanup_pending": cleanup["n"] > 0,
                "resource_budgets": {b["unit"]: {"limit": b["budget_limit"],
                    "reserved": b["reserved"], "spent": b["spent"]} for b in budgets}}

    async def reserve_next(self, pool_id: str, owner_id: str) -> Reservation | None:
        """Called only by an idle executor; no downstream worker queue."""
        async with self.transaction() as c:
            pool = await row(c, "SELECT * FROM runtime_pools WHERE pool_id=:id", id=pool_id)
            if not pool or pool["health"] != "HEALTHY" or pool["valid_until"] <= datetime.now(UTC):
                return None
            profile = parse_pool(pool["spec"])
            group = await row(c, "SELECT * FROM runtime_groups WHERE group_id=:id", id=pool["group_id"])
            used = await row(c, """SELECT count(*) AS group_used,
                count(*) FILTER(WHERE a.pool_id=:pool) AS pool_used FROM runtime_attempts a
                JOIN runtime_pools p USING(pool_id) WHERE a.compute_held AND p.group_id=:g""",
                pool=pool_id, g=pool["group_id"])
            if (group["health"] != "HEALTHY" or used["group_used"] >= group["hard_ceiling"]
                or used["pool_used"] >= min(pool["target"], pool["hard_ceiling"])):
                return None
            authority = await row(c, "SELECT dispatch_clock FROM runtime_authority WHERE id=1")
            preferred = "background" if authority["dispatch_clock"] % 5 == 4 else "interactive"
            # Tenant service history is aggregated first, so splitting a job
            # into many roots does not gain priority over another tenant.
            candidates = await rows(c, """SELECT o.*,r.spec AS root_spec,r.reserved,r.spent,
                r.budget_limit,r.attempts AS root_attempts,r.deadline AS root_deadline,
                clock_timestamp() AS observed_at
                FROM runtime_operations o JOIN runtime_roots r USING(root_id)
                WHERE o.state IN ('READY','RETRY_WAIT') AND (o.retry_at IS NULL OR o.retry_at<=now())
                AND r.state='RUNNING' AND NOT r.cancel_requested AND r.deadline>now()
                AND o.spec->>'model_profile'=:model
                AND COALESCE(o.spec->>'kind','llm')=:kind
                AND (o.spec->>'capacity_profile_id' IS NULL OR o.spec->>'capacity_profile_id'=:profile)
                AND CASE WHEN :kind='speech' THEN (o.spec->>'characters_bound')::bigint
                    ELSE (o.spec->>'input_tokens_bound')::bigint+(o.spec->>'max_output_tokens')::bigint
                    END<=:request_limit
                AND CAST(:capabilities AS jsonb) @> (o.spec->'required_capabilities')
                ORDER BY (r.priority=:preferred) DESC,
                (SELECT COALESCE(max(r2.last_served),0) FROM runtime_roots r2
                 WHERE r2.tenant_id=r.tenant_id),r.last_served,o.created_at,o.operation_id LIMIT 64""",
                model=pool["model_profile"], preferred=preferred, profile=pool["spec"]["profile_id"],
                kind=profile.kind, request_limit=profile.request_limit,
                capabilities=encode(pool["spec"]["capabilities"]))
            for candidate in candidates:
                spec = parse_operation(candidate["spec"])
                if await self._expire_operation(c, spec, candidate["observed_at"]):
                    continue
                if spec.capacity_profile_id and spec.capacity_profile_id != pool["spec"]["profile_id"]:
                    continue
                reason = None
                if not spec.required_capabilities.issubset(pool["spec"]["capabilities"]):
                    continue
                if spec.budget_bound > profile.request_limit:
                    # Another compatible pool may fit: don't fail on this pool's view.
                    continue
                budget = candidate if spec.budget_unit == "tokens" else await row(c,
                    "SELECT * FROM runtime_resource_budgets WHERE root_id=:root AND unit=:unit",
                    root=spec.root_id, unit=spec.budget_unit)
                if budget is None:
                    await self._terminal(c, spec.operation_id, "FAILED", "resource_budget_missing")
                    continue
                if candidate["attempts"] >= spec.max_attempts:
                    reason = "max_attempts"
                elif candidate["root_attempts"] >= candidate["root_spec"]["max_attempts"]:
                    reason = "root_attempt_budget"
                elif budget["spent"] + spec.budget_bound > budget["budget_limit"]:
                    reason = "root_budget_exhausted"
                elif budget["spent"] + budget["reserved"] + spec.budget_bound > budget["budget_limit"]:
                    await execute(c, "UPDATE runtime_operations SET wait_reason='root_budget' WHERE operation_id=:id",
                                  id=spec.operation_id)
                    continue
                if reason:
                    await self._terminal(c, spec.operation_id, "FAILED", reason)
                    continue
                attempt_id = str(uuid4())
                number = candidate["attempts"] + 1
                attempt = await row(c, """INSERT INTO runtime_attempts
                    (attempt_id,operation_id,pool_id,engine_epoch,attempt_number,owner_id,
                     lease_epoch,lease_expires_at,budget_bound,budget_unit,created_at)
                    VALUES (:id,:op,:pool,:epoch,:number,:owner,:fence,:lease,:bound,:unit,clock_timestamp())
                    RETURNING created_at""",
                    id=attempt_id, op=spec.operation_id, pool=pool_id, epoch=pool["engine_epoch"],
                    number=number, fence=number, owner=owner_id, lease=datetime.now(UTC)+timedelta(seconds=self.lease_seconds),
                    bound=spec.budget_bound, unit=spec.budget_unit)
                await execute(c, """UPDATE runtime_operations SET state='EXECUTING',attempts=:n,
                    active_attempt=:attempt,wait_reason=NULL WHERE operation_id=:id""",
                    n=number, attempt=attempt_id, id=spec.operation_id)
                clock = authority["dispatch_clock"] + 1
                await execute(c, "UPDATE runtime_authority SET dispatch_clock=:n WHERE id=1", n=clock)
                await execute(c, """UPDATE runtime_roots SET reserved=reserved+:bound,
                    attempts=attempts+1,last_served=:clock WHERE root_id=:id""",
                    bound=spec.budget_bound if spec.budget_unit == "tokens" else 0,
                    clock=clock, id=spec.root_id)
                if spec.budget_unit != "tokens":
                    await execute(c, """UPDATE runtime_resource_budgets SET reserved=reserved+:bound
                        WHERE root_id=:root AND unit=:unit""",
                        root=spec.root_id, unit=spec.budget_unit, bound=spec.budget_bound)
                return Reservation(attempt_id=attempt_id, operation=spec, pool_id=pool_id,
                    engine_epoch=pool["engine_epoch"], model_revision=pool["spec"]["model_revision"],
                    owner_id=owner_id, lease_epoch=number,
                    attempt_deadline=min(candidate["root_deadline"], spec.deadline or candidate["root_deadline"], attempt["created_at"]
                        + timedelta(seconds=spec.attempt_timeout_seconds)))
            return None

    async def _owned(self, c, reservation: Reservation):
        attempt = await row(c, """SELECT a.*,o.root_id,o.tenant_id,o.active_attempt,
            o.state AS operation_state,o.spec AS operation_spec,r.cancel_requested,
            r.state AS root_state,r.deadline,clock_timestamp() AS observed_at FROM runtime_attempts a
            JOIN runtime_operations o USING(operation_id) JOIN runtime_roots r USING(root_id)
            WHERE a.attempt_id=:id""", id=reservation.attempt_id)
        if (not attempt or attempt["owner_id"] != reservation.owner_id
            or attempt["lease_epoch"] != reservation.lease_epoch
            or attempt["active_attempt"] != reservation.attempt_id):
            raise RuntimeConflict("stale attempt ownership")
        return attempt

    async def mark_send(self, reservation: Reservation):
        async with self.transaction() as c:
            a = await self._owned(c, reservation)
            spec = parse_operation(a["operation_spec"])
            deadline = min(a["deadline"], spec.deadline or a["deadline"],
                a["created_at"] + timedelta(seconds=spec.attempt_timeout_seconds))
            if (a["state"] != "RESERVED" or a["operation_state"] != "EXECUTING"
                or a["lease_expires_at"] <= a["observed_at"]
                or a["cancel_requested"] or a["root_state"] != "RUNNING" or deadline <= a["observed_at"]):
                raise RuntimeConflict("attempt cannot send")
            await execute(c, """UPDATE runtime_attempts SET state='SEND_INTENT',send_intent_at=now()
                WHERE attempt_id=:id""", id=reservation.attempt_id)

    async def heartbeat(self, reservation: Reservation):
        async with self.transaction() as c:
            a = await self._owned(c, reservation)
            if a["state"] not in ("RESERVED", "SEND_INTENT", "ACTIVE", "BACKEND_FINISHED"):
                return False
            await execute(c, "UPDATE runtime_attempts SET lease_expires_at=:until WHERE attempt_id=:id",
                until=datetime.now(UTC)+timedelta(seconds=self.lease_seconds), id=reservation.attempt_id)
            return (not a["cancel_requested"] and a["root_state"] == "RUNNING"
                    and a["operation_state"] not in ("SUCCEEDED", "FAILED", "CANCELLED"))

    async def compute_finished(self, reservation: Reservation):
        async with self.transaction() as c:
            a = await self._owned(c, reservation)
            if a["state"] in ("RESULT_COMMITTED", "FAILED", "FAILED_NOT_SENT"):
                return
            await execute(c, """UPDATE runtime_attempts SET compute_held=false,
                backend_finished_at=COALESCE(backend_finished_at,now()),state='BACKEND_FINISHED'
                WHERE attempt_id=:id""", id=reservation.attempt_id)

            await self._wake(c)

    async def _settle(self, c, a, usage: int | None):
        if not a["budget_held"]:
            return
        charged = a["budget_bound"] if usage is None else usage
        if charged < 0:
            raise ValueError("negative usage")
        await execute(c, """UPDATE runtime_attempts SET budget_held=false,usage=:usage,
            usage_estimated=:estimated WHERE attempt_id=:id""",
            id=a["attempt_id"], usage=charged, estimated=usage is None)
        if a["budget_unit"] == "tokens":
            budget = await row(c, """UPDATE runtime_roots
                SET reserved=reserved-:bound,spent=spent+:usage WHERE root_id=:root
                RETURNING spent,budget_limit""", root=a["root_id"], bound=a["budget_bound"], usage=charged)
        else:
            budget = await row(c, """UPDATE runtime_resource_budgets
                SET reserved=reserved-:bound,spent=spent+:usage WHERE root_id=:root AND unit=:unit
                RETURNING spent,budget_limit""", root=a["root_id"], unit=a["budget_unit"],
                bound=a["budget_bound"], usage=charged)
        if budget is None:
            raise RuntimeConflict("attempt budget ledger missing")
        if budget["spent"] > budget["budget_limit"]:
            await execute(c, """UPDATE runtime_roots SET state='FAILED',
                terminal_reason='observed_usage_exceeds_budget',finished_at=now()
                WHERE root_id=:root AND state='RUNNING'""", root=a["root_id"])
        if charged > a["budget_bound"]:
            await execute(c, "UPDATE runtime_pools SET target=0,health='DEGRADED' WHERE pool_id=:id",
                          id=a["pool_id"])
        await self._wake(c)

    async def commit_result(self, reservation: Reservation, artifact: Artifact, usage: int | None,
                            *, attachments: tuple[Artifact, ...] = ()):
        async with self.transaction() as c:
            a = await self._owned(c, reservation)
            if a["state"] == "RESULT_COMMITTED":
                existing = await row(c, "SELECT result FROM runtime_operations WHERE operation_id=:id",
                                     id=a["operation_id"])
                if existing["result"] and existing["result"] != artifact.model_dump(mode="json"):
                    raise RuntimeConflict("committed result differs")
                return
            if a["compute_held"] or a["state"] != "BACKEND_FINISHED":
                raise RuntimeConflict("backend termination must be recorded first")
            expired = await self._expire_operation(c, parse_operation(a["operation_spec"]), a["observed_at"])
            cancelled = a["cancel_requested"] or a["operation_state"] == "CANCELLED" or a["root_state"] != "RUNNING"
            terminal = expired or a["operation_state"] in ("SUCCEEDED", "FAILED", "CANCELLED")
            from .artifacts import tenant_prefix
            for ref in (artifact, *attachments):
                if not ref.key.startswith(tenant_prefix(a["tenant_id"])):
                    raise NotFound("result")
                await execute(c, """INSERT INTO runtime_artifacts
                    (object_key,tenant_id,operation_id,attempt_id,manifest,disposition)
                    VALUES (:key,:tenant,:op,:attempt,CAST(:manifest AS jsonb),:disposition)""",
                    key=ref.key, tenant=a["tenant_id"], op=a["operation_id"], attempt=a["attempt_id"],
                    manifest=ref.model_dump_json(), disposition=("late_cancelled" if cancelled
                        else "late_terminal" if terminal else "result"))
            await self._settle(c, a, usage)
            await execute(c, """UPDATE runtime_attempts SET state='RESULT_COMMITTED',
                result_committed_at=now() WHERE attempt_id=:id""", id=a["attempt_id"])
            if not cancelled and not terminal:
                await execute(c, """UPDATE runtime_operations SET state='SUCCEEDED',
                    result=CAST(:result AS jsonb),wait_reason=NULL WHERE operation_id=:id""",
                    result=artifact.model_dump_json(), id=a["operation_id"])
            elif cancelled and not terminal:
                await self._terminal(c, a["operation_id"], "FAILED", "root_terminal")
            await self._completion_events(c, a["operation_id"])

    async def unknown(self, reservation: Reservation, reason: str):
        async with self.transaction() as c:
            a = await self._owned(c, reservation)
            if a["state"] in ("RESULT_COMMITTED", "FAILED", "FAILED_NOT_SENT"):
                return
            await execute(c, """UPDATE runtime_attempts SET state='UNKNOWN',error_class=:reason,
                unknown_at=COALESCE(unknown_at,now()) WHERE attempt_id=:id""",
                reason=reason, id=a["attempt_id"])
            expired = await self._expire_operation(c, parse_operation(a["operation_spec"]), a["observed_at"])
            if not expired and a["operation_state"] not in ("SUCCEEDED", "FAILED", "CANCELLED"):
                await execute(c, """UPDATE runtime_operations SET state='RECONCILING',
                    wait_reason=:reason WHERE operation_id=:id""", reason=reason, id=a["operation_id"])

    async def fail(self, reservation: Reservation, reason: str, *, not_sent=False, retry=False, delay=0):
        """Caller must prove not-sent or call compute_finished before a failure."""
        async with self.transaction() as c:
            a = await self._owned(c, reservation)
            if a["state"] in ("RESULT_COMMITTED", "FAILED", "FAILED_NOT_SENT"):
                return
            if a["compute_held"] and not not_sent:
                raise RuntimeConflict("cannot fail/refund an attempt with unknown compute")
            await self._settle(c, a, 0 if not_sent else None)
            await execute(c, """UPDATE runtime_attempts SET state=:state,compute_held=false,
                error_class=:reason WHERE attempt_id=:id""",
                state="FAILED_NOT_SENT" if not_sent else "FAILED", reason=reason, id=a["attempt_id"])
            if a["operation_state"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
                return
            if a["cancel_requested"]:
                await self._terminal(c, a["operation_id"], "CANCELLED", "user_cancelled")
            elif a["root_state"] != "RUNNING":
                await self._terminal(c, a["operation_id"], "FAILED", "root_terminal")
            elif await self._expire_operation(c, parse_operation(a["operation_spec"]), a["observed_at"]):
                return
            elif retry:
                await execute(c, """UPDATE runtime_operations SET state='RETRY_WAIT',wait_reason=:reason,
                    retry_at=:at WHERE operation_id=:id""", reason=reason,
                    at=datetime.now(UTC)+timedelta(seconds=delay), id=a["operation_id"])
            else:
                await self._terminal(c, a["operation_id"], "FAILED", reason)

    async def cancel(self, root_id: str, tenant_id: str):
        async with self.transaction() as c:
            root = await row(c, "SELECT * FROM runtime_roots WHERE root_id=:id AND tenant_id=:tenant",
                             id=root_id, tenant=tenant_id)
            if not root:
                raise NotFound("root")
            if root["state"] in ("SUCCEEDED", "PARTIAL", "FAILED"):
                return
            await execute(c, """UPDATE runtime_roots SET cancel_requested=true,state='CANCELLED',
                terminal_reason='user_cancelled',finished_at=COALESCE(finished_at,now())
                WHERE root_id=:id""", id=root_id)
            ops = await rows(c, """SELECT operation_id FROM runtime_operations WHERE root_id=:id
                AND state NOT IN ('SUCCEEDED','FAILED','CANCELLED')""", id=root_id)
            for op in ops:
                await self._terminal(c, op["operation_id"], "CANCELLED", "user_cancelled")
            await self._event(c, f"cancel:{root_id}", "cancel_workflow", root_id, {"run_id": root_id})

    async def finish_run(
        self,
        root_id: str,
        tenant_id: str,
        state: str,
        result: Artifact | None = None,
        reason: str | None = None,
    ) -> None:
        """Commit a terminal result within the accepted deadline, without refunding compute."""
        if state not in ("SUCCEEDED", "PARTIAL", "FAILED"):
            raise ValueError("invalid terminal state")
        if result is not None:
            from .artifacts import tenant_prefix
            if not result.key.startswith(tenant_prefix(tenant_id)):
                raise NotFound("result")
        if state == "PARTIAL" and (result is None or not reason):
            raise ValueError("partial result requires an artifact and reason")
        async with self.transaction() as c:
            root = await row(
                c,
                """SELECT *,deadline<=clock_timestamp() AS deadline_expired FROM runtime_roots
                    WHERE root_id=:id AND tenant_id=:t""",
                id=root_id,
                t=tenant_id,
            )
            if not root:
                raise NotFound("root")
            manifest = result.model_dump(mode="json") if result else None
            if root["state"] != "RUNNING":
                if root["state"] == state and root["result"] != manifest:
                    raise RuntimeConflict("terminal result is immutable")
                return
            if root["deadline_expired"]:
                state, manifest, reason = "FAILED", None, "deadline_exceeded"
            pending = await row(c, """SELECT count(*) AS n FROM runtime_operations WHERE root_id=:id
                AND state NOT IN ('SUCCEEDED','FAILED','CANCELLED')""", id=root_id)
            if state == "SUCCEEDED" and pending["n"]:
                raise RuntimeConflict("cannot succeed with pending operations")
            # Sample the database clock once at the actual transition. The
            # transaction may have waited on the authority lock, and the
            # deadline can pass while pending operations are checked.
            finished = await row(
                c,
                """WITH completion AS MATERIALIZED (SELECT clock_timestamp() AS at)
                    UPDATE runtime_roots SET
                    state=CASE WHEN deadline<=completion.at THEN 'FAILED' ELSE :state END,
                    result=CASE WHEN deadline<=completion.at THEN NULL ELSE CAST(:result AS jsonb) END,
                    terminal_reason=CASE WHEN deadline<=completion.at
                        THEN 'deadline_exceeded' ELSE :reason END,
                    finished_at=completion.at
                    FROM completion WHERE root_id=:id RETURNING state,terminal_reason""",
                id=root_id,
                state=state,
                result=encode(manifest),
                reason=reason,
            )
            # A partial/failed parent stops pending branches. Active compute
            # remains in the attempt ledger until termination is established.
            if finished["state"] != "SUCCEEDED":
                ops = await rows(c, """SELECT operation_id FROM runtime_operations WHERE root_id=:id
                    AND state NOT IN ('SUCCEEDED','FAILED','CANCELLED')""", id=root_id)
                for op in ops:
                    await self._terminal(
                        c,
                        op["operation_id"],
                        "CANCELLED",
                        finished["terminal_reason"] or "root_finished",
                    )
            await self._wake(c)

    async def _terminal(self, c, operation_id, state, reason):
        await execute(c, """UPDATE runtime_operations SET state=:s,wait_reason=:r WHERE operation_id=:id
            AND state NOT IN ('SUCCEEDED','FAILED','CANCELLED')""",
                      s=state, r=reason, id=operation_id)
        await self._completion_events(c, operation_id)

    async def _expire_operation(self, c, spec: InferenceOperation, observed_at: datetime) -> bool:
        if spec.deadline is None or spec.deadline > observed_at:
            return False
        await self._terminal(c, spec.operation_id, "FAILED", "operation_deadline_exceeded")
        return True

    async def _event(self, c, event_id, kind, aggregate_id, payload):
        await execute(c, """INSERT INTO runtime_outbox(event_id,kind,aggregate_id,payload)
            VALUES (:id,:kind,:agg,CAST(:payload AS jsonb)) ON CONFLICT(event_id) DO NOTHING""",
            id=event_id, kind=kind, agg=aggregate_id, payload=encode(payload))
        await self._wake(c)

    async def _completion_events(self, c, operation_id):
        bindings = await rows(c, "SELECT binding_id FROM runtime_completion_bindings WHERE operation_id=:id",
                              id=operation_id)
        for binding in bindings:
            await self._event(c, f"complete:{binding['binding_id']}", "complete_activity", operation_id,
                              {"binding_id": binding["binding_id"]})

    async def bind_completion(self, operation_id, tenant_id, task_token: bytes):
        from hashlib import sha256
        binding_id = sha256(task_token).hexdigest()
        async with self.transaction() as c:
            op = await row(c, "SELECT * FROM runtime_operations WHERE operation_id=:id AND tenant_id=:t",
                           id=operation_id, t=tenant_id)
            if not op:
                raise NotFound("operation")
            await execute(c, """INSERT INTO runtime_completion_bindings VALUES (:id,:op,:token,now())
                ON CONFLICT(binding_id) DO NOTHING""", id=binding_id, op=operation_id, token=task_token)
            if op["state"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
                await self._completion_events(c, operation_id)

    async def claim_events(self, owner_id: str, limit=32):
        async with self.engine.begin() as c:
            return [dict(x) for x in await rows(c, """UPDATE runtime_outbox SET owner_id=:owner,
                lease_expires_at=now()+interval '30 seconds',deliveries=deliveries+1
                WHERE event_id IN (SELECT event_id FROM runtime_outbox
                  WHERE delivered_at IS NULL AND available_at<=now()
                  AND (lease_expires_at IS NULL OR lease_expires_at<now())
                  ORDER BY created_at LIMIT :limit FOR UPDATE SKIP LOCKED) RETURNING *""",
                owner=owner_id, limit=limit)]

    async def delivered(self, event_id: str, owner_id: str, delivery_number: int) -> bool:
        """Acknowledge only the current claim, including when a process reuses its owner ID."""
        async with self.engine.begin() as c:
            changed = await execute(
                c,
                """UPDATE runtime_outbox SET delivered_at=now()
                    WHERE event_id=:id AND owner_id=:owner AND deliveries=:delivery
                    AND delivered_at IS NULL AND lease_expires_at>now()""",
                id=event_id,
                owner=owner_id,
                delivery=delivery_number,
            )
            return changed.rowcount == 1

    async def defer_event(
        self, event_id: str, owner_id: str, delivery_number: int, delay_seconds: float
    ) -> bool:
        """Retain a failed delivery and persist its next eligible time before releasing ownership."""
        if not 1 <= delay_seconds <= 300:
            raise ValueError("outbox retry delay must be between 1 and 300 seconds")
        async with self.engine.begin() as c:
            changed = await execute(
                c,
                """UPDATE runtime_outbox
                    SET available_at=now()+(:delay * interval '1 second'),
                        owner_id=NULL,lease_expires_at=NULL
                    WHERE event_id=:id AND owner_id=:owner AND deliveries=:delivery
                    AND delivered_at IS NULL AND lease_expires_at>now()""",
                id=event_id,
                owner=owner_id,
                delivery=delivery_number,
                delay=delay_seconds,
            )
            return changed.rowcount == 1

    async def binding(self, binding_id: str):
        async with self.engine.connect() as c:
            return dict(await row(c, """SELECT b.task_token,o.* FROM runtime_completion_bindings b
                JOIN runtime_operations o USING(operation_id) WHERE binding_id=:id""", id=binding_id))

    async def reconcile_expired(self):
        async with self.transaction() as c:
            expired_operations = await rows(c, """SELECT operation_id FROM runtime_operations
                WHERE state NOT IN ('SUCCEEDED','FAILED','CANCELLED')
                AND (spec->>'deadline')::timestamptz<=clock_timestamp()""")
            for op in expired_operations:
                await self._terminal(c, op["operation_id"], "FAILED", "operation_deadline_exceeded")
            attempts = await rows(c, """SELECT a.*,o.root_id FROM runtime_attempts a
                JOIN runtime_operations o USING(operation_id) WHERE (lease_expires_at<now()
                    OR (a.state='RESERVED' AND o.state IN ('SUCCEEDED','FAILED','CANCELLED')))
                AND a.state IN ('RESERVED','SEND_INTENT','ACTIVE','BACKEND_FINISHED')""")
            for a in attempts:
                if a["state"] == "RESERVED":
                    await self._settle(c, a, 0)
                    await execute(c, """UPDATE runtime_attempts SET state='FAILED_NOT_SENT',compute_held=false,
                        error_class='lease_expired_before_send' WHERE attempt_id=:id""", id=a["attempt_id"])
                    await execute(c, """UPDATE runtime_operations SET state='READY',active_attempt=NULL
                        WHERE operation_id=:id AND state NOT IN ('SUCCEEDED','FAILED','CANCELLED')""", id=a["operation_id"])
                else:
                    await execute(c, """UPDATE runtime_attempts SET state='UNKNOWN',unknown_at=now(),
                        error_class='lease_expired_after_send' WHERE attempt_id=:id""", id=a["attempt_id"])
                    await execute(c, """UPDATE runtime_operations SET state='RECONCILING',wait_reason='unknown_compute_or_result'
                        WHERE operation_id=:id AND state NOT IN ('SUCCEEDED','FAILED','CANCELLED')""", id=a["operation_id"])
            expired = await rows(c, """SELECT o.operation_id FROM runtime_operations o
                JOIN runtime_roots r USING(root_id) WHERE r.deadline<=now()
                AND o.state IN ('READY','RETRY_WAIT')""")
            for op in expired:
                await self._terminal(c, op["operation_id"], "FAILED", "deadline_exceeded")
            # A known, impossible request must not remain queued forever.
            ready = await rows(c, """SELECT operation_id,spec FROM runtime_operations
                WHERE state IN ('READY','RETRY_WAIT') ORDER BY created_at LIMIT 256""")
            pools = await rows(c, "SELECT spec FROM runtime_pools")
            for op in ready:
                spec = parse_operation(op["spec"])
                compatible = [parse_pool(p["spec"]) for p in pools
                              if p["spec"]["model_profile"] == spec.model_profile
                              and p["spec"].get("kind", "llm") == spec.kind
                              and spec.required_capabilities.issubset(p["spec"]["capabilities"])]
                if compatible and spec.capacity_profile_id:
                    if not any(p.profile_id == spec.capacity_profile_id for p in compatible):
                        await self._terminal(c, op["operation_id"], "FAILED", "capacity_profile_changed")
                        continue
                    compatible = [p for p in compatible if p.profile_id == spec.capacity_profile_id]
                if compatible and all(p.request_limit < spec.budget_bound for p in compatible):
                    await self._terminal(c, op["operation_id"], "FAILED",
                        "context_exceeds_all_pools" if spec.kind == "llm" else "request_exceeds_all_pools")
            await execute(c, """UPDATE runtime_roots SET state='FAILED',terminal_reason='deadline_exceeded',
                finished_at=now() WHERE state='RUNNING' AND deadline<=now()""")
            terminal = await rows(c, """SELECT o.operation_id FROM runtime_operations o
                JOIN runtime_roots r USING(root_id) WHERE r.state!='RUNNING'
                AND o.state NOT IN ('SUCCEEDED','FAILED','CANCELLED')""")
            for op in terminal:
                await self._terminal(c, op["operation_id"], "CANCELLED", "root_terminal")
            return len(attempts)

    async def confirm_epoch_stopped(self, pool_id: str, engine_epoch: str, evidence: str):
        """Operator action after independent confirmation of instance termination.

        This records evidence, settles unknown usage conservatively and fails
        affected operations. It never retries inference or restarts an engine.
        """
        if not evidence.strip():
            raise ValueError("termination evidence is required")
        async with self.transaction() as c:
            pool = await row(c, "SELECT * FROM runtime_pools WHERE pool_id=:id", id=pool_id)
            if not pool or pool["engine_epoch"] != engine_epoch:
                raise RuntimeConflict("engine epoch does not match")
            await execute(c, "UPDATE runtime_pools SET target=0,health='UNAVAILABLE' WHERE pool_id=:id",
                          id=pool_id)
            attempts = await rows(c, """SELECT a.*,o.root_id,o.state AS operation_state
                FROM runtime_attempts a JOIN runtime_operations o USING(operation_id)
                WHERE a.pool_id=:pool AND a.engine_epoch=:epoch AND (a.compute_held OR a.budget_held)""",
                pool=pool_id, epoch=engine_epoch)
            for a in attempts:
                await self._settle(c, a, 0 if a["state"] == "RESERVED" else None)
                await execute(c, """UPDATE runtime_attempts SET state='FAILED',compute_held=false,
                    backend_finished_at=COALESCE(backend_finished_at,now()),error_class='engine_epoch_stopped'
                    WHERE attempt_id=:id""", id=a["attempt_id"])
                if a["operation_state"] != "CANCELLED":
                    await self._terminal(c, a["operation_id"], "FAILED", "engine_epoch_stopped")
            await execute(c, """INSERT INTO runtime_controller_updates(pool_id,envelope_version,target,reason)
                VALUES (:id,:version,0,:reason)""", id=pool_id, version=pool["envelope_version"],
                reason=f"epoch_stopped:{engine_epoch}:{evidence}")
            return len(attempts)

    async def update_target(self, pool_id: str, target: int, version: int, reason: str):
        async with self.transaction() as c:
            p = await row(c, "SELECT * FROM runtime_pools WHERE pool_id=:id", id=pool_id)
            if not p or p["envelope_version"] != version:
                raise RuntimeConflict("stale envelope")
            if target < 0 or target > p["hard_ceiling"]:
                raise AdmissionDenied("target outside hard ceiling")
            await execute(c, """UPDATE runtime_pools SET target=:target,envelope_version=envelope_version+1
                WHERE pool_id=:id""", target=target, id=pool_id)
            await execute(c, """INSERT INTO runtime_controller_updates(pool_id,envelope_version,target,reason)
                VALUES (:id,:version,:target,:reason)""", id=pool_id, version=version+1, target=target, reason=reason)
            await self._wake(c)
