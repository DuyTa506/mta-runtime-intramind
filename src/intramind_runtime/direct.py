"""Foreground inference shares pool authority without workflow or artifact I/O."""

from datetime import datetime, timedelta
from typing import Literal
from uuid import uuid4

from pydantic import Field, field_validator

from .contracts import AdmissionDenied, Contract, RuntimeConflict, parse_pool
from .store import Store, execute, row


class DirectRequest(Contract):
    request_id: str = Field(min_length=1, max_length=200)
    tenant_id: str = Field(min_length=1, max_length=200)
    payload_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_profile: str = Field(min_length=1, max_length=120)
    capacity_profile_id: str = Field(min_length=1, max_length=200)
    kind: Literal["llm", "embedding", "speech", "rerank"] = "llm"
    request_bound: int = Field(gt=0, strict=True)
    batch_size: int = Field(default=1, gt=0, le=256, strict=True)
    deadline: datetime

    @field_validator("deadline")
    @classmethod
    def aware(cls, value):
        if value.tzinfo is None:
            raise ValueError("deadline must include timezone")
        return value


class DirectReservation(Contract):
    attempt_id: str
    request: DirectRequest
    pool_id: str
    engine_epoch: str
    owner_id: str


async def _background_ready(c, group_id):
    """Reserve a turn only for a compatible background operation that can dispatch."""
    return await row(c, """SELECT 1 FROM runtime_operations o
        JOIN runtime_roots r USING(root_id)
        JOIN runtime_pools p ON p.model_profile=o.spec->>'model_profile'
        LEFT JOIN runtime_resource_budgets b ON b.root_id=r.root_id
            AND b.unit=CASE o.spec->>'kind' WHEN 'speech' THEN 'speech_characters'
                WHEN 'embedding' THEN 'embedding_characters' ELSE 'tokens' END
        WHERE p.group_id=:group AND p.health='HEALTHY' AND p.valid_until>clock_timestamp()
        AND o.state IN ('READY','RETRY_WAIT') AND (o.retry_at IS NULL OR o.retry_at<=now())
        AND r.state='RUNNING' AND NOT r.cancel_requested AND r.deadline>clock_timestamp()
        AND (o.spec->>'deadline' IS NULL OR (o.spec->>'deadline')::timestamptz>clock_timestamp())
        AND COALESCE(o.spec->>'kind','llm')=COALESCE(p.spec->>'kind','llm')
        AND (o.spec->>'capacity_profile_id' IS NULL OR o.spec->>'capacity_profile_id'=p.spec->>'profile_id')
        AND (p.spec->'capabilities') @> (o.spec->'required_capabilities')
        AND o.attempts<(o.spec->>'max_attempts')::integer
        AND r.attempts<(r.spec->>'max_attempts')::integer
        AND (SELECT count(*) FROM runtime_inflight_attempts a WHERE a.pool_id=p.pool_id AND a.compute_held)
            <LEAST(p.target,p.hard_ceiling)
        AND CASE WHEN COALESCE(o.spec->>'kind','llm')='llm' THEN
            (o.spec->>'input_tokens_bound')::bigint+(o.spec->>'max_output_tokens')::bigint
                <=LEAST(p.context_limit,r.budget_limit-r.spent-r.reserved)
            ELSE b.root_id IS NOT NULL AND GREATEST(1,(o.spec->>'characters_bound')::bigint)
                <=LEAST((p.spec->>'character_limit')::bigint,b.budget_limit-b.spent-b.reserved) END
        AND (COALESCE(o.spec->>'kind','llm')<>'embedding' OR
            ((o.spec->>'texts_count')::integer<=(p.spec->>'max_batch_size')::integer
                AND o.spec->>'model_revision'=p.spec->>'model_revision')) LIMIT 1""", group=group_id)


class DirectAdmissions:
    """Only trusted inference adapters report termination; expiry never proves it."""

    def __init__(self, store: Store):
        self.store = store

    async def reserve(self, request: DirectRequest, pool_id: str, owner_id: str):
        if not owner_id:
            raise ValueError("executor identity is required")
        async with self.store.transaction() as c:
            observed = (await row(c, "SELECT clock_timestamp() AS now"))["now"]
            previous = await row(c, """SELECT * FROM runtime_direct_attempts
                WHERE tenant_id=:tenant AND request_id=:request""",
                tenant=request.tenant_id, request=request.request_id)
            if previous:
                if (previous["spec"] != request.model_dump(mode="json") or previous["owner_id"] != owner_id
                    or previous["pool_id"] != pool_id):
                    raise RuntimeConflict("direct request identity already has different ownership or content")
                if previous["state"] != "RESERVED" or min(previous["deadline"], previous["lease_expires_at"]) <= observed:
                    return None
                return DirectReservation(attempt_id=previous["attempt_id"], request=request,
                    pool_id=pool_id, engine_epoch=previous["engine_epoch"], owner_id=owner_id)
            pool = await row(c, "SELECT * FROM runtime_pools WHERE pool_id=:id", id=pool_id)
            if (not pool or pool["health"] != "HEALTHY" or pool["valid_until"] <= observed
                or request.deadline <= observed):
                return None
            profile = parse_pool(pool["spec"])
            if (request.kind != profile.kind or request.model_profile != profile.model_profile
                or request.capacity_profile_id != profile.profile_id
                or request.request_bound > profile.request_limit
                or request.batch_size > getattr(profile, "max_batch_size", 1)):
                raise AdmissionDenied("direct request exceeds its qualified pool profile")
            group = await row(c, "SELECT * FROM runtime_groups WHERE group_id=:id", id=pool["group_id"])
            used = await row(c, """SELECT count(*) AS group_used,
                count(*) FILTER (WHERE a.pool_id=:pool) AS pool_used FROM runtime_inflight_attempts a
                JOIN runtime_pools p USING(pool_id) WHERE a.compute_held AND p.group_id=:group""",
                pool=pool_id, group=pool["group_id"])
            if (group["health"] != "HEALTHY" or used["group_used"] >= group["hard_ceiling"]
                or used["pool_used"] >= min(pool["target"], pool["hard_ceiling"])):
                return None
            clock = (await row(c, "SELECT dispatch_clock FROM runtime_authority WHERE id=1"))["dispatch_clock"]
            if clock % 5 == 4 and await _background_ready(c, pool["group_id"]):
                return None
            result = DirectReservation(attempt_id=str(uuid4()), request=request, pool_id=pool_id,
                                       engine_epoch=pool["engine_epoch"], owner_id=owner_id)
            await execute(c, """INSERT INTO runtime_direct_attempts
                (attempt_id,request_id,tenant_id,spec,pool_id,engine_epoch,owner_id,lease_expires_at,deadline)
                VALUES (:id,:request,:tenant,CAST(:spec AS jsonb),:pool,:epoch,:owner,:lease,:deadline)""",
                id=result.attempt_id, request=request.request_id, tenant=request.tenant_id,
                spec=request.model_dump_json(), pool=pool_id, epoch=result.engine_epoch, owner=owner_id,
                lease=observed+timedelta(seconds=self.store.lease_seconds), deadline=request.deadline)
            await execute(c, "UPDATE runtime_authority SET dispatch_clock=dispatch_clock+1 WHERE id=1")
            return result

    async def _owned(self, c, reservation):
        attempt = await row(c, "SELECT *,clock_timestamp() AS now FROM runtime_direct_attempts WHERE attempt_id=:id",
                            id=reservation.attempt_id)
        if (not attempt or attempt["owner_id"] != reservation.owner_id
            or attempt["spec"] != reservation.request.model_dump(mode="json")
            or attempt["pool_id"] != reservation.pool_id or attempt["engine_epoch"] != reservation.engine_epoch):
            raise RuntimeConflict("stale direct request ownership")
        return attempt

    async def mark_send(self, reservation):
        async with self.store.transaction() as c:
            a = await self._owned(c, reservation)
            if a["state"] != "RESERVED" or min(a["deadline"], a["lease_expires_at"]) <= a["now"]:
                raise RuntimeConflict("direct attempt cannot send")
            await execute(c, """UPDATE runtime_direct_attempts SET state='SEND_INTENT',send_intent_at=now()
                WHERE attempt_id=:id""", id=reservation.attempt_id)

    async def heartbeat(self, reservation):
        async with self.store.transaction() as c:
            a = await self._owned(c, reservation)
            if a["state"] not in {"RESERVED", "SEND_INTENT"} or min(a["deadline"], a["lease_expires_at"]) <= a["now"]:
                return False
            await execute(c, "UPDATE runtime_direct_attempts SET lease_expires_at=:until WHERE attempt_id=:id",
                until=min(a["deadline"], a["now"]+timedelta(seconds=self.store.lease_seconds)), id=reservation.attempt_id)
            return True

    async def finish(self, reservation, evidence="completed_response"):
        if evidence not in {"completed_response", "not_sent"}:
            raise ValueError("termination evidence is required")
        async with self.store.transaction() as c:
            a = await self._owned(c, reservation)
            if not a["compute_held"]:
                return
            if evidence == "completed_response" and a["state"] == "RESERVED":
                raise RuntimeConflict("direct attempt has not sent")
            await execute(c, """UPDATE runtime_direct_attempts SET compute_held=false,
                state=:state,backend_finished_at=now(),error_class=NULL WHERE attempt_id=:id""",
                state="FAILED_NOT_SENT" if evidence == "not_sent" else "FINISHED", id=reservation.attempt_id)
            await self.store._wake(c)

    async def unknown(self, reservation, reason):
        async with self.store.transaction() as c:
            a = await self._owned(c, reservation)
            if a["compute_held"]:
                await execute(c, """UPDATE runtime_direct_attempts SET state='UNKNOWN',
                    unknown_at=COALESCE(unknown_at,now()),error_class=:reason WHERE attempt_id=:id""",
                    reason=reason, id=reservation.attempt_id)

    async def reconcile_expired(self):
        await self.store.reconcile_expired()
