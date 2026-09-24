"""Foreground inference shares pool authority without workflow or artifact I/O."""

from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import uuid4

from pydantic import Field, field_validator

from .contracts import AdmissionDenied, Contract, RuntimeConflict, parse_pool
from .store import Store, endpoint_usage, execute, row, rows


class DirectRequest(Contract):
    request_id: str = Field(min_length=1, max_length=200)
    tenant_id: str = Field(min_length=1, max_length=200)
    payload_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_profile: str = Field(min_length=1, max_length=120)
    model_revision: str | None = Field(default=None, min_length=1)
    expected_engine_epoch: str | None = Field(default=None, min_length=1)
    capacity_profile_id: str = Field(min_length=1, max_length=200)
    kind: Literal["llm", "embedding", "speech", "rerank"] = "llm"
    request_bound: int = Field(gt=0, strict=True)
    batch_size: int = Field(default=1, gt=0, le=256, strict=True)
    deadline: datetime
    workload_class: Literal["qa", "user_task", "background", "maintenance"] = "qa"
    logical_request_id: str | None = Field(default=None, max_length=200)

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
    generation: int = 0


class DirectAdmissions:
    """Only trusted inference adapters report termination; expiry never proves it."""

    def __init__(self, store: Store):
        self.store = store

    async def enqueue(self, request: DirectRequest, pool_id: str, owner_id: str,
                      *, endpoint_limit: int = 1024, tenant_limit: int = 256):
        """A pending connection writes once, regardless of how long it waits."""
        async with self.store.transaction() as c:
            pool = await row(c, "SELECT pool_id FROM runtime_pools WHERE pool_id=:id FOR UPDATE", id=pool_id)
            if not pool:
                raise AdmissionDenied("qualified inference endpoint unavailable", retryable=True)
            existing = await row(c, "SELECT * FROM runtime_direct_waiters WHERE request_id=:id",
                                 id=request.request_id)
            if existing:
                if (existing["tenant_id"] != request.tenant_id or existing["pool_id"] != pool_id
                    or existing["owner_id"] != owner_id or existing["workload_class"] != request.workload_class):
                    raise RuntimeConflict("direct waiting identity conflict")
                return
            counts = await row(c, """SELECT count(*) AS endpoint,
                count(*) FILTER(WHERE tenant_id=:tenant) AS tenant
                FROM runtime_direct_waiters WHERE pool_id=:pool AND deadline>now()""",
                tenant=request.tenant_id, pool=pool_id)
            if counts["endpoint"] >= endpoint_limit or counts["tenant"] >= tenant_limit:
                raise AdmissionDenied("inference waiting buffer full", retryable=True)
            await execute(c, """INSERT INTO runtime_direct_waiters
                (request_id,tenant_id,pool_id,owner_id,workload_class,deadline)
                VALUES (:id,:tenant,:pool,:owner,:workload,:deadline)""",
                id=request.request_id, tenant=request.tenant_id, pool=pool_id,
                owner=owner_id, workload=request.workload_class, deadline=request.deadline)
            await self.store._wake(c)

    async def leave(self, request_id: str, owner_id: str):
        async with self.store.transaction() as c:
            changed = await execute(c, """DELETE FROM runtime_direct_waiters
                WHERE request_id=:id AND owner_id=:owner""", id=request_id, owner=owner_id)
            if changed.rowcount:
                await self.store._wake(c)

    async def recovery_statuses(self, attempt_ids: list[str]):
        if not attempt_ids:
            return {}
        async with self.store.engine.connect() as c:
            found = await rows(c, """SELECT a.attempt_id,a.state,a.engine_epoch,
                p.engine_epoch AS current_epoch,p.health,p.target
                FROM runtime_direct_attempts a JOIN runtime_pools p USING(pool_id)
                WHERE a.attempt_id=ANY(CAST(:ids AS text[]))""", ids=attempt_ids[:1000])
            return {a["attempt_id"]: dict(a) for a in found}

    async def retry_confirmed(self, request: DirectRequest, pool_id: str, owner_id: str,
                              previous_attempt_id: str):
        """Only an unsent attempt or host-confirmed stopped epoch can be resent."""
        async with self.store.transaction() as c:
            pool = await row(c, "SELECT * FROM runtime_pools WHERE pool_id=:id FOR UPDATE", id=pool_id)
            previous = await row(c, """SELECT * FROM runtime_direct_attempts
                WHERE attempt_id=:id FOR UPDATE""", id=previous_attempt_id)
            if (not previous or previous["spec"] != request.model_dump(mode="json")
                or previous["owner_id"] != owner_id or previous["pool_id"] != pool_id):
                raise RuntimeConflict("recovery identity conflict")
            if previous["state"] not in {"FAILED_NOT_SENT", "FAILED_RECOVERABLE"}:
                raise RuntimeConflict("inference termination is not proven")
            if previous["generation"] >= 2:
                raise AdmissionDenied("inference recovery limit reached")
            if (not pool or pool["health"] != "HEALTHY" or pool["target"] == 0
                or pool["valid_until"] <= datetime.now(UTC)):
                return None
            profile = parse_pool(pool["spec"])
            if (previous["state"] == "FAILED_RECOVERABLE"
                and previous["engine_epoch"] == pool["engine_epoch"]):
                return None
            if (profile.kind != request.kind or profile.profile_id != request.capacity_profile_id
                or profile.model_profile != request.model_profile
                or (request.model_revision is not None
                    and profile.model_revision != request.model_revision)):
                raise RuntimeConflict("qualified inference profile changed during recovery")
            used = await row(c, """SELECT count(*) AS n FROM runtime_inflight_attempts
                WHERE pool_id=:pool AND compute_held""", pool=pool_id)
            limit = profile.transport_limit if profile.kind == "llm" else min(
                pool["target"], pool["hard_ceiling"])
            if used["n"] >= limit:
                return None
            if profile.kind == "llm" and request.workload_class != "qa":
                usage = await endpoint_usage(c, pool_id)
                if usage["lower_class"] >= profile.background_transport_limit:
                    return None
            first = await row(c, """SELECT request_id FROM runtime_direct_waiters
                WHERE pool_id=:pool AND deadline>now()
                ORDER BY CASE workload_class WHEN 'qa' THEN 0 WHEN 'user_task' THEN 1
                    WHEN 'background' THEN 2 ELSE 3 END,created_at,request_id LIMIT 1""",
                pool=pool_id)
            if first and first["request_id"] != request.request_id:
                return None
            claimed = await row(c, """DELETE FROM runtime_direct_waiters
                WHERE request_id=:id AND owner_id=:owner RETURNING 1""",
                id=request.request_id, owner=owner_id)
            if not claimed:
                return None
            latest = await row(c, """SELECT attempt_id FROM runtime_direct_attempts
                WHERE tenant_id=:tenant AND request_id=:request ORDER BY generation DESC LIMIT 1""",
                tenant=request.tenant_id, request=request.request_id)
            if latest["attempt_id"] != previous_attempt_id:
                raise RuntimeConflict("newer inference generation already owns request")
            result = DirectReservation(attempt_id=str(uuid4()), request=request, pool_id=pool_id,
                engine_epoch=pool["engine_epoch"], owner_id=owner_id,
                generation=previous["generation"]+1)
            await execute(c, """INSERT INTO runtime_direct_attempts
                (attempt_id,request_id,tenant_id,spec,pool_id,engine_epoch,owner_id,
                 lease_expires_at,deadline,generation)
                VALUES (:id,:request,:tenant,CAST(:spec AS jsonb),:pool,:epoch,:owner,
                    now()+(:lease * interval '1 second'),:deadline,:generation)""",
                id=result.attempt_id, request=request.request_id, tenant=request.tenant_id,
                spec=request.model_dump_json(), pool=pool_id, epoch=result.engine_epoch,
                owner=owner_id, lease=self.store.lease_seconds, deadline=request.deadline,
                generation=result.generation)
            await self.store._wake(c)
            return result

    async def reserve(self, request: DirectRequest, pool_id: str, owner_id: str):
        if not owner_id:
            raise ValueError("executor identity is required")
        async with self.store.transaction() as c:
            observed = (await row(c, "SELECT clock_timestamp() AS now"))["now"]
            initial = await row(c, "SELECT group_id,spec FROM runtime_pools WHERE pool_id=:id", id=pool_id)
            if initial and parse_pool(initial["spec"]).kind != "llm":
                await execute(c, "SELECT group_id FROM runtime_groups WHERE group_id=:id FOR UPDATE",
                              id=initial["group_id"])
            pool = await row(c, "SELECT * FROM runtime_pools WHERE pool_id=:id FOR UPDATE", id=pool_id)
            previous = await row(c, """SELECT * FROM runtime_direct_attempts
                WHERE tenant_id=:tenant AND request_id=:request
                ORDER BY generation DESC LIMIT 1""",
                tenant=request.tenant_id, request=request.request_id)
            if previous:
                if (previous["spec"] != request.model_dump(mode="json") or previous["owner_id"] != owner_id
                    or previous["pool_id"] != pool_id):
                    raise RuntimeConflict("direct request identity already has different ownership or content")
                if previous["state"] != "RESERVED" or min(previous["deadline"], previous["lease_expires_at"]) <= observed:
                    return None
                claimed = await row(c, """DELETE FROM runtime_direct_waiters
                    WHERE request_id=:id AND owner_id=:owner RETURNING 1""",
                    id=request.request_id, owner=owner_id)
                if not claimed:
                    return None
                return DirectReservation(attempt_id=previous["attempt_id"], request=request,
                    pool_id=pool_id, engine_epoch=previous["engine_epoch"], owner_id=owner_id,
                    generation=previous["generation"])
            if (not pool or pool["health"] != "HEALTHY" or pool["valid_until"] <= observed
                or request.deadline <= observed or pool["target"] == 0):
                return None
            if (request.expected_engine_epoch is not None
                and request.expected_engine_epoch != pool["engine_epoch"]):
                raise RuntimeConflict("proxy has an older engine epoch")
            profile = parse_pool(pool["spec"])
            if initial and (pool["group_id"] != initial["group_id"]
                or profile.kind != parse_pool(initial["spec"]).kind):
                return None
            if (request.kind != profile.kind or request.model_profile != profile.model_profile
                or (request.model_revision is not None
                    and request.model_revision != profile.model_revision)
                or request.capacity_profile_id != profile.profile_id
                or request.request_bound > profile.request_limit
                or request.batch_size > getattr(profile, "max_batch_size", 1)):
                raise AdmissionDenied("direct request exceeds its qualified pool profile")
            group = await row(c, "SELECT * FROM runtime_groups WHERE group_id=:id", id=pool["group_id"])
            used = await row(c, """SELECT count(*) AS group_used,
                count(*) FILTER (WHERE a.pool_id=:pool) AS pool_used FROM runtime_inflight_attempts a
                JOIN runtime_pools p USING(pool_id) WHERE a.compute_held AND p.group_id=:group
                AND (:llm OR COALESCE(p.spec->>'kind','llm')<>'llm')""",
                pool=pool_id, group=pool["group_id"], llm=profile.kind == "llm")
            endpoint_limit = profile.transport_limit if profile.kind == "llm" else min(
                pool["target"], pool["hard_ceiling"])
            if (group["health"] != "HEALTHY"
                or (profile.kind != "llm" and used["group_used"] >= group["hard_ceiling"])
                or used["pool_used"] >= endpoint_limit):
                return None
            if profile.kind == "llm" and request.workload_class != "qa":
                usage = await endpoint_usage(c, pool_id)
                if usage["lower_class"] >= profile.background_transport_limit:
                    return None
            first = await row(c, """SELECT request_id FROM runtime_direct_waiters
                WHERE pool_id=:pool AND deadline>now()
                ORDER BY CASE workload_class WHEN 'qa' THEN 0 WHEN 'user_task' THEN 1
                    WHEN 'background' THEN 2 ELSE 3 END,created_at,request_id LIMIT 1""",
                pool=pool_id)
            if first and first["request_id"] != request.request_id:
                return None
            claimed = await row(c, """DELETE FROM runtime_direct_waiters
                WHERE request_id=:id AND owner_id=:owner RETURNING 1""",
                id=request.request_id, owner=owner_id)
            if not claimed:
                return None
            result = DirectReservation(attempt_id=str(uuid4()), request=request, pool_id=pool_id,
                                       engine_epoch=pool["engine_epoch"], owner_id=owner_id)
            await execute(c, """INSERT INTO runtime_direct_attempts
                (attempt_id,request_id,tenant_id,spec,pool_id,engine_epoch,owner_id,lease_expires_at,deadline)
                VALUES (:id,:request,:tenant,CAST(:spec AS jsonb),:pool,:epoch,:owner,:lease,:deadline)""",
                id=result.attempt_id, request=request.request_id, tenant=request.tenant_id,
                spec=request.model_dump_json(), pool=pool_id, epoch=result.engine_epoch, owner=owner_id,
                lease=observed+timedelta(seconds=self.store.lease_seconds), deadline=request.deadline)
            await self.store._wake(c)
            return result

    async def _owned(self, c, reservation):
        attempt = await row(c, "SELECT *,clock_timestamp() AS now FROM runtime_direct_attempts WHERE attempt_id=:id FOR UPDATE",
                            id=reservation.attempt_id)
        if (not attempt or attempt["owner_id"] != reservation.owner_id
            or attempt["spec"] != reservation.request.model_dump(mode="json")
            or attempt["pool_id"] != reservation.pool_id or attempt["engine_epoch"] != reservation.engine_epoch):
            raise RuntimeConflict("stale direct request ownership")
        return attempt

    async def mark_send(self, reservation):
        async with self.store.transaction() as c:
            a = await self._owned(c, reservation)
            if (a["state"] != "RESERVED" or a["deadline"] <= a["now"]
                or (a["lease_expires_at"] <= a["now"]
                    and not await self.store.owner_alive(c, reservation.owner_id))):
                raise RuntimeConflict("direct attempt cannot send")
            await execute(c, """UPDATE runtime_direct_attempts SET state='SEND_INTENT',send_intent_at=now()
                WHERE attempt_id=:id""", id=reservation.attempt_id)

    async def heartbeat(self, reservation):
        async with self.store.transaction() as c:
            a = await self._owned(c, reservation)
            owner_alive = await self.store.owner_alive(c, reservation.owner_id)
            if (a["state"] not in {"RESERVED", "SEND_INTENT"}
                or a["deadline"] <= a["now"]
                or (not owner_alive and a["lease_expires_at"] <= a["now"])):
                return False
            if not owner_alive:
                # Legacy owner without process-level lease; old workers stay
                # compatible until their live requests have drained.
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
