"""Process-local endpoint permits shared by direct and durable inference.

Only pool control and durable recovery read PostgreSQL. A direct enqueue, grant,
send, heartbeat, and settlement never execute SQL.
"""

import asyncio
import heapq
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isfinite
from time import monotonic
from uuid import uuid4

import httpx

from .contracts import AdmissionDenied, RuntimeConflict, parse_operation, parse_pool
from .direct import DirectReservation
from .store import execute, row, rows

log = logging.getLogger(__name__)
RANK = {"qa": 0, "user_task": 1, "background": 2, "maintenance": 3}


def root_workload(priority: str) -> str:
    return "user_task" if priority == "interactive" else priority


@dataclass
class _Pool:
    spec: object
    epoch: str
    health: str
    target: int
    group_id: str
    group_health: str
    group_ceiling: int


@dataclass
class _Waiter:
    identity: str
    tenant_id: str
    pool_id: str
    owner_id: str
    workload_class: str
    deadline: float
    queued_at: float
    kind: str


@dataclass
class _Permit:
    attempt_id: str
    pool_id: str
    owner_id: str
    workload_class: str
    engine_epoch: str
    kind: str
    lease_until: float
    state: str = "RESERVED"
    reservation: DirectReservation | None = None


class SlotCapacity:
    """Current fixed-slot implementation of the provider capacity interface."""

    @staticmethod
    def try_acquire(pool: _Pool, permits: dict[str, _Permit], pools: dict[str, _Pool],
                    workload_class: str) -> bool:
        if (pool.health != "HEALTHY" or pool.group_health != "HEALTHY"
                or pool.target <= 0):
            return False
        active = [p for p in permits.values() if p.pool_id == pool.spec.pool_id]
        endpoint_limit = (pool.spec.transport_limit if pool.spec.kind == "llm" else
                          min(pool.target, pool.spec.hard_ceiling))
        if len(active) >= endpoint_limit:
            return False
        if (pool.spec.kind == "llm" and workload_class != "qa" and
                sum(p.workload_class != "qa" for p in active)
                >= pool.spec.background_transport_limit):
            return False
        if pool.spec.kind != "llm":
            group_used = sum(1 for p in permits.values() if
                (other := pools.get(p.pool_id)) is not None and
                other.group_id == pool.group_id and other.spec.kind != "llm")
            if group_used >= pool.group_ceiling:
                return False
        return True

    @staticmethod
    def release(permits: dict[str, _Permit], attempt_id: str) -> None:
        permits.pop(attempt_id, None)


class MemoryScheduler:
    """Single runtime-api owner for all configured endpoint capacity."""

    def __init__(self, store, *, grace_seconds=0, endpoint_limit=1024,
                 tenant_limit=256, request_ttl_seconds=3600, slot_probes=None,
                 attempt_timeouts=None, restart_drain_seconds=None):
        self.store = store
        self.grace_seconds = grace_seconds
        self.endpoint_limit = endpoint_limit
        self.tenant_limit = tenant_limit
        self.request_ttl_seconds = request_ttl_seconds
        self.slot_probes = slot_probes or {}
        self.attempt_timeouts = attempt_timeouts or {}
        self.restart_drain_seconds = restart_drain_seconds or {}
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not isfinite(value) or value <= 0
               for value in self.restart_drain_seconds.values()):
            raise ValueError("restart_drain_seconds must be finite and positive")
        self._slot_clients = {}
        self._slot_idle = Counter()
        self._dirty_pools = set()
        self._old_owner_expires = {}
        self._ready_by_pool = {}
        self._slot_fallback_at = {}
        self._dirty_timers = []
        self._valid_until_warned_at = {}
        self.control_ready = True
        self.boot_generation = uuid4().hex
        self.pools: dict[str, _Pool] = {}
        self.waiters: dict[str, _Waiter] = {}
        self._queues: dict[str, list[tuple[int, float, str]]] = {}
        self._queue_counts = Counter()
        self._tenant_counts = Counter()
        self._durable_waiter_refs = Counter()
        self.permits: dict[str, _Permit] = {}
        self.direct_attempts: dict[str, _Permit] = {}
        self.request_identity: dict[str, tuple[object, str, float]] = {}
        self.generation = 0
        self._event = asyncio.Event()
        self._start_lock = asyncio.Lock()
        self._task = None
        self._started = False
        self._closing = False
        self.capacity = SlotCapacity()

    def _notify(self):
        self.generation += 1
        self._event.set()

    async def wait(self, generation, timeout=5):
        if self.generation != generation:
            return
        self._event.clear()
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
        except TimeoutError:
            pass

    async def start(self):
        async with self._start_lock:
            if self._started:
                return
            await self._refresh(initial=True)
            now = monotonic()
            for pool_id, pool in self.pools.items():
                if pool_id in self._dirty_pools:
                    timeout = self.restart_drain_seconds.get(
                        pool_id, self.attempt_timeouts.get(pool_id, 1800))
                    expiry = self._old_owner_expires[pool_id] + timedelta(seconds=timeout)
                    remaining = max(0, (expiry-datetime.now(UTC)).total_seconds())
                    self._slot_fallback_at[pool_id] = now + remaining
                    self._ready_by_pool[pool_id] = now + remaining
                    self._dirty_timers.append(asyncio.get_running_loop().call_later(
                        remaining, self._expire_dirty))
                    log.warning("pool %s blocks capacity after unclean runtime-api exit "
                                "until former owner lease plus %.1fs restart drain",
                                pool_id, timeout)
                else:
                    self._ready_by_pool[pool_id] = now + self.grace_seconds
            self._started = True
            self._expire_dirty()
            self._task = asyncio.create_task(self._maintain())

    async def close(self):
        self._closing = True
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        for timer in self._dirty_timers:
            timer.cancel()
        # A clean shutdown is evidence only for pools with no orphan direct
        # compute. Durable attempts remain independently represented in DB.
        if self._started:
            async with self.store.transaction() as c:
                for pool_id in self.pools:
                    if any(p.pool_id == pool_id and p.kind in {"direct", "legacy"}
                           for p in self.permits.values()):
                        continue
                    await execute(c, """DELETE FROM runtime_owners
                        WHERE owner_id=:id AND boot_generation=:boot""",
                        id="runtime-api-pool:"+pool_id, boot=self.boot_generation)
        for client in self._slot_clients.values():
            await client.aclose()

    async def _refresh(self, *, initial=False):
        """Bounded control-plane read; never called from a direct request."""
        async with self.store.transaction() as c:
            if initial:
                # rc17 direct owners had no endpoint identity. During rolling
                # deploy, refuse to race any still-live rc17 API process.
                legacy = await row(c, """SELECT owner_id FROM runtime_owners
                    WHERE owner_id LIKE 'direct-api-%' AND lease_expires_at>now() LIMIT 1""")
                if legacy:
                    raise RuntimeConflict("legacy direct API owner is still alive")
            records = await rows(c, """SELECT p.*,g.health AS group_health,
                g.hard_ceiling AS group_ceiling FROM runtime_pools p
                JOIN runtime_groups g USING(group_id)""")
            if initial:
                for item in records:
                    owner = "runtime-api-pool:" + item["pool_id"]
                    old = await row(c, """SELECT boot_generation,lease_expires_at
                        FROM runtime_owners WHERE owner_id=:id""",
                                    id=owner)
                    if old and old["boot_generation"] != self.boot_generation:
                        self._dirty_pools.add(item["pool_id"])
                        self._old_owner_expires[item["pool_id"]] = old["lease_expires_at"]
                    claimed = await row(c, """INSERT INTO runtime_owners
                        (owner_id,boot_generation,lease_expires_at)
                        VALUES (:id,:boot,now()+(:lease * interval '1 second'))
                        ON CONFLICT(owner_id) DO UPDATE SET
                            boot_generation=EXCLUDED.boot_generation,
                            lease_expires_at=EXCLUDED.lease_expires_at,updated_at=now()
                        WHERE runtime_owners.lease_expires_at<=now()
                            OR runtime_owners.boot_generation=EXCLUDED.boot_generation
                        RETURNING owner_id""", id=owner, boot=self.boot_generation,
                        lease=self.store.lease_seconds)
                    if not claimed:
                        raise RuntimeConflict(f"another runtime-api owns pool {item['pool_id']}")
                durable = await rows(c, """SELECT a.attempt_id,a.pool_id,a.owner_id,
                    a.engine_epoch,a.state,r.priority FROM runtime_attempts a
                    JOIN runtime_operations o USING(operation_id)
                    JOIN runtime_roots r USING(root_id) WHERE a.compute_held
                    AND a.state IN ('SEND_INTENT','ACTIVE','UNKNOWN')""")
                legacy_direct = await rows(c, """SELECT attempt_id,pool_id,owner_id,
                    engine_epoch,state,spec->>'workload_class' AS workload_class
                    FROM runtime_direct_attempts WHERE compute_held
                    AND state IN ('SEND_INTENT','UNKNOWN')""")
            else:
                durable = legacy_direct = []
                for item in records:
                    owner = "runtime-api-pool:" + item["pool_id"]
                    claimed = await row(c, """UPDATE runtime_owners SET
                        lease_expires_at=now()+(:lease * interval '1 second'),updated_at=now()
                        WHERE owner_id=:id AND boot_generation=:boot RETURNING owner_id""",
                        id=owner, boot=self.boot_generation, lease=self.store.lease_seconds)
                    if not claimed:
                        raise RuntimeConflict(f"runtime-api ownership lost for {item['pool_id']}")
        now = monotonic()
        for item in records:
            pool_id = item["pool_id"]
            previous = self.pools.get(pool_id)
            current = _Pool(parse_pool(item["spec"]), item["engine_epoch"], item["health"],
                            item["target"], item["group_id"], item["group_health"],
                            item["group_ceiling"])
            self.pools[pool_id] = current
            if (current.spec.valid_until is not None
                    and current.spec.valid_until <= datetime.now(UTC)
                    and now - self._valid_until_warned_at.get(pool_id, float("-inf")) >= 3600):
                log.warning("pool %s valid_until passed; capacity remains available", pool_id)
                self._valid_until_warned_at[pool_id] = now
            if previous and previous.epoch != current.epoch:
                for permit in list(self.permits.values()):
                    if permit.pool_id != pool_id or permit.engine_epoch == current.epoch:
                        continue
                    if permit.kind == "direct":
                        permit.state = ("FAILED_NOT_SENT" if permit.state == "RESERVED"
                                        else "FAILED_RECOVERABLE")
                        self.capacity.release(self.permits, permit.attempt_id)
                    # Durable settlement remains in PostgreSQL. Reconciliation
                    # below releases it only after compute_held becomes false.
                if current.health == "HEALTHY" and pool_id in self._dirty_pools:
                    self._ready_by_pool[pool_id] = now
                    self._dirty_pools.discard(pool_id)
        for item in durable:
            self.permits[item["attempt_id"]] = _Permit(
                item["attempt_id"], item["pool_id"], item["owner_id"],
                root_workload(item["priority"]), item["engine_epoch"],
                "durable", now + self.store.lease_seconds, item["state"])
        for item in legacy_direct:
            self.permits[item["attempt_id"]] = _Permit(
                item["attempt_id"], item["pool_id"], item["owner_id"],
                item["workload_class"] or "background", item["engine_epoch"],
                "legacy", float("inf"), item["state"])
        self._notify()

    async def _maintain(self):
        while not self._closing:
            try:
                await asyncio.sleep(max(1, min(2, self.store.lease_seconds / 3)))
                await self._refresh()
                self.control_ready = True
                await self._probe_slots()
                await self._reconcile_durable()
                self._expire_requests()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("in-memory capacity owner lost control-plane refresh")
                self.control_ready = False
                await asyncio.sleep(1)

    async def _probe_slots(self):
        for pool_id in list(self._dirty_pools):
            probe = self.slot_probes.get(pool_id)
            if not probe or self.pools[pool_id].spec.kind != "llm":
                continue
            base_url, alias, token = probe
            client = self._slot_clients.get(pool_id)
            if client is None:
                client = httpx.AsyncClient(base_url=base_url.rstrip("/") + "/",
                    headers={"Authorization": f"Bearer {token}"}, timeout=3,
                    follow_redirects=False, transport=httpx.AsyncHTTPTransport(retries=0))
                self._slot_clients[pool_id] = client
            try:
                response = await client.get("slots", params={"model": alias})
                slots = response.json() if response.status_code == 200 else None
                idle = (isinstance(slots, list) and bool(slots) and
                        all(isinstance(slot, dict) and slot.get("is_processing") is False
                            for slot in slots))
            except (httpx.HTTPError, ValueError):
                idle = False
            self._slot_idle[pool_id] = self._slot_idle[pool_id] + 1 if idle else 0
            if self._slot_idle[pool_id] >= 2:
                self._ready_by_pool[pool_id] = monotonic()
                self._dirty_pools.discard(pool_id)
                self._notify()
        self._expire_dirty()

    def _expire_dirty(self):
        for pool_id in list(self._dirty_pools):
            if monotonic() >= self._slot_fallback_at[pool_id]:
                self._dirty_pools.discard(pool_id)
                log.warning("pool %s resumes after former owner lease plus attempt timeout",
                            pool_id)
                self._notify()

    async def _reconcile_durable(self):
        active = [p.attempt_id for p in self.permits.values() if
                  p.kind in {"durable", "legacy"}]
        if not active:
            return
        async with self.store.engine.connect() as c:
            held = await rows(c, """SELECT attempt_id FROM runtime_attempts
                WHERE attempt_id=ANY(CAST(:ids AS text[])) AND compute_held
                UNION ALL SELECT attempt_id FROM runtime_direct_attempts
                WHERE attempt_id=ANY(CAST(:ids AS text[])) AND compute_held""",
                ids=active)
        live = {r["attempt_id"] for r in held}
        for attempt_id in active:
            if attempt_id not in live:
                self.capacity.release(self.permits, attempt_id)
                self._notify()

    def _expire_requests(self):
        now = monotonic()
        for identity, (_, _, expires) in list(self.request_identity.items()):
            if expires <= now and identity not in self.waiters:
                self.request_identity.pop(identity, None)
                for attempt_id, permit in list(self.direct_attempts.items()):
                    if (permit.reservation is not None and
                        permit.reservation.request.request_id == identity and
                        attempt_id not in self.permits):
                        self.direct_attempts.pop(attempt_id, None)
        for identity, waiter in list(self.waiters.items()):
            if waiter.deadline <= now:
                self._remove_waiter(identity)

    def _pool(self, pool_id):
        pool = self.pools.get(pool_id)
        if pool is None:
            raise AdmissionDenied("qualified inference endpoint unavailable", retryable=True)
        return pool

    def _first(self, pool_id):
        queue = self._queues.get(pool_id, [])
        while queue:
            _, queued_at, identity = queue[0]
            waiter = self.waiters.get(identity)
            if (waiter is not None and waiter.pool_id == pool_id
                    and waiter.queued_at == queued_at):
                if waiter.deadline <= monotonic():
                    self._remove_waiter(identity)
                    continue
                return waiter
            heapq.heappop(queue)
        return None

    def _add_waiter(self, waiter):
        self.waiters[waiter.identity] = waiter
        heapq.heappush(self._queues.setdefault(waiter.pool_id, []),
                       (RANK[waiter.workload_class], waiter.queued_at, waiter.identity))
        if waiter.kind == "direct":
            self._queue_counts[waiter.pool_id] += 1
            self._tenant_counts[(waiter.pool_id, waiter.tenant_id)] += 1
        self._notify()

    def _remove_waiter(self, identity):
        waiter = self.waiters.pop(identity, None)
        if waiter is not None:
            if waiter.kind == "direct":
                self._queue_counts[waiter.pool_id] -= 1
                self._tenant_counts[(waiter.pool_id, waiter.tenant_id)] -= 1
            self._notify()

    async def enqueue(self, request, pool_id, owner_id):
        pool = self._pool(pool_id)
        now = monotonic()
        identity = request.request_id
        known = self.request_identity.get(identity)
        if known:
            if known[0] != request or known[1] != owner_id:
                raise RuntimeConflict("direct waiting identity conflict")
            existing = self.waiters.get(identity)
            if existing:
                return existing.queued_at
            previous = [p for p in self.direct_attempts.values() if
                        p.reservation and p.reservation.request.request_id == identity]
            if previous:
                latest = max(previous, key=lambda p: p.reservation.generation)
                if latest.state not in {"FAILED_NOT_SENT", "FAILED_RECOVERABLE"}:
                    raise RuntimeConflict("direct request already accepted")
        if request.deadline <= datetime.now(UTC):
            raise AdmissionDenied("inference request deadline exceeded", retryable=True)
        if (self._queue_counts[pool_id] >= self.endpoint_limit or
            self._tenant_counts[(pool_id, request.tenant_id)] >= self.tenant_limit):
            raise AdmissionDenied("inference waiting buffer full", retryable=True)
        self._validate(request, pool)
        deadline = now + (request.deadline - datetime.now(UTC)).total_seconds()
        self._add_waiter(_Waiter(identity, request.tenant_id, pool_id, owner_id,
                                request.workload_class, deadline, now, "direct"))
        self.request_identity[identity] = (request, owner_id,
                                           now + self.request_ttl_seconds)
        return now

    def _validate(self, request, pool):
        profile = pool.spec
        if (request.kind != profile.kind or request.model_profile != profile.model_profile
            or (request.model_revision is not None
                and request.model_revision != profile.model_revision)
            or request.capacity_profile_id != profile.profile_id
            or request.request_bound > profile.request_limit
            or request.batch_size > getattr(profile, "max_batch_size", 1)):
            raise AdmissionDenied("direct request exceeds its qualified pool profile")

    async def leave(self, request_id, owner_id):
        waiter = self.waiters.get(request_id)
        if waiter and waiter.owner_id == owner_id:
            self._remove_waiter(request_id)

    async def reserve(self, request, pool_id, owner_id, *, previous_attempt_id=None):
        self._expire_dirty()
        pool = self._pool(pool_id)
        waiter = self.waiters.get(request.request_id)
        if waiter is None or waiter.owner_id != owner_id:
            return None
        if (request.deadline <= datetime.now(UTC) or not self.control_ready
            or monotonic() < self._ready_by_pool.get(pool_id, float("inf"))):
            return None
        self._validate(request, pool)
        if (request.expected_engine_epoch is not None
                and request.expected_engine_epoch != pool.epoch
                and previous_attempt_id is None):
            raise RuntimeConflict("proxy has an older engine epoch")
        generation = 0
        if previous_attempt_id is not None:
            old = self.direct_attempts.get(previous_attempt_id)
            if (old is None or old.reservation is None or old.reservation.request != request
                or old.owner_id != owner_id or old.pool_id != pool_id):
                raise RuntimeConflict("recovery identity conflict")
            if old.state not in {"FAILED_NOT_SENT", "FAILED_RECOVERABLE"}:
                raise RuntimeConflict("inference termination is not proven")
            generation = old.reservation.generation + 1
            if generation > 2:
                raise AdmissionDenied("inference recovery limit reached")
            if old.state == "FAILED_RECOVERABLE" and old.engine_epoch == pool.epoch:
                return None
        if self._first(pool_id) is not waiter or not self.capacity.try_acquire(
                pool, self.permits, self.pools, request.workload_class):
            return None
        self._remove_waiter(request.request_id)
        result = DirectReservation(attempt_id=uuid4().hex, request=request,
                                   pool_id=pool_id, engine_epoch=pool.epoch,
                                   owner_id=owner_id, generation=generation)
        permit = _Permit(result.attempt_id, pool_id, owner_id, request.workload_class,
                         pool.epoch, "direct", float("inf"), reservation=result)
        self.permits[result.attempt_id] = permit
        self.direct_attempts[result.attempt_id] = permit
        self._notify()
        return result

    async def retry_confirmed(self, request, pool_id, owner_id, previous_attempt_id):
        return await self.reserve(request, pool_id, owner_id,
                                  previous_attempt_id=previous_attempt_id)

    def _direct(self, reservation):
        attempt = self.direct_attempts.get(reservation.attempt_id)
        if not attempt or attempt.reservation != reservation:
            raise RuntimeConflict("stale direct request ownership")
        return attempt

    async def mark_send(self, reservation):
        attempt = self._direct(reservation)
        if (attempt.state != "RESERVED" or attempt.attempt_id not in self.permits
            or reservation.request.deadline <= datetime.now(UTC)):
            raise RuntimeConflict("direct attempt cannot send")
        attempt.state = "SEND_INTENT"

    async def heartbeat(self, reservation):
        attempt = self._direct(reservation)
        return (attempt.attempt_id in self.permits and
                attempt.state in {"RESERVED", "SEND_INTENT"} and
                reservation.request.deadline > datetime.now(UTC))

    async def finish(self, reservation, evidence="completed_response"):
        if evidence not in {"completed_response", "not_sent"}:
            raise ValueError("termination evidence is required")
        attempt = self._direct(reservation)
        if evidence == "completed_response" and attempt.state == "RESERVED":
            raise RuntimeConflict("direct attempt has not sent")
        if attempt.attempt_id not in self.permits:
            return
        attempt.state = "FINISHED" if evidence == "completed_response" else "FAILED_NOT_SENT"
        self.capacity.release(self.permits, attempt.attempt_id)
        self._notify()

    async def unknown(self, reservation, reason):
        attempt = self._direct(reservation)
        if attempt.attempt_id in self.permits:
            attempt.state = "UNKNOWN"

    async def recovery_statuses(self, attempt_ids):
        result = {}
        for attempt_id in attempt_ids:
            attempt = self.direct_attempts.get(attempt_id)
            if attempt is None:
                continue
            pool = self.pools.get(attempt.pool_id)
            if pool:
                result[attempt_id] = {"state": attempt.state,
                    "engine_epoch": attempt.engine_epoch, "current_epoch": pool.epoch,
                    "health": pool.health, "target": pool.target}
        return result

    async def _durable_waiter_valid(self, attempt_id):
        async with self.store.engine.connect() as c:
            state = await row(c, """SELECT a.compute_held,a.state,
                o.state AS operation_state,r.state AS root_state,
                r.cancel_requested,r.deadline FROM runtime_attempts a
                JOIN runtime_operations o USING(operation_id)
                JOIN runtime_roots r USING(root_id) WHERE a.attempt_id=:id""",
                id=attempt_id)
        return bool(state and state["state"] == "RESERVED"
                    and state["operation_state"] == "EXECUTING"
                    and state["root_state"] == "RUNNING" and not state["cancel_requested"]
                    and state["deadline"] > datetime.now(UTC))

    async def durable(self, attempt_id, pool_id, owner_id, engine_epoch,
                      workload_class, deadline):
        if workload_class not in RANK:
            raise ValueError("invalid root priority")
        # The internal caller is authenticated, and the durable ledger is the
        # authority for attempt identity and root class.
        async with self.store.engine.connect() as c:
            actual = await row(c, """SELECT a.pool_id,a.owner_id,a.engine_epoch,
                a.compute_held,a.state,a.created_at,o.created_at AS operation_created_at,
                r.priority,
                r.deadline AS root_deadline,o.spec AS operation_spec FROM runtime_attempts a
                JOIN runtime_operations o USING(operation_id)
                JOIN runtime_roots r USING(root_id) WHERE a.attempt_id=:id""", id=attempt_id)
        if (not actual or actual["pool_id"] != pool_id
            or actual["owner_id"] != owner_id or actual["engine_epoch"] != engine_epoch
            or root_workload(actual["priority"]) != workload_class
            or actual["state"] not in {"RESERVED", "SEND_INTENT", "ACTIVE"}
            or (actual["state"] != "RESERVED" and not actual["compute_held"])):
            raise RuntimeConflict("durable permit identity conflicts with ledger")
        operation = parse_operation(actual["operation_spec"])
        authoritative_deadline = min(actual["root_deadline"],
            operation.deadline or actual["root_deadline"],
            actual["created_at"] + timedelta(seconds=operation.attempt_timeout_seconds))
        if deadline.tzinfo is None or deadline > authoritative_deadline:
            raise RuntimeConflict("durable permit deadline exceeds ledger deadline")
        pool = self._pool(pool_id)
        if pool.epoch != engine_epoch:
            raise RuntimeConflict("durable attempt engine epoch changed before permit")
        if (operation.model_profile != pool.spec.model_profile or
            (operation.capacity_profile_id is not None and
             operation.capacity_profile_id != pool.spec.profile_id)):
            raise RuntimeConflict("durable attempt profile changed before permit")
        existing = self.permits.get(attempt_id)
        if existing:
            if (existing.kind != "durable" or existing.pool_id != pool_id
                or existing.owner_id != owner_id or existing.engine_epoch != engine_epoch):
                raise RuntimeConflict("durable permit ownership conflict")
            existing.lease_until = monotonic() + self.store.lease_seconds
            return existing
        waiter = self.waiters.get(attempt_id)
        if waiter is None:
            now = monotonic()
            observed_at = datetime.now(UTC)
            waiter = _Waiter(attempt_id, "", pool_id, owner_id, workload_class,
                             now + max(0, (deadline-observed_at).total_seconds()),
                             now - max(0, (observed_at-actual["operation_created_at"]).total_seconds()),
                             "durable")
            self._add_waiter(waiter)
        elif (waiter.kind != "durable" or waiter.pool_id != pool_id or
              waiter.owner_id != owner_id or waiter.workload_class != workload_class):
            raise RuntimeConflict("durable waiter identity conflict")
        self._durable_waiter_refs[attempt_id] += 1
        next_validation = monotonic() + 1
        try:
            while True:
                self._expire_dirty()
                pool = self._pool(pool_id)
                if pool.epoch != engine_epoch:
                    raise RuntimeConflict("durable attempt engine epoch changed before permit")
                if (operation.model_profile != pool.spec.model_profile or
                    (operation.capacity_profile_id is not None and
                     operation.capacity_profile_id != pool.spec.profile_id)):
                    raise RuntimeConflict("durable attempt profile changed before permit")
                existing = self.permits.get(attempt_id)
                if existing is not None:
                    return existing
                if monotonic() >= next_validation:
                    if not await self._durable_waiter_valid(attempt_id):
                        raise RuntimeConflict("durable attempt no longer dispatchable")
                    next_validation = monotonic() + 1
                remaining = waiter.deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError("durable permit deadline exceeded")
                if (self.control_ready and
                    monotonic() >= self._ready_by_pool.get(pool_id, float("inf"))
                    and pool.epoch == engine_epoch
                    and self._first(pool_id) is waiter and self.capacity.try_acquire(
                        pool, self.permits, self.pools, workload_class)):
                    self._remove_waiter(attempt_id)
                    permit = _Permit(attempt_id, pool_id, owner_id, workload_class,
                                     engine_epoch, "durable",
                                     monotonic()+self.store.lease_seconds)
                    self.permits[attempt_id] = permit
                    self._notify()
                    return permit
                generation = self.generation
                await self.wait(generation, min(remaining, 1))
        finally:
            self._durable_waiter_refs[attempt_id] -= 1
            if self._durable_waiter_refs[attempt_id] <= 0:
                if self.waiters.get(attempt_id) is waiter:
                    self._remove_waiter(attempt_id)
                self._durable_waiter_refs.pop(attempt_id, None)

    def durable_heartbeat(self, attempt_id, owner_id):
        permit = self.permits.get(attempt_id)
        if not permit or permit.kind != "durable" or permit.owner_id != owner_id:
            raise RuntimeConflict("durable permit not owned")
        permit.lease_until = monotonic() + self.store.lease_seconds

    async def durable_release(self, attempt_id, owner_id):
        permit = self.permits.get(attempt_id)
        if permit is None:
            return
        if permit.kind != "durable" or permit.owner_id != owner_id:
            raise RuntimeConflict("durable permit not owned")
        async with self.store.engine.connect() as c:
            active = await row(c, "SELECT compute_held FROM runtime_attempts WHERE attempt_id=:id",
                               id=attempt_id)
        if active and active["compute_held"]:
            raise RuntimeConflict("compute termination is not proven")
        self.capacity.release(self.permits, attempt_id)
        self._notify()

    def metrics(self):
        queued = Counter((w.pool_id, w.workload_class) for w in self.waiters.values()
                         if w.kind == "direct")
        held = Counter((p.pool_id, p.workload_class) for p in self.permits.values()
                       if p.kind == "direct")
        return queued, held

    def dirty_blocked(self):
        self._expire_dirty()
        return {pool_id: int(pool_id in self._dirty_pools)
                for pool_id in self.pools}
