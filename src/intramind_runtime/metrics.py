from datetime import UTC, datetime

from prometheus_client import CollectorRegistry, Gauge, generate_latest

from .store import Store, row, rows


async def snapshot(store: Store, scheduler=None) -> bytes:
    registry = CollectorRegistry()
    states = Gauge("intramind_runtime_operations", "Persisted operation count", ["state"], registry=registry)
    held = Gauge("intramind_runtime_compute_held", "Attempts without termination evidence", ["pool"], registry=registry)
    target = Gauge("intramind_runtime_admission_target", "Current pool target", ["pool"], registry=registry)
    valid_until = Gauge("intramind_runtime_pool_valid_until_seconds",
                        "Seconds until the advisory pool review date (negative when overdue)",
                        ["pool"], registry=registry)
    unknown_age = Gauge("intramind_runtime_unknown_oldest_seconds", "Oldest unreconciled attempt", registry=registry)
    outbox_age = Gauge("intramind_runtime_outbox_oldest_seconds", "Oldest undelivered event", registry=registry)
    direct_queue = Gauge("intramind_runtime_direct_waiting", "RAM direct queue depth",
                         ["pool", "workload_class"], registry=registry)
    direct_inflight = Gauge("intramind_runtime_direct_inflight", "RAM direct permits",
                            ["pool", "workload_class"], registry=registry)
    dirty_recovery = Gauge("intramind_runtime_dirty_recovery_blocked",
                           "Pool blocked after an unclean runtime-api exit",
                           ["pool"], registry=registry)
    async with store.engine.connect() as c:
        for r in await rows(c, "SELECT state,count(*) AS n FROM runtime_operations GROUP BY state"):
            states.labels(r["state"]).set(r["n"])
        for r in await rows(c, """SELECT p.pool_id,p.target,p.valid_until,count(a.attempt_id) AS n
            FROM runtime_pools p LEFT JOIN runtime_attempts a ON a.pool_id=p.pool_id AND a.compute_held
            GROUP BY p.pool_id,p.target,p.valid_until"""):
            count = (sum(p.pool_id == r["pool_id"] for p in scheduler.permits.values())
                     if scheduler is not None else r["n"])
            held.labels(r["pool_id"]).set(count)
            target.labels(r["pool_id"]).set(r["target"])
            if r["valid_until"] is not None:
                valid_until.labels(r["pool_id"]).set(
                    (r["valid_until"] - datetime.now(UTC)).total_seconds())
        value = await row(c, """SELECT COALESCE(EXTRACT(EPOCH FROM now()-min(unknown_at)),0) AS age
            FROM runtime_attempts WHERE state='UNKNOWN'""")
        unknown_age.set(value["age"])
        value = await row(c, """SELECT COALESCE(EXTRACT(EPOCH FROM now()-min(created_at)),0) AS age
            FROM runtime_outbox WHERE delivered_at IS NULL""")
        outbox_age.set(value["age"])
    if scheduler is not None:
        queued, inflight = scheduler.metrics()
        for (pool_id, workload), count in queued.items():
            direct_queue.labels(pool_id, workload).set(count)
        for (pool_id, workload), count in inflight.items():
            direct_inflight.labels(pool_id, workload).set(count)
        for pool_id, blocked in scheduler.dirty_blocked().items():
            dirty_recovery.labels(pool_id).set(blocked)
    return generate_latest(registry)
