from prometheus_client import CollectorRegistry, Gauge, generate_latest

from .store import Store, row, rows


async def snapshot(store: Store) -> bytes:
    registry = CollectorRegistry()
    states = Gauge("intramind_runtime_operations", "Persisted operation count", ["state"], registry=registry)
    held = Gauge("intramind_runtime_compute_held", "Attempts without termination evidence", ["pool"], registry=registry)
    target = Gauge("intramind_runtime_admission_target", "Current pool target", ["pool"], registry=registry)
    unknown_age = Gauge("intramind_runtime_unknown_oldest_seconds", "Oldest unreconciled attempt", registry=registry)
    outbox_age = Gauge("intramind_runtime_outbox_oldest_seconds", "Oldest undelivered event", registry=registry)
    async with store.engine.connect() as c:
        for r in await rows(c, "SELECT state,count(*) AS n FROM runtime_operations GROUP BY state"):
            states.labels(r["state"]).set(r["n"])
        for r in await rows(c, """SELECT p.pool_id,p.target,count(a.attempt_id) AS n
            FROM runtime_pools p LEFT JOIN runtime_attempts a ON a.pool_id=p.pool_id AND a.compute_held
            GROUP BY p.pool_id,p.target"""):
            held.labels(r["pool_id"]).set(r["n"])
            target.labels(r["pool_id"]).set(r["target"])
        value = await row(c, """SELECT COALESCE(EXTRACT(EPOCH FROM now()-min(unknown_at)),0) AS age
            FROM runtime_attempts WHERE state='UNKNOWN'""")
        unknown_age.set(value["age"])
        value = await row(c, """SELECT COALESCE(EXTRACT(EPOCH FROM now()-min(created_at)),0) AS age
            FROM runtime_outbox WHERE delivered_at IS NULL""")
        outbox_age.set(value["age"])
    return generate_latest(registry)
