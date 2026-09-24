import asyncio

from intramind_runtime.artifacts import tenant_prefix
from intramind_runtime.contracts import Artifact, CancelOutcome, EngineResult, digest


class MemoryArtifacts:
    def __init__(self):
        self.data = {}

    async def put(self, tenant_id, data, content_type="application/json"):
        key = tenant_prefix(tenant_id)+digest(data)
        self.data[key] = data
        return Artifact(key=key, sha256=digest(data), size=len(data), content_type=content_type)

    async def get(self, artifact):
        data = self.data[artifact.key]
        if len(data) != artifact.size or digest(data) != artifact.sha256:
            raise ValueError("checksum mismatch")
        return data

    async def put_file(self, tenant_id, source, content_type="application/json"):
        source.seek(0)
        return await self.put(tenant_id, source.read(), content_type)


class TestPermits:
    """Explicit permit stand-in for executor tests not exercising admission."""

    async def acquire(self, reservation):
        return {"attempt_id": reservation.attempt_id,
                "pool_id": reservation.pool_id,
                "engine_epoch": reservation.engine_epoch}

    async def heartbeat(self, reservation):
        return None

    async def release(self, reservation):
        return None


class IndependentEngine:
    """Engine task is shielded from the client's transport cancellation."""
    def __init__(self):
        self.calls = []
        self.finished = []
        self.gate = asyncio.Event()
        self.started = asyncio.Event()
        self.jobs = []

    async def execute(self, reservation, payload):
        self.calls.append(reservation.attempt_id)
        self.started.set()

        async def compute():
            await self.gate.wait()
            self.finished.append(reservation.attempt_id)
            return EngineResult(body={"choices": [{"message": {"content": "result"}}]},
                                input_tokens=10, output_tokens=10)

        job = asyncio.create_task(compute())
        self.jobs.append(job)
        return await asyncio.shield(job)

    async def cancel(self, reservation):
        return CancelOutcome.UNSUPPORTED
