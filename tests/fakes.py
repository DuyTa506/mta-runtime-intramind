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
