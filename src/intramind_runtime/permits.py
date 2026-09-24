"""Service-token HTTP client for executor permits held by runtime-api."""

import asyncio
from datetime import UTC, datetime

import httpx

from .contracts import RuntimeConflict


class PermitClient:
    def __init__(self, base_url: str, service_token: str, *, client=None):
        self.client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {service_token}"},
            timeout=httpx.Timeout(None, connect=10), follow_redirects=False,
            transport=httpx.AsyncHTTPTransport(retries=0))

    async def close(self):
        await self.client.aclose()

    async def acquire(self, reservation):
        body = {"attempt_id": reservation.attempt_id,
                "pool_id": reservation.pool_id,
                "owner_id": reservation.owner_id,
                "engine_epoch": reservation.engine_epoch,
                "workload_class": reservation.workload_class,
                "deadline": reservation.attempt_deadline.isoformat()}
        while True:
            remaining = (reservation.attempt_deadline-datetime.now(UTC)).total_seconds()
            if remaining <= 0:
                raise TimeoutError("durable permit deadline exceeded")
            try:
                async with asyncio.timeout(remaining):
                    response = await self.client.post("internal/permits/acquire", json=body)
                if response.status_code == 200:
                    data = response.json()
                    if (data.get("attempt_id") != reservation.attempt_id
                        or data.get("pool_id") != reservation.pool_id
                        or data.get("engine_epoch") != reservation.engine_epoch):
                        raise RuntimeConflict("permit response identity changed")
                    return data
                if response.status_code == 409:
                    raise RuntimeConflict("durable permit fenced by runtime-api")
                if response.status_code == 504:
                    raise TimeoutError("durable permit deadline exceeded")
                if response.status_code in {401, 403, 422}:
                    raise RuntimeConflict("durable permit authentication or contract failed")
            except (httpx.TransportError, TimeoutError) as exc:
                if isinstance(exc, TimeoutError) and remaining <= 1:
                    raise
            await asyncio.sleep(min(1, max(0, (reservation.attempt_deadline-
                                            datetime.now(UTC)).total_seconds())))

    async def heartbeat(self, reservation):
        response = await self.client.post("internal/permits/heartbeat", json={
            "attempt_id": reservation.attempt_id, "owner_id": reservation.owner_id}, timeout=5)
        if response.status_code != 200:
            raise RuntimeConflict("durable permit heartbeat was fenced")

    async def release(self, reservation):
        response = await self.client.post("internal/permits/release", json={
            "attempt_id": reservation.attempt_id, "owner_id": reservation.owner_id}, timeout=5)
        if response.status_code != 200:
            raise RuntimeConflict("durable permit release requires termination evidence")
