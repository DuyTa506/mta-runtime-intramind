"""Trusted-service HTTP client; public identity verification remains at gateway."""

import json

import httpx


class RuntimeClient:
    def __init__(self, base_url: str, service_token: str, tenant_id: str, *, client=None):
        self.client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=15,
            headers={"Authorization": f"Bearer {service_token}", "X-Tenant-ID": tenant_id},
        )

    async def submit(self, submission: dict):
        response = await self.client.post("/v1/runs", json=submission)
        response.raise_for_status()
        return response.json()

    async def get_run(self, run_id: str):
        response = await self.client.get(f"/v1/runs/{run_id}")
        response.raise_for_status()
        return response.json()

    async def cancel(self, run_id: str):
        response = await self.client.post(f"/v1/runs/{run_id}/cancel")
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self.client.aclose()

    async def put_json(self, value: dict):
        return await self.put_bytes(
            json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode(),
            content_type="application/json",
        )

    async def put_bytes(self, value: bytes, *, content_type: str):
        response = await self.client.post(
            "/v1/artifacts",
            content=value,
            headers={"Content-Type": content_type},
        )
        response.raise_for_status()
        return response.json()

    async def read_json(self, ref: dict):
        response = await self.client.post("/v1/artifacts/read", json=ref)
        response.raise_for_status()
        return response.json()

    async def read_bytes(self, ref: dict) -> bytes:
        response = await self.client.post("/v1/artifacts/read", json=ref)
        response.raise_for_status()
        return response.content

    async def prepare(self, *, model_profile: str, payload: dict, max_output_tokens: int):
        response = await self.client.post(
            "/v1/requests/prepare",
            json={
                "model_profile": model_profile,
                "payload": payload,
                "max_output_tokens": max_output_tokens,
            },
        )
        response.raise_for_status()
        return response.json()
