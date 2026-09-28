import asyncio
from types import SimpleNamespace

import pytest
from test_artifact_reads import fixture_blob
from test_artifact_uploads import TOKEN, api

from intramind_runtime.artifacts import ArtifactCapacityBusy, SpoolBudget
from intramind_runtime.client import RuntimeClient, client_budget


def test_disk_headroom_and_negative_reservations(monkeypatch, tmp_path):
    import intramind_runtime.artifacts as module
    monkeypatch.setattr(module.os, "statvfs", lambda _: SimpleNamespace(f_bavail=1000, f_frsize=1))
    budget = SpoolBudget(spool_directory=tmp_path, min_free_bytes=900)
    with pytest.raises(ValueError, match="nonnegative"):
        with budget.reserve(-1):
            pass
    with budget.reserve(75):
        with pytest.raises(ArtifactCapacityBusy, match="headroom"):
            with budget.reserve(26):
                pass
        assert budget.total_bytes == 75
    assert budget.total_bytes == 0
    with budget.reserve(100):
        pass


@pytest.mark.asyncio
async def test_separate_activity_clients_share_admission_and_release(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNTIME_ARTIFACT_SPOOL_DIRECTORY", str(tmp_path))
    _, blobs, ref = await fixture_blob()
    async with api(blobs) as http:
        first = RuntimeClient("unused", TOKEN, "owner", client=http, read_concurrency=1)
        second = RuntimeClient("unused", TOKEN, "owner", client=http, read_concurrency=1)
        assert first.spool_directory == str(tmp_path)
        assert first.spool_budget is second.spool_budget
        async with first.open_verified(ref.model_dump()):
            with pytest.raises(ArtifactCapacityBusy):
                async with second.open_verified(ref.model_dump()):
                    pytest.fail("separate clients must not multiply read admission")
        async with second.open_verified(ref.model_dump()) as source:
            assert source.read() == b"verified payload"
        assert first.spool_budget.total_bytes == 0


def test_closed_event_loop_does_not_share_budget_with_next_loop(tmp_path):
    async def budget():
        return client_budget(tmp_path, 2, 100, 200, 0)
    with asyncio.Runner() as runner:
        first = runner.run(budget())
    with asyncio.Runner() as runner:
        second = runner.run(budget())
    assert first is not second


@pytest.mark.asyncio
async def test_http_low_disk_rejection_is_retryable_and_does_not_read_object(monkeypatch):
    import intramind_runtime.artifacts as module
    backend, blobs, ref = await fixture_blob()
    backend.get_object = lambda *_: pytest.fail("low disk must reject before object read")
    monkeypatch.setattr(module.os, "statvfs", lambda _: SimpleNamespace(f_bavail=1, f_frsize=1))
    async with api(blobs) as http:
        response = await http.post("/v1/artifacts/read", json=ref.model_dump())
        assert response.status_code == 503
        assert response.headers["retry-after"] == "1"
    assert blobs.spool_budget.total_bytes == 0
