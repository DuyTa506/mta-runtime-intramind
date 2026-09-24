"""Timestamp epochs must match the engine instance; labels are not start times."""

from unittest.mock import AsyncMock

import pytest

from intramind_runtime.admin import confirm_epoch_stopped
from intramind_runtime.cli import reject_stale_engine_epochs
from intramind_runtime.epochs import epoch_lags_engine


def test_timestamp_epoch_lags_a_newer_engine_start():
    assert epoch_lags_engine("2026-09-17T22:35:10.86158907Z", "2026-09-21T22:35:31.756479357Z")
    assert not epoch_lags_engine("2026-09-22T05:44:39.695819472Z", "2026-09-22T05:44:39.695819472Z")


def test_profile_labels_are_not_compared_to_container_start():
    assert not epoch_lags_engine("serving-cutover-20260921", "2026-09-21T22:35:31Z")


def test_configure_rejects_only_a_stale_timestamp_epoch():
    config = {
        "pools": [
            {
                "admission": {"pool_id": "answer", "engine_epoch": "2026-09-17T22:35:10Z"},
                "base_url": "http://llamacpp-gemma:8000/v1",
            },
            {
                "admission": {"pool_id": "embed", "engine_epoch": "serving-cutover-20260921"},
                "base_url": "http://serving:5003/v1",
            },
        ]
    }
    starts = {"llamacpp-gemma": "2026-09-21T22:35:31Z", "serving": "2026-09-22T00:00:00Z"}

    with pytest.raises(ValueError, match="answer"):
        reject_stale_engine_epochs(config, lambda host: starts.get(host))


def test_configure_accepts_an_epoch_that_matches_the_running_engine():
    config = {
        "pools": [
            {
                "admission": {"pool_id": "tool", "engine_epoch": "2026-09-22T05:44:39.695819472Z"},
                "base_url": "http://llamacpp-qwen3-tool:8000/v1",
            }
        ]
    }
    reject_stale_engine_epochs(config, lambda host: "2026-09-22T05:44:39.695819472Z")


async def test_confirm_epoch_stopped_command_reports_the_settled_count():
    store = AsyncMock()
    store.confirm_epoch_stopped.return_value = 2

    report = await confirm_epoch_stopped(store, "tool", "2026-09-22T05:44:39Z", "process exited")

    store.confirm_epoch_stopped.assert_awaited_once_with(
        "tool", "2026-09-22T05:44:39Z", "process exited", recover=False
    )
    assert report == {
        "pool_id": "tool",
        "engine_epoch": "2026-09-22T05:44:39Z",
        "settled": 2,
        "recovery_requested": False,
    }
