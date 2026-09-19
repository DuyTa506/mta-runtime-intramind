from dataclasses import replace

from hypothesis import given
from hypothesis import strategies as st

from intramind_runtime.controller import ControlProfile, ControlSample, decide

PROFILE = ControlProfile(8, 1, 10, 1.3, 0.75, 30, 20, 120)
SAMPLE = ControlSample(100, 20, 1.0, False, 10, True, True, True)


@given(target=st.integers(0, 8), age=st.floats(min_value=21, max_value=1e8))
def test_stale_telemetry_never_probes_up(target, age):
    value, reason = decide(PROFILE, SAMPLE, target=target, now=100+age, last_change=0)
    assert value <= target and reason == "stale_or_insufficient"


@given(target=st.integers(0, 8), ratio=st.floats(0, 20, allow_nan=False))
def test_controller_never_exceeds_hard_ceiling(target, ratio):
    value, _ = decide(PROFILE, replace(SAMPLE, service_latency_ratio=ratio),
                      target=target, now=100, last_change=0)
    assert 0 <= value <= PROFILE.hard_ceiling


def test_profile_change_closes_admission():
    assert decide(PROFILE, replace(SAMPLE, profile_matches=False), target=8, now=100,
                  last_change=0) == (0, "unhealthy_or_profile_mismatch")


def test_congestion_needs_more_than_app_queue_delay():
    sample = replace(SAMPLE, service_latency_ratio=2, pressure=True)
    assert decide(PROFILE, sample, target=8, now=100, last_change=0)[0] == 6
    assert decide(PROFILE, sample, target=8, now=100, last_change=90) == (8, "cooldown")


def test_missing_latency_is_not_zero():
    assert decide(PROFILE, replace(SAMPLE, service_latency_ratio=None), target=8,
                  now=100, last_change=0) == (1, "stale_or_insufficient")
