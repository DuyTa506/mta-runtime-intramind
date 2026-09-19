"""Pure feedback policy; profiles and database CAS own applied decisions."""

from dataclasses import dataclass
from math import floor, isfinite


@dataclass(frozen=True)
class ControlProfile:
    hard_ceiling: int
    safe_target: int
    min_samples: int
    congestion_ratio: float
    decrease_factor: float
    cooldown_seconds: float
    telemetry_max_age_seconds: float
    max_inflight_age_seconds: float

    def __post_init__(self):
        if not (0 < self.safe_target <= self.hard_ceiling and self.min_samples > 0
                and self.congestion_ratio > 1 and 0 < self.decrease_factor < 1
                and self.cooldown_seconds > 0 and self.telemetry_max_age_seconds > 0
                and self.max_inflight_age_seconds > 0):
            raise ValueError("invalid calibrated controller profile")


@dataclass(frozen=True)
class ControlSample:
    observed_at: float
    samples: int
    service_latency_ratio: float | None
    pressure: bool | None
    oldest_inflight_seconds: float
    healthy: bool
    demand: bool
    profile_matches: bool


def decide(profile: ControlProfile, sample: ControlSample, *, target: int,
           now: float, last_change: float) -> tuple[int, str]:
    if not sample.healthy or not sample.profile_matches:
        return 0, "unhealthy_or_profile_mismatch"
    ratio = sample.service_latency_ratio
    if (sample.observed_at > now or now - sample.observed_at > profile.telemetry_max_age_seconds
        or sample.samples < profile.min_samples or ratio is None or not isfinite(ratio)):
        return min(target, profile.safe_target), "stale_or_insufficient"
    congested = ratio > profile.congestion_ratio and (
        sample.pressure is True or sample.oldest_inflight_seconds > profile.max_inflight_age_seconds)
    if congested:
        if now - last_change < profile.cooldown_seconds:
            return target, "cooldown"
        return max(1, floor(target * profile.decrease_factor)), "sustained_congestion"
    if now - last_change < profile.cooldown_seconds:
        return target, "cooldown"
    if sample.demand and sample.pressure is False and ratio <= 1:
        return min(target + 1, profile.hard_ceiling), "bounded_probe"
    return target, "hold"
