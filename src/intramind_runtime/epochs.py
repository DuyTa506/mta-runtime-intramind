"""Compare a timestamp engine epoch with the instance that is actually running.

Profile labels such as ``serving-cutover-20260921`` are not instance start times
and are left unchanged. A newer container start means the configured epoch is
stale. The caller decides whether it can see that start time; this module never
rewrites an epoch, because changing it while attempts are open settles them.
"""

from datetime import datetime


def parse_timestamp(value: str) -> datetime | None:
    """Parse an RFC3339 engine timestamp, or None when the value is a label."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def epoch_lags_engine(engine_epoch: str, started_at: str) -> bool:
    """True only when both values are timestamps and the engine started later."""
    epoch = parse_timestamp(engine_epoch)
    started = parse_timestamp(started_at)
    if epoch is None or started is None:
        return False
    return started > epoch
