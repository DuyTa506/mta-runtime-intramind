"""Read-only retention inventory. Age is not proof an object can be deleted."""
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import json
import os

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Fixed identifiers only; never interpolate caller-provided table/column names.
COUNTS = {
    'roots': "SELECT count(*) AS total,count(*) FILTER (WHERE finished_at < :cutoff AND state <> 'RUNNING') AS aged_terminal FROM runtime_roots",
    'operations': "SELECT count(*) AS total FROM runtime_operations",
    'attempts': "SELECT count(*) AS total,count(*) FILTER (WHERE compute_held OR budget_held) AS held FROM runtime_attempts",
    'submission_identities': "SELECT count(*) AS retained FROM runtime_submissions",
    'completion_bindings': "SELECT count(*) AS retained FROM runtime_completion_bindings",
    'outbox': "SELECT count(*) AS total,count(*) FILTER (WHERE delivered_at < :cutoff) AS aged_delivered FROM runtime_outbox",
    'buffer_batches': "SELECT count(*) AS retained FROM runtime_buffer_batches",
    'buffer_items': "SELECT count(*) AS retained FROM runtime_buffer_items",
    'registered_artifacts': "SELECT count(*) AS rows,count(DISTINCT object_key) AS keys FROM runtime_artifacts",
}


async def inventory(engine, *, grace_days=30, now=None):
    if type(grace_days) is not int or grace_days < 1:
        raise ValueError('Grace period must be a positive whole number of days')
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=grace_days)
    tables = {}
    async with engine.connect() as connection:
        async with connection.begin():
            await connection.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY'))
            await connection.execute(text("SET LOCAL statement_timeout = '10000ms'"))
            for name, sql in COUNTS.items():
                result = await connection.execute(text(sql), {'cutoff': cutoff})
                tables[name] = dict(result.mappings().one())
    return {'mode': 'inventory_only', 'delete_authorized': False, 'cutoff': cutoff.isoformat(),
            'tables': tables, 'object_deletion_candidates': None,
            'reason': 'Complete external references, pinned histories and retry/idempotency tombstones are not established.'}


async def execute(args):
    engine = create_async_engine(os.environ['RUNTIME_DATABASE_URL'])
    try:
        print(json.dumps(await inventory(engine, grace_days=args.grace_days), sort_keys=True))
    finally:
        await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grace-days', type=int, default=30)
    # Deliberately no apply/delete switch or object-store write client.
    asyncio.run(execute(parser.parse_args()))


if __name__ == '__main__':
    main()
