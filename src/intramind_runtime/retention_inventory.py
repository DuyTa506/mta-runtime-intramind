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


def object_inventory(client, bucket, cutoff):
    """Stream metadata only; absence from ledger is never deletion evidence."""
    count = size = aged_count = aged_bytes = 0
    oldest = None
    for item in client.list_objects(bucket, prefix="tenants/", recursive=True):
        count += 1
        size += item.size or 0
        if item.last_modified is not None:
            oldest = min(oldest, item.last_modified) if oldest else item.last_modified
            if item.last_modified < cutoff:
                aged_count += 1
                aged_bytes += item.size or 0
    return {"objects": count, "bytes": size, "aged_objects": aged_count, "aged_bytes": aged_bytes,
            "oldest": oldest.isoformat() if oldest else None, "deletion_candidates": None}


async def execute(args):
    engine = create_async_engine(os.environ['RUNTIME_DATABASE_URL'])
    try:
        report = await inventory(engine, grace_days=args.grace_days)
        if args.include_objects:
            from minio import Minio
            client = Minio(os.environ['RUNTIME_MINIO_ENDPOINT'],
                access_key=os.environ['RUNTIME_MINIO_ACCESS_KEY'],
                secret_key=os.environ['RUNTIME_MINIO_SECRET_KEY'],
                secure=os.environ.get('RUNTIME_MINIO_SECURE', 'true').lower() in {'true', '1'})
            report['objects'] = await asyncio.to_thread(object_inventory, client,
                os.environ['RUNTIME_MINIO_BUCKET'], datetime.fromisoformat(report['cutoff']))
        print(json.dumps(report, sort_keys=True))
    finally:
        await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grace-days', type=int, default=30)
    parser.add_argument('--include-objects', action='store_true')
    # Deliberately no apply/delete switch or object-store write client.
    asyncio.run(execute(parser.parse_args()))


if __name__ == '__main__':
    main()
