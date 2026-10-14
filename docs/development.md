# Development

## Development and verification

Python 3.12, `uv` and Make are sufficient for unit tests. Docker Compose v2 is
needed for the isolated integration services. No sibling application checkout,
GPU, production credentials or existing Intramind deployment is required.

```bash
git clone --branch dev https://github.com/DuyTa506/mta-runtime-intramind.git
cd mta-runtime-intramind
make setup check test
make dev-up
make integration
make build
make dev-down
```

`dev-up` creates only the `intramind-runtime-dev` Compose project. `dev-down`
retains its data volumes. The development credentials in `dev/` are fixtures
for loopback services, and must not be used for a shared deployment.

| Local service | Address |
| --- | --- |
| PostgreSQL | `127.0.0.1:55440` |
| Temporal | `127.0.0.1:17234` |
| Temporal UI | `http://127.0.0.1:18088` |
| MinIO API / console | `127.0.0.1:19010` / `http://127.0.0.1:19011` |

Tests use a dedicated `runtime_test` database, a test namespace, unique
workflow queues and temporary buckets. They never reset the development
`runtime_dev` database. Run `make integration` serially against a given test DB.
Application feature tests stay opt-in in the consuming application's environment.
Each pytest session creates a fresh `intramind-runtime-test-*` namespace so retained
worker deployments do not exhaust a previous session's limit. Set
`RUNTIME_TEST_TEMPORAL_NAMESPACE` to a name with that prefix when retaining an
explicit qualification run. Prior histories are not deleted.

The MinIO image is a compatibility test fixture pinned to the existing installation's
binary, fetched from the official Quay mirror. It is not a production version recommendation:
MinIO published a later [security release](https://github.com/minio/minio/releases/tag/RELEASE.2025-10-15T17-29-55Z)
with source-build instructions. Production storage upgrades and recovery validation belong
to the infrastructure release and do not run as part of this development setup.

The integration baseline is PostgreSQL 17.11, Temporal server 1.32.0 and Python
SDK 1.33.0. The lockfile pins Python dependencies. Full integration tests use
explicitly configured disposable services and reset the test database schema:

```bash
RUNTIME_TEST_DATABASE_URL=postgresql+asyncpg://runtime_test:runtime_test_only@127.0.0.1:55440/runtime_test \
RUNTIME_TEST_ALLOW_RESET=yes \
RUNTIME_TEST_TEMPORAL_ADDRESS=127.0.0.1:17234 \
RUNTIME_TEST_MINIO_ENDPOINT=127.0.0.1:19010 \
.venv/bin/python -m pytest -q --junitxml=.test-data/runtime-tests.xml
```

Missing external endpoints are explicit skips. Cross-application feature tests
also require their dependency environment and opt-in. CI/library tests do not
certify a GPU configuration, disaster recovery or application cutover.

`config/pools.example.json` intentionally registers no deployment capacity.
An installation supplies measured context, concurrency, model/template and
resource-group profiles. The pure controller is not automatically enabled.
`config/migration-status.json` is the Intramind migration checklist snapshot;
other consumers supply their own evidence to `intramind_runtime.release_gate`.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contract, compatibility and release rules,
and [CHANGELOG.md](../CHANGELOG.md) for the unreleased package contents.

### Output-language contracts

`intramind_runtime.language` provides immutable target-language policies and a
bounded text-patch validator. Policy v2 combines script checks with deterministic
offline language identification, including Latin-script mismatches (e.g. English
when Vietnamese was requested). Saved v1 policies retain their Han-only semantics.
Application adapters choose prose fields and protected source literals.
`LanguageGuardedPort` records one optional correction through the existing
`ModelPort`, retaining operation identity, replay and root accounting. Explicit
Chinese/Japanese/Korean targets permit Han. A rejected correction is terminal for
the leaf and cannot silently change numbers, citations or protected identifiers.
Short/ambiguous text and technical names are not classified as wrong merely for
being foreign words. Longer sentences require strong detector confidence; this is
not a semantic translation guarantee. `langdetect==1.0.9` ships the small profiles
inside its wheel, covered by the lockfile: no runtime download, GPU or extra LLM
call is needed for detection. See the [detector documentation](https://github.com/Mimino666/langdetect)
for supported languages and the deterministic seed behavior.

### Verified artifact downloads

`POST /v1/artifacts/read` keeps its existing request and byte-response contract.
MinIO downloads are verified into a temporary disk file before response headers or
payload bytes are sent. Admission covers verification, slow HTTP consumers, and
cancellation drainage. Exhausted capacity returns HTTP 503 with `Retry-After: 1`;
there is no unbounded waiting queue. Internal `get()` calls use the same read
admission. Reference-only checks use `verify_artifact()` without building bytes.

Configure `RUNTIME_ARTIFACT_SPOOL_DIRECTORY` to a writable **disk-backed** mount
with at least 1 GiB available; do not use tmpfs under the 512 MiB API memory limit.
Without this setting Python's temporary directory is used. Per-process defaults:
`RUNTIME_ARTIFACT_READ_CONCURRENCY=2`,
`RUNTIME_ARTIFACT_READ_SPOOL_BYTES=268435456`, and
`RUNTIME_ARTIFACT_SPOOL_BYTES=536870912` (combined upload and read reservations).
Chunks are 64 KiB. Unknown-length uploads reserve their route limit up front;
known-length uploads reserve their declared length and reject mismatches.
Keep runtime API admission single-authority: increasing processes also multiplies
these local budgets and does not extend a shared disk quota.

Binary SDK consumers should use `async with client.open_verified(ref) as source`
or `await client.read_to_file(ref, destination)` with a caller-owned binary file.
Both validate size and SHA-256 before exposing/copying bytes. `open_verified`
holds the file and client admission until context exit. `RuntimeClient` accepts
`spool_directory=` and defaults to two reads / 256 MiB read spool. Legacy
`read_bytes` / `read_json` remain compatible and materialize their results; their
callers still own the resulting memory. File-copy failures can leave partial
output in caller-owned destinations, so publish such files only after success.

`await client.put_file(source, content_type=...)` streams a seekable caller-owned
file in 64 KiB chunks and checks the returned digest/length before returning its
reference. The caller retains file ownership; cancellation drains active reads.

Spool admission also checks filesystem free space. The default
`RUNTIME_ARTIFACT_SPOOL_MIN_FREE_BYTES=1073741824` retains 1 GiB headroom after
subtracting the new and outstanding local reservations. This conservative check
is not a cross-process disk quota; budget other users of a shared volume separately.
The runtime image creates `/var/lib/intramind/artifact-spool` for UID 10001 so an
initial named-volume mount inherits writable ownership.

SDK file readers honor `RUNTIME_ARTIFACT_SPOOL_DIRECTORY` and the free-space setting.
Their concurrency/byte admission is shared by clients on the same event loop and
spool configuration, including clients constructed separately by each activity.
