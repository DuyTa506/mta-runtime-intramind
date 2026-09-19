# Intramind Runtime

[Repository](https://github.com/DuyTa506/mta-runtime-intramind) · development branch: `dev`.
The package and Intramind migration are in active development; no production release has been declared.

Rollout uses direct Temporal cutover after functional/recovery and deploy checks.
The release gate no longer requires overload observation, 72-hour soak, canary
percentages or seven days at full routing. Existing-work reconciliation, static
capacity qualification, security, MinIO and restore evidence still apply.

A Python package for durable agent workflows and shared inference admission.
Application code declares the next step; the runtime accounts for inference
attempts; Temporal persists workflow decisions; the serving engine executes them.

## Boundaries

| Repository/component | Owns |
| --- | --- |
| This package | Task facade, operation ledger, admission, budgets, adapters and runtime services |
| Consuming application | Prompts, validators, workflow steps, source access and rendering |
| Infrastructure | Temporal/PostgreSQL deployment, MinIO, credentials, engine placement and capacity profiles |

The package is maintained independently and consumed as a versioned wheel.
It does not vendor AI/BE application code or start application services on import.
The existing MinIO artifact backend remains supported; installing this library
neither replaces storage nor migrates buckets.

## Components

| Module | Responsibility |
| --- | --- |
| `sdk` | Durable activities, LLM operations, child/map/wait and Continue-As-New |
| `client` / `api` | Authenticated submission, status, cancellation and artifact access |
| `temporal_adapter` | Idempotent submit/attach, asynchronous completion and outbox delivery |
| `store` | Atomic tenant/root admission, attempts, reservations, fencing and budgets |
| `buffering` | Bounded durable inputs, accepted policy, delayed batches and partition serialization |
| `executor` / `drivers` | One transport attempt and separate result persistence |
| `speech` | Qualified TTS preparation, termination contract and bounded WAV transport |
| `artifacts` | Tenant-scoped immutable objects and checksum verification |
| `uploads` | Bounded temporary files and concurrency for artifact ingestion |
| `admin` | Namespace and pinned worker deployment administration |
| `controller` | Pure feedback policy, requiring installation-specific telemetry/profiles |
| `release_gate` | Application-supplied migration and qualification evidence validation |

UNKNOWN reservations remain accounted for until compute termination is known.
No DB transaction spans inference. The completion driver does not promise
reliable remote cancellation, backend status lookup or exactly-once inference.

## Install and use

Use Python 3.12. Applications install the release wheel from the maintainer's
release artifacts or internal package index and pin its version and hash.

```bash
python -m pip install /release/intramind_runtime-0.2.0rc4-py3-none-any.whl
```

Declare features with `@durable_task` and call `TaskContext.activity`, `llm`,
`run_child`, `map_children`, `sleep` and `wait_event`. See
[the example workflow](src/intramind_runtime/example_workflow.py).
Feature modules do not import Temporal primitives directly. Keep HTTP/database
client initialization in activity modules, outside replayable workflow imports.
Use stable item keys and immutable artifacts; paginate large plans rather than
embedding source documents or unbounded child lists in workflow history.

Version `0.2.0rc4` adds deferred batch submission and migration `0003` to
the accepted-policy, speech, artifact and deadline contracts. It is a test release
candidate and must not replace an earlier wheel
under the same filename. Commit each passing phase, build immutable images from
that commit, then deploy and smoke-test those images before the next phase.

The AI translation integration suite exercises partial batch acceptance, finite
repair, publication retry and Continue-As-New through real Temporal/PostgreSQL.
It uses fake inference and the application's real Markdown parser/renderer;
these tests do not declare a production migration or measured serving capacity.

The opt-in `tests/test_ai_pptx_models.py` suite sends the application's native
condense and brief prompts through real Temporal/PostgreSQL and the broker ledger.
It verifies publication retry and history replay without new inference.
`tests/test_ai_pptx_context.py` also exercises native document/corpus reading,
bounded child windows, source and accepted-policy retention, mid-phase rollover,
designated-primary selection and root accounting across all replayed histories.
`tests/test_ai_pptx_planning.py` extends reading through deck planning, recorded
schema repair, split batches and publication retry. It replays every child and
planning rollover, verifies retained policy/output limits and settled root usage.
`tests/test_ai_pptx_manuscript.py` qualifies manuscript waves, bounded native
parse/audit repairs and a failed batch retaining the other slides. Lost join and
publication acknowledgements, rollover and replay keep root usage settled without
new inference. `tests/test_ai_pptx_validation.py` adds whole-deck coherence,
ordered repair windows, layout persistence, lost acknowledgements and replay.
An oversized audit is rejected by request preparation before admission; native
deterministic repairs continue, and only dispatched repairs spend root budget.
`tests/test_ai_pptx_render.py` runs a separate render queue, losing render and
publication acknowledgements before replay. It verifies Temporal cancellation
reaches the export subprocess through heartbeats and prevents publication.
It also runs the application's complete `pptx/v1` root through all five phases,
retains accepted queue/model/flags and tenant/root accounting, and replays every
child and rollover without repeating inference or rendering on publication retry.
Model/tokenizer responses, rendered file contents and feature artifacts are
fixtures; Chromium image verification and public PPTX cutover remain application work.

`RuntimeClient.llm_profile()` reads the authenticated
`GET /v1/llm/profiles/{model_profile}` descriptor: context limit, profile identity,
validated response formats and tool support. Persist it with agent planning state
and reject a changed profile before reusing its journal; it is configuration, not
live capacity or an admission permit. The HTTP driver preserves explicit boolean
`parallel_tool_calls` alongside the tool transcript, while still issuing exactly
one completion per admitted attempt and rejecting caller-controlled routing.

Submission may include a separate immutable `configuration` artifact selected by
the trusted application. Request identity remains the input digest and submission
key; concurrent retries retain the first committed configuration and one start
event. `TaskContext.configuration` carries that reference through children and
Continue-As-New. Keep enable flags and credentials outside this snapshot.
`map_children(window=...)` bounds each feature phase within `TaskPolicy.child_window`;
it does not reserve inference capacity. These additions require matching API/SDK
builds and are not present in the previously published wheel.

`RuntimeClient.buffer()` accepts tenant-scoped immutable input/configuration
artifacts, a partition key, a stable submission key, delay and batch size. Enable
`buffering` bounds in the task catalog explicitly. The outbox process seals due
inputs and starts a regular durable workflow; only one batch per tenant/task/partition
is active. Lost artifact/submit ACKs reuse the frozen batch and root. Policy changes
start a subsequent batch. Pending records do not expire or get trimmed; backpressure
rejects new inputs. `get_buffered(item_id)` reports pending, root state or an explicit
dispatch deadline failure. This facility does not require Celery Beat or Redis.

`prepare(..., attempt_timeout_seconds=...)` carries a trusted total attempt limit
outside the model payload. Values must be finite, positive numbers up to 86,400
seconds. Existing operations without this field retain a 1,800-second default;
the root deadline can shorten it. These are runtime bounds, not GPU capacity
recommendations. Reservation time comes from PostgreSQL, and send intent rechecks
expiry against its clock. The executor converts the remaining duration to an
asyncio timer for payload loading and transport, then stops that timer before
persisting a completed result. Transport timeouts remain separate limits.

Expiry before transport permits retry/refund; expiry after send enters `UNKNOWN`
and retains compute/budget accounting until termination is established. Neither
the attempt deadline nor the worker lease proves that a backend has stopped.
SDK calls without an explicit attempt limit retain the earlier activity command
shape; new feature histories and policy schemas still require matching pinned
worker builds. Reservation now includes its derived `attempt_deadline`; no new
database column is needed because attempt creation time, operation policy and
root deadline are already durable.

An optional absolute `deadline` on `ctx.llm`, `ctx.speech` and `ctx.activity`
bounds a feature phase across queueing, retries and worker recovery. Derive it
once from the workflow clock and reuse it for every effect in that phase.
`OperationDeadlineExceeded` lets the feature apply its explicit timeout policy;
it is never replayed as a model error that permits schema repair. A broker outage
cannot extend this wait. Operations already submitted retain their ledger and
deadline even when the waiting activity times out.

Expiry marks the operation terminal while independently running compute remains
accounted for. Late output is retained as `late_terminal`, usage settles once,
and neither lease recovery nor a late result can resurrect the operation. Only
confirmed not-sent work can refund its reservation. Calls without a phase deadline
keep their prior activity command shape. This is a source contract addition;
upgrade the API, broker and SDK together and retain pinned workers for old histories.

Speech uses `TaskContext.speech` / `speech_outcome` with the same operation/attempt
ledger, root attempt limit and resource-group admission. Its character budget is
separate from LLM tokens: the trusted task catalog must allocate
`resource_budgets: {"speech_characters": <positive limit>}`. Missing allocation is
rejected, and retries/fallback do not receive a new root budget. Migration `0002`
adds the resource ledger and labels existing attempts as `tokens`; it preserves
their active reservations. Install it before starting the new runtime processes.

A speech pool declares `admission.kind="speech"` and `character_limit` instead
of `context_limit`. The pool's `speech` configuration contains
`validated_profile_id`, `termination_contract="termination-v1"`, `sample_rate`,
`max_audio_bytes` and the allowed `voices`. The validated profile ID must match
the admission profile. `base_url` belongs to this deployment configuration;
`api_key_env` is optional for an isolated, unauthenticated serving deployment.
Pool/group ceilings and profile limits require measurement; no speech capacity
is enabled by the example configuration.

The consuming app checkpoints `GET /v1/speech/profiles/{model_profile}`, then uses
`prepare_speech` with that profile ID and its captured attempt timeout. Changed
profiles fail explicitly. The driver requires the serving response to echo its
attempt ID, `X-Intramind-TTS-Contract: termination-v1` and a compute state of
`not_started` or `terminated`. Unconfirmed responses/timeouts retain UNKNOWN
accounting. Headers are termination evidence, not idempotency or remote fencing.
There is no backend status/cancel API; unresolved compute still needs reconciliation.
WAV bytes and their result manifest are immutable, separately recorded artifacts;
persistence retry never synthesizes again. Binary data never enters workflow history.

Artifact uploads stream into a temporary file before MinIO publication. JSON and
`+json` media types retain a 16 MiB input limit; binary artifacts default to 128 MiB
to cover long WAV fallback. `RUNTIME_ARTIFACT_MAX_BYTES` may lower the shared API
and storage limit, and `RUNTIME_ARTIFACT_UPLOAD_CONCURRENCY` defaults to two uploads
per API process. Excess uploads receive 503 with Retry-After before reading the
body. Declared and received lengths are checked independently. MinIO multipart
upload uses one part worker, and checksum verification reads fixed-size chunks.
Temporary files and upload capacity are retained until storage I/O has stopped
on cancellation; only verified objects produce a reference. Orphan immutable
objects from lost responses still need the artifact retention/cleanup policy.

Provision temporary storage for the configured simultaneous uploads (256 MiB at
the defaults). If `/tmp` is a tmpfs, that space counts against container memory;
this upload bound does not qualify download/render memory or GPU capacity. Custom
`ArtifactPort` adapters must support seekable, caller-owned files via `put_file`.
The existing published wheel does not include this contract extension.

Service commands are `intramind-runtime api|worker|executor|outbox|reconciler|configure`.
They require explicit `RUNTIME_*` configuration. An application installs matching
schema migrations and provisions its namespace, storage bucket and pools before
starting workers. The API never initializes tables at startup.

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
RUNTIME_TEST_DATABASE_URL=postgresql+asyncpg://runtime_test:runtime_test_only@127.0.0.1:55439/runtime_test \
RUNTIME_TEST_ALLOW_RESET=yes \
RUNTIME_TEST_TEMPORAL_ADDRESS=127.0.0.1:17233 \
RUNTIME_TEST_MINIO_ENDPOINT=127.0.0.1:19009 \
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for contract, compatibility and release rules,
and [CHANGELOG.md](CHANGELOG.md) for the unreleased package contents.
