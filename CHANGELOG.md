# Changelog

## 0.1.0 — unreleased

- Durable task facade for activities, LLM operations, child workflows and waits.
- Stable identities and root accounting across children and Continue-As-New.
- PostgreSQL operation, attempt, reservation and budget ledger.
- Temporal asynchronous completion and durable outbox delivery.
- Immutable artifact adapter for existing MinIO/S3 storage.
- Admission, cancellation, timeout reconciliation and graceful executor drain.
- Authenticated service API, client and worker deployment administration.
- Registered pure model continuations with request-digest replay checks and bounded repair calls.
- Explicit structured-output and tool-call sizing profiles; tool schemas and transcript results
  are included in chat-template token accounting. Profiles remain deployment-qualified inputs.
- Independent development setup on `dev`, with pinned disposable PostgreSQL/Temporal/MinIO,
  Make targets, real integration CI and wheel installation checks.

The first release is being integrated into Intramind. Feature migrations and
deployment qualification are tracked by the application, not inferred from this
package version.
