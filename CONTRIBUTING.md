# Maintaining Intramind Runtime

This repository owns the reusable Python package and runtime processes. Feature
prompts, document access and rendering remain in the consuming applications.
Deployment topology, credentials and GPU allocations remain in infrastructure.

## Development

1. Branch from `dev`; use Python 3.12 and `make setup`.
2. Run `make check test` for lint and tests that require no services.
3. Run `make dev-up integration` for disposable PostgreSQL/Temporal/MinIO tests.
4. Run `make build`; install the resulting wheel in a fresh environment.
5. Open a pull request back to `dev`, describing contract changes and recovery checks.
   Run `make dev-down` when finished; it preserves local development volumes.

For contract changes, include failure/recovery tests, not only the happy path.
Changes to attempt settlement must cover unknown compute state and duplicate
delivery. Changes to workflow commands must replay an earlier workflow history.

## Release

- Keep `pyproject.toml` and `uv.lock` in the same change. Increment the package
  version for every distributed build; never replace a published wheel.
- Version public contracts and application workflow definitions independently.
  Keep a compatible worker available for workflows pinned to an older build.
- Build wheels and images from the same reviewed revision. Record SHA256 for
  the wheel, source and immutable image IDs in the consuming release manifest.
- The application pins a wheel/version. It must not install a moving Git branch
  on startup or pull a new package into an active worker environment.
- Production deployment and GPU qualification belong to each installation.
  A green library CI run is not an application cutover or a capacity benchmark.

Release artifacts can be distributed through GitHub Releases or an internal
Python index. Publishing to PyPI is a separate maintainer decision. CI builds
artifacts; it does not publish packages or deploy infrastructure automatically.
