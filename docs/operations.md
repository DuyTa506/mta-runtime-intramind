# Operations

## Rollout gate

Rollout uses direct Temporal cutover after functional/recovery and deploy checks.
The release gate no longer requires overload observation, 72-hour soak, canary
percentages or seven days at full routing. Existing-work reconciliation, static
capacity qualification, security, MinIO and restore evidence still apply.

## Image build and offline bundle

```bash
scripts/build-image.sh                        # intramind/runtime:<sha40> from `git archive HEAD`, revision label = full SHA
deploy/04-make-bundle.sh --out /path/bundle   # images/runtime.tar + manifest.json + config/pools.example.json + SHA256SUMS.bundle
deploy/04-make-bundle.sh --out DIR --image <id|tag>   # bundle an existing image (its revision label must equal HEAD)
```

`manifest.json` (schema 1) lists `repo`, `git_sha`, `created`, `images[]` (`name`, `file`, `id`, `tags`,
`sha256`, `size_bytes`, `revision_label`) and `templates[]`. Both scripts require a clean working tree.
