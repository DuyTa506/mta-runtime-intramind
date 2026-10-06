#!/usr/bin/env bash
# Build the runtime image from the committed source (git archive HEAD), never from the
# working tree, and label it with the full 40-char revision. Prints RUNTIME_IMAGE=<tag>.
#
#   scripts/build-image.sh            -> intramind/runtime:<sha40>
#   IMAGE_NAME=foo/rt scripts/build-image.sh
#
# Consumers (launcher/compose) pin the image by this revision; a short or missing
# label is rejected downstream, so the label is always the full SHA.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ] \
  || { echo "Commit source before building a release" >&2; exit 1; }
REVISION="$(git -C "$ROOT" rev-parse HEAD)"
case "$REVISION" in *[!0-9a-f]*|"") echo "bad revision: $REVISION" >&2; exit 1 ;; esac
[ "${#REVISION}" -eq 40 ] || { echo "revision is not 40 chars: $REVISION" >&2; exit 1; }
IMAGE="${IMAGE_NAME:-intramind/runtime}:${REVISION}"
CONTEXT="$(mktemp -d)"
trap 'rm -rf "$CONTEXT"' EXIT
git -C "$ROOT" archive "$REVISION" | tar -x -C "$CONTEXT"
docker build --label "org.opencontainers.image.revision=${REVISION}" -t "$IMAGE" "$CONTEXT" >&2
printf 'RUNTIME_IMAGE=%s\n' "$IMAGE"
