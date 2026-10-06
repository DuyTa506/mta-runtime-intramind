#!/usr/bin/env bash
# 04-make-bundle.sh — offline bundle for mta-runtime-intramind (one image: intramind/runtime).
#
#   deploy/04-make-bundle.sh --out DIR [--image REF]
#
#   --out DIR     output directory (required; created if missing)
#   --image REF   use an existing image (ID or tag) instead of building; its
#                 org.opencontainers.image.revision label must equal HEAD
#
# Without --image the image is built by scripts/build-image.sh (git archive HEAD +
# revision label). The working tree must be clean.
#
# Output in DIR:
#   images/runtime.tar     docker save of the exact image ID
#   manifest.json          repo, git_sha (40), created, images[], templates[]
#   config/pools.example.json   pool catalog template (no secrets; the live pools.json
#                               is rendered per host profile by the infra installer)
#   SHA256SUMS.bundle      checksums of every file above except itself
# Temporal, runtime-postgres and the compose files come from the infra bundle.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO="$(dirname "$HERE")"
REPO_NAME="mta-runtime-intramind"

log(){ echo "[bundle $(date +%H:%M:%S)] $*"; }
die(){ echo "[bundle] ERROR: $*" >&2; exit 1; }
usage(){ sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

OUT=""; REF=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="${2:?--out needs a value}"; shift 2 ;;
    --image) REF="${2:?--image needs a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
done
[ -n "$OUT" ] || { usage >&2; die "missing --out"; }
command -v docker >/dev/null || die "docker is required"
command -v python3 >/dev/null || die "python3 is required (manifest.json)"
docker info >/dev/null 2>&1 || die "docker daemon not reachable / no permission"
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || die "working tree is dirty; commit first"
SHA="$(git -C "$REPO" rev-parse HEAD)"
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || die "git SHA is not 40 hex chars: $SHA"
mkdir -p "$OUT/images" "$OUT/config"; OUT="$(cd "$OUT" && pwd -P)"

log "[1/3] image"
if [ -z "$REF" ]; then
  REF="$(bash "$REPO/scripts/build-image.sh" | sed -n 's/^RUNTIME_IMAGE=//p')"
  [ -n "$REF" ] || die "build-image.sh printed no RUNTIME_IMAGE"
fi
ID="$(docker image inspect --format '{{.Id}}' "$REF" 2>/dev/null)" || die "image not found: $REF"
REV="$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$ID")"
[ "$REV" = "$SHA" ] || die "image revision label '$REV' != HEAD $SHA"
TAGS=(); while IFS= read -r t; do [ -z "$t" ] || TAGS+=("$t"); done \
  < <(docker image inspect --format '{{range .RepoTags}}{{println .}}{{end}}' "$ID")
if [ "${#TAGS[@]}" -eq 0 ]; then TAGS=("intramind/runtime:$SHA"); docker tag "$ID" "${TAGS[0]}"; fi

log "[2/3] docker save ${ID:0:19}..."
docker save -o "$OUT/images/runtime.tar" "${TAGS[@]}"
SUM="$(sha256sum "$OUT/images/runtime.tar" | cut -d' ' -f1)"; SIZE="$(stat -c %s "$OUT/images/runtime.tar")"
log "   images/runtime.tar $(du -h "$OUT/images/runtime.tar" | cut -f1)"

log "[3/3] templates + manifest.json + SHA256SUMS.bundle"
git -C "$REPO" show HEAD:config/pools.example.json > "$OUT/config/pools.example.json"
python3 - "$OUT" "$REPO_NAME" "$SHA" "$ID" "$(IFS=,; echo "${TAGS[*]}")" "$SUM" "$SIZE" "$REV" <<'PY'
import datetime, json, os, sys
out, repo, sha, iid, tags, sha256, size, rev = sys.argv[1:9]
manifest = {
    "schema": 1,
    "repo": repo,
    "git_sha": sha,
    "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "images": [{"name": "runtime", "file": "images/runtime.tar", "id": iid,
                "tags": [t for t in tags.split(",") if t], "sha256": sha256,
                "size_bytes": int(size), "revision_label": rev}],
    "templates": ["config/pools.example.json"],
    "guides": [],
}
with open(os.path.join(out, "manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=2); fh.write("\n")
PY
( cd "$OUT" && find . -type f ! -name SHA256SUMS.bundle -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.bundle )
log "DONE — $OUT ($(du -sh "$OUT" | cut -f1))"
