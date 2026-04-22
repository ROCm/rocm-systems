#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
CONTAINER_NAME="${PERFXPERT_GUIDE_VHS_CONTAINER:-perfxpert-guide-vhs}"
ROCM_IMAGE="${PERFXPERT_GUIDE_VHS_IMAGE:-rocm/dev-ubuntu-22.04:7.2.2}"

if docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  exit 0
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER_NAME" \
  -v "$REPO_ROOT":/src:ro \
  "$ROCM_IMAGE" \
  bash -lc '
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends git jq less ca-certificates
    cp -a /src /tmp/src
    python3 -m pip install -U pip setuptools wheel
    python3 -m pip install -e "/tmp/src/experimental/python/perfxpert[all]"
    /tmp/src/experimental/python/perfxpert/docs/guides/assets/scripts/prepare-vhs-demo.sh /tmp/src /tmp/perfxpert-vhs-demo
    exec sleep infinity
  ' >/dev/null

for _ in $(seq 1 300); do
  if docker exec "$CONTAINER_NAME" bash -lc 'test -f /tmp/perfxpert-vhs-demo/.ready'; then
    exit 0
  fi
  sleep 1
done

echo "Timed out waiting for $CONTAINER_NAME to finish setup." >&2
exit 1
