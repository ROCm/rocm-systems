#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
ROCM_IMAGE="${PERFXPERT_GUIDE_VHS_IMAGE:-rocm/dev-ubuntu-22.04:7.2.2}"

exec docker run --rm -it \
  -v "$REPO_ROOT":/src:ro \
  "$ROCM_IMAGE" \
  bash -lc '
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends git jq less ca-certificates
    cp -a /src /tmp/src
    cd /tmp/src
    exec bash --noprofile --norc
  '
