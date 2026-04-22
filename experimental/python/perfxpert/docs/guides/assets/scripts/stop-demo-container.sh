#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${PERFXPERT_GUIDE_VHS_CONTAINER:-perfxpert-guide-vhs}"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

