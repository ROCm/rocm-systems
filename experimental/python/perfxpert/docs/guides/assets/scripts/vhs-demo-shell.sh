#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/start-demo-container.sh"

CONTAINER_NAME="${PERFXPERT_GUIDE_VHS_CONTAINER:-perfxpert-guide-vhs}"
exec docker exec -it "$CONTAINER_NAME" bash --noprofile --norc

