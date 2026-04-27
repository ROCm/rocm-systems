#!/bin/bash
#
# standalone/stop.sh -- stop and remove the standalone container on this node.
#   CONTAINER_NAME defaults to "rccl-mn"; override via env.

set -e
CONTAINER_NAME="${CONTAINER_NAME:-rccl-mn}"

if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null
    echo "[ok] removed ${CONTAINER_NAME}"
else
    echo "[ok] no container named ${CONTAINER_NAME} on $(hostname)"
fi
