#!/bin/bash
#
# standalone/launch.sh -- start the rccl-mn:standalone container on this node.
#
# Run this on every host you want a container on.  No shared FS required:
# the only file the container needs from the host is ~/.docker-ssh-keys/,
# which setup_ssh.sh produces (and which you copy to every node manually
# unless your $HOME is on NFS).
#
# Configurable via env vars (defaults shown):
#   IMAGE             rccl-mn:standalone
#   CONTAINER_NAME    rccl-mn
#   SSH_PORT          2224
#   SHM_SIZE          64g
#   GPUS              all (or e.g. "0,1,2,3")
#   KEY_DIR           ~/.docker-ssh-keys
#   EXTRA_VOLUMES     "" (space-separated -v flags, e.g. "-v /data:/data")
#
# Examples:
#   ./launch.sh
#   IMAGE=rccl-mn:dev ./launch.sh
#   EXTRA_VOLUMES="-v /shared:/shared -v $HOME/work:/work" ./launch.sh

set -e

IMAGE="${IMAGE:-rccl-mn:standalone}"
CONTAINER_NAME="${CONTAINER_NAME:-rccl-mn}"
SSH_PORT="${SSH_PORT:-2224}"
SHM_SIZE="${SHM_SIZE:-64g}"
GPUS="${GPUS:-all}"
KEY_DIR="${KEY_DIR:-${HOME}/.docker-ssh-keys}"
EXTRA_VOLUMES="${EXTRA_VOLUMES:-}"

if [ ! -d "${KEY_DIR}" ]; then
    echo "ERROR: ${KEY_DIR} not found.  Run ./setup_ssh.sh first." >&2
    exit 1
fi

# Stop any pre-existing container with the same name (idempotency)
if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    echo "[..] removing existing container ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

# Resolve host UID/GID and the render-group GID (so the container user can
# touch /dev/dri/render* without chmod 666 hacks).
HOST_UID=$(id -u)
HOST_GID=$(id -g)
RENDER_GID=$(getent group render 2>/dev/null | awk -F: '{print $3}')
RENDER_GID="${RENDER_GID:-109}"

# IB device passthrough is optional -- omit if /dev/infiniband doesn't exist.
IB_FLAG=""
if [ -d /dev/infiniband ]; then
    IB_FLAG="--device=/dev/infiniband"
fi

# GPUS=all maps to "all visible cards"; otherwise pin specific render nodes.
GPU_FLAGS="--device=/dev/kfd --device=/dev/dri"

echo "[..] launching ${CONTAINER_NAME} from ${IMAGE}"
docker run -d --rm \
    --name "${CONTAINER_NAME}" \
    --network host \
    --ipc host \
    ${GPU_FLAGS} ${IB_FLAG} \
    --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
    --security-opt seccomp=unconfined \
    --shm-size "${SHM_SIZE}" \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --group-add "${RENDER_GID}" \
    -e HOST_UID="${HOST_UID}" \
    -e HOST_GID="${HOST_GID}" \
    -e RENDER_GID="${RENDER_GID}" \
    -e SSH_PORT="${SSH_PORT}" \
    -e GPUS="${GPUS}" \
    -v "${KEY_DIR}:/opt/ssh-keys:ro" \
    ${EXTRA_VOLUMES} \
    "${IMAGE}" >/dev/null

# Wait for the entrypoint to print "=== Ready ==="
echo -n "[..] waiting for container to be ready"
for i in $(seq 1 60); do
    if docker logs "${CONTAINER_NAME}" 2>&1 | grep -q "=== Ready ==="; then
        echo
        echo "[ok] container ${CONTAINER_NAME} is ready on $(hostname)"
        echo "     ssh into it from another node:    ssh -p ${SSH_PORT} root@$(hostname)"
        echo "     open a shell here:                docker exec -it ${CONTAINER_NAME} bash"
        exit 0
    fi
    sleep 1
    echo -n "."
done

echo
echo "ERROR: container did not reach Ready state in 60s" >&2
echo "       check logs: docker logs ${CONTAINER_NAME}" >&2
exit 1
