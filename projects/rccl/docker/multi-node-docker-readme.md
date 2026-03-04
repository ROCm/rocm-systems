# Multi-Node Docker Setup for RCCL/MPI

Run RCCL/MPI workloads across multiple AMD GPU nodes using Docker containers.

## Architecture

```
  NFS (shared across all nodes)
  ┌─────────────────────────────────────────────────────┐
  │  ~/.docker-shared        →     /opt/shared   (rw)   │
  │  ~/.docker-builds        →     /opt/builds   (rw)   │
  │  ~/.docker-ssh-keys      →     /opt/ssh-keys (ro)   │
  └──────────────┬────────────────────────┬─────────────┘
          ┌──────┴──────┐          ┌──────┴──────┐
          │   Node A    │          │   Node B    │
          │  Container  │◀───────▶│  Container  │
          │ --net host  │ ssh:2224 │  --net host │
          │  8 GPUs     │          │  8 GPUs     │
          └─────────────┘          └─────────────┘
```

All containers use `--network host` and share a single SSH keypair via NFS
for passwordless inter-node communication on port 2224.

## Prerequisites

- Linux host(s) with AMD GPUs and ROCm drivers installed
- Docker Engine 20.10+
- NFS (or equivalent) shared filesystem mounted on all nodes

## Quick Start

### 1. Create the MPI hostfile

```bash
cat > ~/.mpi_hostfile << 'EOF'
node-a slots=8
node-b slots=8
EOF
```

### 2. Launch everywhere

```bash
cd projects/rccl/docker
./setup_multinode.sh --launch-all
```

This single command handles the entire setup on every node:
- Creates shared directories and SSH keys (idempotent)
- Builds the Docker image (skips if already built)
- Builds shared deps — UCX and OpenMPI — into `~/.docker-shared/` (once, shared via NFS)
- Launches containers (skips nodes already running)

### 3. Verify

```bash
./setup_multinode.sh --verify
```

All hosts show `[OK]` — ready for MPI.

---

## Options Reference

```
./setup_multinode.sh [OPTIONS] [ROCM_IMAGE]

Lifecycle:
  --launch-all          Build + launch on ALL nodes in the hostfile
  --setup-deps          Build shared deps (UCX, OpenMPI) into shared dir (once)
  --stop-all            Stop + remove containers on ALL nodes
  --run                 Build + launch on the current node only
  --run-only            Launch without building (image must exist)
  --verify              Test SSH connectivity to all hostfile nodes
  --rebuild             Force rebuild image, shared deps, and replace containers

Configuration:
  --name NAME           Container name              (default: rccl-mn)
  --gpus N              Number of GPUs              (default: auto-detect)
  --ssh-port PORT       Container SSH port           (default: 2224)
  --shm-size SIZE       Shared memory                (default: 64g)
  --hostfile PATH       MPI hostfile path            (default: ~/.mpi_hostfile)
  --shared-dir PATH     Shared workspace dir         (default: ~/.docker-shared)
  --builds-dir PATH     Shared builds dir            (default: ~/.docker-builds)
  --ssh-key-dir PATH    SSH key directory            (default: ~/.docker-ssh-keys)
  --host-ssh-port PORT  Host SSH port for orchestration (default: 22)
  --volume SRC:DST      Extra volume mount           (repeatable)
  --verbose             Detailed debug logging

Environment variables:
  ROCM_IMAGE, SHARED_DIR, BUILDS_DIR, SSH_KEY_DIR, HOSTFILE, SSH_PORT, GPUS, VERBOSE

Path expansion:
  All path options support ~ and $VAR expansion:
    --hostfile '~/my_hostfile'
    HOSTFILE='$PROJECT/hosts' ./setup_multinode.sh --launch-all
```

### Option Compatibility

The script validates option combinations and will **error** or **warn** on misuse:

| Combination | Result | Reason |
|---|---|---|
| `--verify` + any other action | Error | `--verify` is a standalone check |
| `--stop-all` + `--run`/`--launch-all`/`--setup-deps` | Error | Cannot stop and start at the same time |
| `--launch-all` + `--run` or `--run-only` | Error | `--launch-all` already launches on all nodes |
| `--launch-all`/`--stop-all`/`--verify` without hostfile | Error | These actions require a hostfile |
| `--run-only` when image doesn't exist | Error | Must build first or use `--run` |
| `--setup-deps` + `--launch-all` | Warning | Redundant; `--launch-all` builds deps automatically |
| `--rebuild` + `--verify`/`--stop-all` | Warning | `--rebuild` has no effect on read-only actions |
| `--gpus` without `--run`/`--launch-all` | Warning | No container is launched to use the GPU count |
| `--host-ssh-port` without `--launch-all`/`--stop-all` | Warning | Only used for host-to-host orchestration |

---

## Common Workflows

```bash
# Launch on all nodes (default image: rocm/dev-ubuntu-24.04:7.1.1-complete)
./setup_multinode.sh --launch-all

# Launch with a specific ROCm version
./setup_multinode.sh --launch-all rocm/dev-ubuntu-24.04:6.4

# Build shared deps only (UCX/OpenMPI) — once, shared via NFS
./setup_multinode.sh --setup-deps

# Force rebuild image, shared deps, and replace containers
./setup_multinode.sh --launch-all --rebuild

# Mount RCCL source for development
./setup_multinode.sh --launch-all \
    --volume '$HOME/code/rocm-systems/projects/rccl:/media/rccl'

# Custom hostfile location
./setup_multinode.sh --launch-all --hostfile '~/cluster/my_hostfile'

# Debug a failing setup
./setup_multinode.sh --launch-all --verbose

# Teardown all containers
./setup_multinode.sh --stop-all

# Single-node only
./setup_multinode.sh --run
```

---

## Running Workloads

### Getting into a container

Launch containers with your source trees mounted, then `docker exec` in:

```bash
# Launch with project source directories mounted
./setup_multinode.sh --launch-all \
    --volume '$HOME/code/rccl:/media/rccl' \
    --volume '$HOME/code/rccl-tests:/media/rccl-tests'

# Get a shell inside the container
docker exec -it rccl-mn bash              # as root
docker exec -it -u ubuntu rccl-mn bash    # as ubuntu user
```

Any `--volume` paths are available inside every container.

> **Tip:** Mount paths are persistent — source edits on the host (or in
> another container) are visible immediately. Build artifacts stay in the
> mounted directory across container restarts.

### Build RCCL from source

Inside the container:

```bash
docker exec -it rccl-mn bash
cd /media/rccl

# Default release build (all supported GPU targets)
./install.sh

# Quick build for local GPU only
./install.sh -f

# Build with tests
./install.sh -t

# Build for a specific GPU target
./install.sh --amdgpu_targets gfx942

# See all options
./install.sh --help
```

### Build RCCL-Tests from source

Uses the custom RCCL built above and the shared OpenMPI installation:

```bash
docker exec -it rccl-mn bash
cd /media/rccl-tests

# Build with MPI support, pointing to the custom RCCL build
./install.sh -m \
  --rccl_home /media/rccl/build/release \
  --mpi_home /opt/shared/ompi

# Build for a specific GPU target
./install.sh -m \
  --rccl_home /media/rccl/build/release \
  --mpi_home /opt/shared/ompi \
  --gpu_targets gfx942

# Build without MPI (single-node only)
./install.sh --rccl_home /media/rccl/build/release

# See all options
./install.sh --help
```

The built binaries land in `./build/` (e.g., `./build/all_reduce_perf`).

### Run RCCL-Tests (multi-node)

```bash
docker exec -it rccl-mn bash

MPI_HOME=/opt/shared/ompi
$MPI_HOME/bin/mpirun -np 16 \
  --hostfile ~/.mpi_hostfile --map-by slot \
  --mca plm_rsh_agent "ssh -p 2224 -o StrictHostKeyChecking=no -q" \
  --allow-run-as-root \
  -mca pml ^ucx -mca osc ^ucx -mca btl ^openib \
  -mca oob_tcp_if_exclude docker,lo,usb0 \
  -mca btl_tcp_if_exclude docker,lo,usb0 \
  -x NCCL_SOCKET_IFNAME=eth1,eth0 \
  /media/rccl-tests/build/all_reduce_perf -b 1 -e 16G -f 2 -g 1
```

---

## How It Works

| Phase | What happens |
|---|---|
| **Host setup** | Creates `~/.docker-shared`, `~/.docker-builds`, `~/.docker-ssh-keys`; generates SSH keypair |
| **Image build** | `docker build` with `Dockerfile.Multinode.Ubuntu` (SSH, user, build tools — lightweight) |
| **Shared deps** | Builds UCX + OpenMPI once into `~/.docker-shared/` via a temporary container; shared across all nodes via NFS |
| **Container launch** | `docker run` with GPU/IB passthrough, host networking, NFS mounts, UID mapping |
| **Orchestration** | `--launch-all` reads hostfile, SSHes to each node, runs the above phases remotely |

The entrypoint (`entrypoint.sh`) handles runtime setup:
1. Remaps container user UID/GID to match host (NFS compatibility)
2. Distributes shared SSH keys to `~/.ssh/`
3. Starts sshd on port 2224
4. Executes the given command or idles

---

## Troubleshooting

Run with `--verbose` for full debug output at every step.

| Problem | Fix |
|---|---|
| SSH to node fails | Check sshd: `docker exec rccl-mn /usr/sbin/sshd -p2224` |
| GPUs not visible | Verify `/dev/kfd` exists on host; check ROCm driver |
| NFS permission denied | Ensure host `id -u` matches `docker exec -u ubuntu rccl-mn id -u` |
| MPI hangs | Verify `NCCL_SOCKET_IFNAME` matches your NIC; check `ibstat` |
| Container won't start | Clear stale: `docker rm -f rccl-mn`; check `docker logs rccl-mn` |

---

## Quick Reference

```bash
./setup_multinode.sh --launch-all           # deploy everywhere
./setup_multinode.sh --setup-deps           # build shared UCX/MPI only
./setup_multinode.sh --launch-all --rebuild # force rebuild everything
./setup_multinode.sh --verify               # check SSH
./setup_multinode.sh --stop-all             # teardown everywhere
docker exec -it rccl-mn bash                # shell into container
docker logs rccl-mn                         # view startup logs
```
