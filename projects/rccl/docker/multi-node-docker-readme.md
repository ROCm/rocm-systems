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

All containers use `--network host` and communicate via passwordless SSH on
port 2224. SSH keys are **opt-in** — provide `--ssh-key`, `--ssh-keygen`, or
both `--ssh-key` and `--ssh-authorized-keys` (see [SSH Key Models](#ssh-key-models) below).

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

> **SLURM users:** skip this step. If `SLURM_NODELIST` is set and no hostfile
> exists, the script auto-generates one from the allocation. See
> [SLURM Integration](#slurm-integration) below.

### 2. Launch everywhere

```bash
cd projects/rccl/docker

# Option A: use your existing SSH key pair
./setup_multinode.sh --launch-all --ssh-key ~/.ssh/id_rsa

# Option B: auto-generate a shared key pair
./setup_multinode.sh --launch-all --ssh-keygen
```

This single command handles the entire setup on every node:
- Creates shared directories (SSH keys only when `--ssh-key` or `--ssh-keygen` is given)
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
  --hostfile PATH       MPI hostfile path            (default: ~/.mpi_hostfile; auto from SLURM)
  --shared-dir PATH     Shared workspace dir         (default: ~/.docker-shared)
  --builds-dir PATH     Shared builds dir            (default: ~/.docker-builds)
  --ssh-key-dir PATH    SSH key directory            (default: ~/.docker-ssh-keys)
  --ssh-key PATH        Use existing SSH key pair for inter-container SSH
  --ssh-authorized-keys PATH  Custom authorized_keys (for mesh SSH setups)
  --ssh-keygen          Auto-generate a shared SSH key pair
  --host-ssh-port PORT  Host SSH port for orchestration (default: 22)
  --post-setup PATH     Post-setup dir with setup.sh/env.sh (optional)
  --volume SRC:DST      Extra volume mount           (repeatable)
  --verbose             Detailed debug logging

Environment variables:
  ROCM_IMAGE, SHARED_DIR, BUILDS_DIR, SSH_KEY_DIR, SSH_KEY, SSH_AUTHORIZED_KEYS,
  HOSTFILE, SSH_PORT, GPUS, POST_SETUP_DIR, VERBOSE

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
| `--launch-all`/`--stop-all`/`--verify` without hostfile | Error | Requires a hostfile (or SLURM allocation) |
| `--run-only` when image doesn't exist | Error | Must build first or use `--run` |
| `--setup-deps` + `--launch-all` | Warning | Redundant; `--launch-all` builds deps automatically |
| `--rebuild` + `--verify`/`--stop-all` | Warning | `--rebuild` has no effect on read-only actions |
| `--gpus` without `--run`/`--launch-all` | Warning | No container is launched to use the GPU count |
| `--host-ssh-port` without `--launch-all`/`--stop-all` | Warning | Only used for host-to-host orchestration |
| `--ssh-key` + `--ssh-keygen` | Error | Mutually exclusive — choose one |
| `--ssh-authorized-keys` without `--ssh-key`/`--ssh-keygen` | Error | Needs a private key for outbound SSH |

---

## SSH Key Models

SSH is required for multi-node MPI but is **off by default**. Three models are supported:

### 1. Shared keypair (simplest)

All containers share one keypair. Best for fresh clusters without existing SSH keys.

```bash
# Auto-generate a shared keypair
./setup_multinode.sh --launch-all --ssh-keygen

# Or use an existing keypair (same key on all nodes via NFS)
./setup_multinode.sh --launch-all --ssh-key ~/.ssh/id_rsa
```

```
All nodes:  private_KEY  +  authorized_keys = [public_KEY]
```

### 2. Mesh SSH (per-node keys)

Each node keeps its **own** unique keypair, but every node's `authorized_keys`
contains the public keys of all other nodes. This is typical for clusters where
each host already has SSH configured.

```bash
./setup_multinode.sh --launch-all \
    --ssh-key ~/.ssh/id_rsa \
    --ssh-authorized-keys ~/.ssh/authorized_keys
```

```
Node A:  private_A  +  authorized_keys = [public_A, public_B, public_C]
Node B:  private_B  +  authorized_keys = [public_A, public_B, public_C]
Node C:  private_C  +  authorized_keys = [public_A, public_B, public_C]
```

The `--ssh-authorized-keys` file is copied into the container and the node's
own public key is appended automatically (so it does not need to be included
in the file, though duplicates are harmless).

> **Note:** `--ssh-key` and `--ssh-authorized-keys` paths must exist on every
> node. This works automatically when home directories are NFS-shared. If not,
> ensure each node has its own keypair at the same path and a matching
> `authorized_keys` file.

### 3. No SSH keys (single-node only)

Without `--ssh-key` or `--ssh-keygen`, containers generate local-only keys.
This is fine for single-node use but inter-container SSH across nodes will not work.

```bash
./setup_multinode.sh --run   # single-node, no SSH needed
```

### Key persistence

Once keys are configured in `~/.docker-ssh-keys/`, they persist across
container restarts and re-runs. The script reuses existing keys and skips
setup. To force re-setup, delete the key directory:

```bash
rm -rf ~/.docker-ssh-keys
```

---

## Common Workflows

```bash
# Launch on all nodes with your SSH key
./setup_multinode.sh --launch-all --ssh-key ~/.ssh/id_rsa

# Launch on all nodes with auto-generated SSH keys
./setup_multinode.sh --launch-all --ssh-keygen

# Mesh SSH: each node has its own key, authorized_keys contains all public keys
./setup_multinode.sh --launch-all \
    --ssh-key ~/.ssh/id_rsa \
    --ssh-authorized-keys ~/.ssh/authorized_keys

# Launch with a specific ROCm version
./setup_multinode.sh --launch-all --ssh-keygen rocm/dev-ubuntu-24.04:6.4

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

## Post-Setup Hooks

The setup supports a generic `--post-setup` hook for running any custom
configuration after containers are launched: NIC driver installation,
Python packages, custom library builds, benchmark tooling, etc.

### How it works

Pass a directory containing one or both of:

| File | Purpose |
|---|---|
| `env.sh` | Environment variables exported in every shell |
| `setup.sh` | Installation script (runs once at container startup, cached by content hash) |

The directory is mounted **read-only** at `/opt/post-setup` inside the container.
The entrypoint sources `env.sh` into `/etc/profile.d/` and runs `setup.sh` (if present)
with its SHA256 hash logged for auditability.

### Built-in examples

| Directory | Use Case | Notes |
|---|---|---|
| `post-setup/ainic/` | AMD AINIC NIC driver | Requires driver source at `/opt/nic-drivers` (use `--volume`) |
| `post-setup/mellanox/` | Mellanox ConnectX tuning | Host RDMA libs auto bind-mounted; env template only |

### Examples

```bash
# Mellanox (env vars only, no driver install needed)
./setup_multinode.sh --launch-all --post-setup ./post-setup/mellanox

# AINIC with driver source
./setup_multinode.sh --launch-all \
    --post-setup ./post-setup/ainic \
    --volume /path/to/drivers-linux:/opt/nic-drivers:ro

# Custom post-setup: install a Python package
mkdir -p ~/my-setup && cat > ~/my-setup/setup.sh << 'EOF'
#!/bin/bash
pip install torch-tb-profiler
EOF
./setup_multinode.sh --launch-all --post-setup ~/my-setup

# Custom post-setup: set environment variables
mkdir -p ~/my-setup && cat > ~/my-setup/env.sh << 'EOF'
#!/bin/bash
export NCCL_IB_GID_INDEX=1
export MY_CUSTOM_VAR=value
EOF
./setup_multinode.sh --launch-all --post-setup ~/my-setup
```

### Writing a custom config

See [post-setup/README.md](post-setup/README.md) for the full convention. Key rules:

- `setup.sh` **must** start with a `#!/bin/bash` shebang (validated before execution)
- `setup.sh` should be idempotent; results are cached in `/opt/builds/` by content hash
- `env.sh` should contain only `export` statements
- The config dir is read-only; `setup.sh` runs from a writable copy in `/tmp`

### Security

- **Read-only mount**: post-setup dir is mounted `:ro` — scripts cannot modify their source
- **Shebang validation**: `setup.sh` is rejected if it lacks a bash shebang
- **SHA256 logging**: Hash of each script is logged before execution for auditability
- **Explicit opt-in**: Nothing runs unless `--post-setup` is explicitly passed
- **Container-scoped**: Scripts run inside the (already privileged) container; no host-side execution

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
| **SLURM detection** | If hostfile is missing and `SLURM_NODELIST` is set, auto-generates one from the allocation |
| **Host setup** | Creates `~/.docker-shared`, `~/.docker-builds`, `~/.docker-ssh-keys`; configures SSH keys if `--ssh-key` or `--ssh-keygen` provided |
| **Image build** | `docker build` with `Dockerfile.Multinode.Ubuntu` (SSH, user, build tools — lightweight) |
| **Shared deps** | Builds UCX + OpenMPI once into `~/.docker-shared/` via a temporary container; shared across all nodes via NFS |
| **Container launch** | `docker run` with GPU/IB passthrough, host networking, NFS mounts, UID mapping |
| **Orchestration** | `--launch-all` reads hostfile, SSHes to each node, runs the above phases remotely |

The entrypoint (`entrypoint.sh`) handles runtime setup:
1. Remaps container user UID/GID to match host (NFS compatibility)
2. Distributes shared SSH keys to `~/.ssh/`
3. Starts sshd on port 2224
4. Runs post-setup hook (if `--post-setup` was provided)
5. Executes the given command or idles

---

## SLURM Integration

When running inside a SLURM allocation, the script auto-detects the allocation
and handles **both** hostfile generation and SSH key configuration — no extra
flags needed.

```bash
salloc -N 4 --gpus-per-node=8
./setup_multinode.sh --launch-all          # just this — everything is auto-detected
```

### What gets auto-detected

| Component | Source | Fallback |
|---|---|---|
| **Hostfile** | `scontrol show hostnames $SLURM_NODELIST` | Must create manually |
| **SSH private key** | `~/.ssh/id_rsa` or `~/.ssh/id_ed25519` | Auto-generates a new pair (`--ssh-keygen`) |
| **authorized_keys** | `~/.ssh/authorized_keys` | Derived from the public key |
| **Slots per node** | `--gpus`, auto-detect, `SLURM_GPUS_PER_NODE`, or `SLURM_NTASKS_PER_NODE` | `1` |

### How it works

SLURM detection triggers automatically when **all** of these are true:

1. The hostfile (`~/.mpi_hostfile` by default) does not exist
2. `SLURM_NODELIST` (or `SLURM_JOB_NODELIST`) is set in the environment
3. `scontrol` is available in `PATH`

When triggered, the script:

1. Expands `SLURM_NODELIST` into individual hostnames via `scontrol show hostnames`
2. Writes the MPI hostfile at the default path (one `hostname slots=N` per node)
3. Picks up `~/.ssh/id_rsa` (or `id_ed25519`) as the SSH key for inter-container SSH
4. Picks up `~/.ssh/authorized_keys` for the mesh SSH model (each node's key pair is different, but all are mutually authorized)

This means each container gets the same SSH config as the host — the user's
existing key pair and authorized keys are copied into every container.

If no SSH key exists at `~/.ssh/`, the script falls back to auto-generating a
shared keypair (equivalent to `--ssh-keygen`).

### Slot count priority

1. `--gpus N` (if explicitly provided)
2. Auto-detected GPU count on the current host
3. `SLURM_GPUS_PER_NODE` (handles formats like `4`, `gpu:4`, `gpu:4(S:0-1)`)
4. `SLURM_NTASKS_PER_NODE`
5. Falls back to `1` with a warning

### Examples

```bash
# Simplest: everything auto-detected
salloc -N 4 --gpus-per-node=8
./setup_multinode.sh --launch-all

# Override SSH keys (use a specific key pair instead of auto-detect)
salloc -N 4 --gpus-per-node=8
./setup_multinode.sh --launch-all --ssh-key ~/.ssh/cluster_key

# Override slot count
salloc -N 2 --gpus-per-node=8
./setup_multinode.sh --launch-all --gpus 4
```

### Generated hostfile

For a SLURM allocation with `SLURM_NODELIST=node[01-04]` and 8 GPUs per node,
the auto-generated `~/.mpi_hostfile` contains:

```
node01 slots=8
node02 slots=8
node03 slots=8
node04 slots=8
```

The file persists across re-runs. To regenerate it (e.g., for a new allocation),
delete it first:

```bash
rm ~/.mpi_hostfile
```

> **Note:** If `scontrol` is not available (e.g., on a non-SLURM login node),
> the script prints a warning and falls back to the standard hostfile-missing
> error. Install `slurm-client` or create the hostfile manually in this case.
>
> Explicit `--ssh-key` or `--ssh-keygen` flags always take precedence over
> SLURM SSH auto-detection.

---

## Troubleshooting

Run with `--verbose` for full debug output at every step.

| Problem | Fix |
|---|---|
| SSH to node fails | Check sshd: `docker exec rccl-mn /usr/sbin/sshd -p2224`. If keys are missing, re-launch with `--ssh-key` or `--ssh-keygen` |
| GPUs not visible | Verify `/dev/kfd` exists on host; check ROCm driver |
| NFS permission denied | Ensure host `id -u` matches `docker exec -u ubuntu rccl-mn id -u` |
| MPI hangs | Verify `NCCL_SOCKET_IFNAME` matches your NIC; check `ibstat` |
| Container won't start | Clear stale: `docker rm -f rccl-mn`; check `docker logs rccl-mn` |
| Post-setup fails | Check `docker logs rccl-mn` for `[FAIL] Post-setup`; re-run with `--verbose` |
| Post-setup env not applied | Verify: `docker exec rccl-mn bash -c 'cat /etc/profile.d/post-setup-env.sh'` |

---

## Quick Reference

```bash
./setup_multinode.sh --launch-all --ssh-keygen                            # deploy (auto SSH keys)
./setup_multinode.sh --launch-all --ssh-key ~/.ssh/id_rsa                 # use your keys
./setup_multinode.sh --launch-all --ssh-key ~/.ssh/id_rsa \
    --ssh-authorized-keys ~/.ssh/authorized_keys                          # mesh SSH
./setup_multinode.sh --setup-deps                                         # build shared UCX/MPI only
./setup_multinode.sh --launch-all --rebuild                               # force rebuild everything
./setup_multinode.sh --verify                                             # check SSH
./setup_multinode.sh --stop-all                                           # teardown everywhere
docker exec -it rccl-mn bash                                              # shell into container
docker logs rccl-mn                                                       # view startup logs
```
