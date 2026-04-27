# Multi-Node Docker for RCCL / MPI on AMD GPUs

A from-scratch walkthrough for running [RCCL-Tests](https://github.com/ROCm/rccl-tests)
across multiple AMD GPU nodes inside Docker containers, using OpenMPI over
Mellanox InfiniBand and a single shared SSH key for inter-container login.

Everything you need lives in this directory:

| File             | Purpose                                                               |
| ---------------- | --------------------------------------------------------------------- |
| `Dockerfile`     | Self-contained image (UCX, OpenMPI, RCCL, RCCL-Tests, sshd, RDMA libs) |
| `entrypoint.sh`  | Runtime setup (UID remap, SSH key install, sshd start)                |
| `setup_ssh.sh`   | Generate the shared keypair used by every container                   |
| `launch.sh`      | `docker run` with all the right GPU / IB / SSH flags                  |
| `stop.sh`        | Stop and remove the container on a node                               |

> This tutorial covers two scenarios:
>
> 1. **SLURM-managed cluster** — host SSH already works, nodes share `$HOME`. Easy mode.
> 2. **Unmanaged nodes** — no SLURM, no shared FS. You distribute keys manually, once.

---

## 1. Overview

Each node runs one container. The container exposes its own SSH server on
port `2224` (so it doesn't collide with the host's sshd on 22). All
containers share the same SSH keypair, which lets `mpirun` start RCCL ranks
across containers without password prompts.

```
   ┌─────────────────────── node-A ──────────────────────┐    ┌──── node-B ───┐
   │  host sshd:22                                       │    │  host sshd:22 │
   │      │                                              │    │      │        │
   │  ┌───┴────────────────────────────────────────┐     │    │  ┌───┴──────┐ │
   │  │  container rccl-mn                         │     │    │  │ rccl-mn  │ │
   │  │    sshd:2224                               │◄────┼────┼──┤ sshd:2224│ │
   │  │    /opt/ompi  /opt/ucx  /opt/rccl          │     │    │  └──────────┘ │
   │  │    /workspace/rccl-tests/build/            │     │    │               │
   │  │    GPU 0..7  (--device /dev/kfd /dev/dri)  │     │    └──── ... ──────┘
   │  │    Mellanox HCA (--device /dev/infiniband) │     │
   │  └────────────────────────────────────────────┘     │
   └─────────────────────────────────────────────────────┘
              ▲                                         ▲
              │ host SSH on :22 (rsync, orchestration)  │
              │ container SSH on :2224 (mpirun)         │
              └─────────────────────────────────────────┘
```

**Design choices:**

* **One image, no shared FS required.** UCX, OpenMPI, RCCL, and RCCL-Tests
  are all baked into the image so each node is fully self-contained once
  the image is pulled or built locally.
* **`--network host`**. The container reuses the host's network stack so
  RDMA traffic never touches a virtual bridge.
* **Container sshd on port 2224.** Avoids clashing with the host's sshd
  on 22 and lets `mpirun` use SSH for process launch (`--mca plm_rsh_agent
  "ssh -p 2224 ..."`).
* **Shared keypair, mounted read-only.** The container reads
  `/opt/ssh-keys/id_rsa` at startup and copies it into both `/root/.ssh`
  and `/home/ubuntu/.ssh`.
* **UID/GID remap at runtime.** The container's `ubuntu` user is renamed
  to your host UID so any bind-mounted directory keeps the right owner.

---

## 2. Prerequisites

* AMD GPUs (this tutorial targets MI300X / `gfx942`; change `--build-arg
  GPU_TARGETS=...` for other architectures).
* ROCm-capable kernel modules visible as `/dev/kfd` and `/dev/dri/render*`.
* Mellanox ConnectX HCA visible at `/dev/infiniband`.
* Docker Engine on every node, and your user in the `docker` group.
* Host-level SSH between nodes — see §4 if you don't have it yet.

Quick sanity check on each node:

```bash
ls /dev/kfd /dev/dri/render* /dev/infiniband 2>/dev/null
ibv_devinfo 2>/dev/null | head -20
docker info | grep -i 'server version'
```

---

## 3. Build the Docker image

From this directory, on **one** node:

```bash
docker build -t rccl-mn:standalone -f Dockerfile .
```

The build clones RCCL and RCCL-Tests from `develop` and compiles them for
`gfx942` by default. To pin different versions or target a different GPU:

```bash
docker build \
    --build-arg ROCM_IMAGE=rocm/dev-ubuntu-24.04:6.4-complete \
    --build-arg GPU_TARGETS=gfx942 \
    --build-arg ROCM_SYSTEMS_BRANCH=develop \
    --build-arg UCX_VERSION=1.16.0 \
    --build-arg OMPI_VERSION=4.1.6 \
    -t rccl-mn:standalone -f Dockerfile .
```

> RCCL and RCCL-Tests are sparse-checked-out of the
> [`ROCm/rocm-systems`](https://github.com/ROCm/rocm-systems) monorepo
> (only `projects/{rccl,rccl-tests,rocshmem}` are pulled, blob-filtered to
> keep the clone small). To pin to a release tag instead of `develop`, pass
> `--build-arg ROCM_SYSTEMS_BRANCH=rocm-6.4.0` (or whichever tag matches
> your `ROCM_IMAGE`).

The build takes ~25–40 min the first time (UCX + OpenMPI + RCCL all from
source). Subsequent rebuilds re-use Docker layer cache.

If your nodes don't share a registry, repeat the `docker build` on every
node, or `docker save | ssh node-N docker load` the resulting tarball.

---

## 4. SSH setup

The container's sshd needs a keypair so MPI processes in different
containers can ssh into each other. There are two ways to deliver that
keypair to every node, depending on whether your cluster is managed.

### Method 1 — SLURM-managed cluster (easy path)

A SLURM cluster typically gives you:

* host-level password-less SSH between compute nodes already configured;
* a shared `$HOME` (NFS, GPFS, Lustre, ...) visible from every node.

That's all we need. Generate the keypair **once** on any node:

```bash
./setup_ssh.sh
# -> ~/.docker-ssh-keys/{id_rsa, id_rsa.pub, authorized_keys, config}
```

Because `~/` is shared, every node sees the same `~/.docker-ssh-keys/`
directory. No copying needed.

Now grab a hostfile from the current allocation:

```bash
salloc -N 2 --gpus-per-node=8 --time=01:00:00      # if you don't have one yet
scontrol show hostnames "$SLURM_JOB_NODELIST" \
    | awk '{print $1 " slots=8"}' > ~/.mpi_hostfile
cat ~/.mpi_hostfile
# node-a slots=8
# node-b slots=8
```

> Adjust `slots=8` to the number of GPUs per node.

You're done with SSH setup. Skip ahead to **§5 Launch containers**.

### Method 2 — Unmanaged nodes (no SLURM, no shared FS)

If your nodes are bare hosts with no shared storage, you have to do two
one-time setup steps yourself:

**Step A — host-level password-less SSH between nodes.**
This is needed so `mpirun` can ssh from one container to another (it
ultimately uses the host's network stack). On each node, append every
*other* node's `id_rsa.pub` (the host's, not the container's) to
`~/.ssh/authorized_keys`:

```bash
# from your laptop / orchestrator host
for h in node-a node-b node-c; do
    ssh-copy-id "${USER}@${h}"          # interactive password prompt is OK
done

# verify (no password prompt expected)
for h in node-a node-b node-c; do
    ssh "${USER}@${h}" hostname
done
```

**Step B — generate the container keypair on one node, then push to all.**

```bash
./setup_ssh.sh                          # creates ~/.docker-ssh-keys/

for h in node-a node-b node-c; do
    rsync -a ~/.docker-ssh-keys/ "${h}:.docker-ssh-keys/"
done
```

> The same private key has to live on every node, because every container
> will copy it into its own `~/.ssh/id_rsa` and use it to authenticate to
> every other container.

Finally, write the hostfile by hand:

```bash
cat > ~/.mpi_hostfile <<EOF
node-a slots=8
node-b slots=8
node-c slots=8
EOF
```

---

## 5. Launch containers

On **every** node (or scripted via ssh from one node), run:

```bash
./launch.sh
```

This picks up the keypair from `~/.docker-ssh-keys/` and starts a container
named `rccl-mn` with all the GPU / IB / SSH flags wired in. It waits until
the container prints `=== Ready ===` and prints how to reach it.

Tweakables (env vars):

```bash
IMAGE=rccl-mn:standalone \
CONTAINER_NAME=rccl-mn \
SSH_PORT=2224 \
SHM_SIZE=64g \
GPUS=all \
EXTRA_VOLUMES="-v /shared:/shared -v $HOME/work:/work" \
    ./launch.sh
```

Launch on every node from a single host:

```bash
# SLURM:
srun --ntasks-per-node=1 ./launch.sh

# Manual:
for h in node-a node-b node-c; do
    ssh "${h}" "cd $(pwd) && ./launch.sh"
done
```

---

## 6. Verify SSH connectivity

From any node, hop into its container and check that you can reach every
other container on port 2224:

```bash
docker exec -it rccl-mn bash

# inside the container:
for h in $(awk '{print $1}' ~/.mpi_hostfile); do
    ssh -p 2224 "root@${h}" hostname
done
```

You should see every hostname print without a password prompt. If a node
hangs or asks for a password, see Troubleshooting.

> The hostfile isn't auto-mounted into the container; either
> `-v ~/.mpi_hostfile:/root/.mpi_hostfile:ro` in `launch.sh` (add it to
> `EXTRA_VOLUMES`), or just `scp` it in once you're inside.

---

## 7. Run RCCL-Tests

First, identify the host network interface that connects your nodes (NCCL
will pick the wrong one on its own if multiple interfaces exist — common
culprits are `usb0`, `docker0`, or anything with a `169.254.x.x`
link-local address that is the same on every node):

```bash
# on the host:
ip -4 -o addr | awk '{print $2, $4}'
# usb0    169.254.3.1/24      <- link-local, identical on every node, skip
# eth1    10.7.38.119/20      <- routable subnet, this is the one
# docker0 172.17.0.1/16       <- bridge, skip
```

Pick the interface whose subnet covers all your hosts (here, `eth1`) and
pass it via `NCCL_SOCKET_IFNAME` (NCCL bootstrap) plus
`oob_tcp_if_include` / `btl_tcp_if_include` (OpenMPI's own out-of-band
channel).

From inside the container on the head node:

```bash
docker exec -it rccl-mn bash

# inside:
NIC=eth1                # adjust to your inter-node interface
mpirun -np 16 \
       --hostfile ~/.mpi_hostfile \
       --map-by ppr:8:node \
       --mca plm_rsh_agent "ssh -p 2224 -o StrictHostKeyChecking=no -q" \
       --mca pml ucx --mca btl ^openib \
       --mca oob_tcp_if_include "$NIC" \
       --mca btl_tcp_if_include "$NIC" \
       -x NCCL_SOCKET_IFNAME="$NIC" \
       -x NCCL_IB_HCA=mlx5 \
       -x NCCL_DEBUG=VERSION \
       /workspace/rocm-systems/projects/rccl-tests/build/all_reduce_perf \
            -b 1G -e 16G -f 2 -g 1
```

Flag reference:

| Flag                                              | Why                                                             |
| ------------------------------------------------- | --------------------------------------------------------------- |
| `-np 16 --map-by ppr:8:node`                      | 8 ranks per node × 2 nodes = 16 ranks (one per GPU).            |
| `--hostfile ~/.mpi_hostfile`                      | List of nodes with `slots=N`.                                   |
| `--mca plm_rsh_agent "ssh -p 2224..."`            | Tells OpenMPI to launch remote `orted` over the container sshd. |
| `--mca pml ucx --mca btl ^openib`                 | Force the UCX PML; disable the legacy openib BTL.               |
| `--mca {oob,btl}_tcp_if_include $NIC`             | Pin OpenMPI's TCP traffic to the routable interface.            |
| `-x NCCL_SOCKET_IFNAME=$NIC`                      | Pin NCCL bootstrap traffic to the routable interface.           |
| `-x NCCL_IB_HCA=mlx5`                             | Restrict NCCL to Mellanox HCAs (`mlx5_0`, `mlx5_1`, ...).       |
| `-g 1`                                            | One GPU per rank in the rccl-tests harness.                     |

A healthy run prints a perf table climbing through buffer sizes 1G → 16G,
hitting ~150–250 GB/s busbw per group on MI300X. Reference run on a
2-node MI300X cluster (8 GPUs/node, ConnectX-7):

```text
        size         count      type   redop    root     time   busbw
   1073741824     268435456     float     sum      -1  11277.9  178.51
   2147483648     536870912     float     sum      -1  22718.5  177.24
   4294967296    1073741824     float     sum      -1  44699.4  180.16
# Avg bus bandwidth: 177.527 GB/s
```

---

## 8. Tear down

On every node:

```bash
./stop.sh
```

Or scripted:

```bash
for h in $(awk '{print $1}' ~/.mpi_hostfile); do
    ssh "${h}" "cd $(pwd) && ./stop.sh"
done
```

The image, keypair, and hostfile remain — you can re-launch with
`./launch.sh` instantly.

---

## 9. Extending the image (NIC drivers, plugins, custom setup)

Real clusters usually need a few extras on top of the base image:
Mellanox OFED userspace, the AWS / libfabric `aws-ofi-nccl` plugin, an
out-of-tree `librccl-net` plugin, NIC tuning (`mlxconfig`, IRQ pinning),
monitoring agents, application code, and so on.

There are two clean ways to add these — pick whichever matches the
artifact you're delivering:

| Pattern                              | Use when …                                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------------------------- |
| **A. Bake into the image**           | The step is fully self-contained (downloads from the internet, compiles, installs files).   |
| **B. Run at container launch time**  | The step needs host-level info (NIC names, MAC addresses, kernel modules, mounted paths).   |

### Pattern A — extend the image with a derived Dockerfile

Create `Dockerfile.custom` in this directory:

```dockerfile
# Dockerfile.custom -- our cluster-specific extensions
FROM rccl-mn:standalone

# Example 1: Mellanox OFED userspace (driver lives on the host kernel; we
#   only need the matching userspace libraries inside the container).
ARG MOFED_VERSION=24.10-1.1.4.0
ARG MOFED_OS=ubuntu24.04
RUN cd /tmp \
    && wget -q https://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VERSION}/MLNX_OFED_LINUX-${MOFED_VERSION}-${MOFED_OS}-x86_64.tgz \
    && tar -xzf MLNX_OFED_LINUX-*.tgz && cd MLNX_OFED_LINUX-* \
    && ./mlnxofedinstall --user-space-only --without-fw-update --force \
    && cd /tmp && rm -rf MLNX_OFED_LINUX-*

# Example 2: aws-ofi-nccl plugin (lets RCCL ride libfabric instead of
#   the built-in IB transport -- often a perf win on EFA / RoCE).
RUN cd /tmp && git clone --depth 1 https://github.com/aws/aws-ofi-nccl.git \
    && cd aws-ofi-nccl && ./autogen.sh \
    && ./configure --prefix=/opt/aws-ofi-nccl \
                   --with-libfabric=/usr --with-hip=${ROCM_PATH} \
                   --with-mpi=${MPI_INSTALL_PREFIX} \
    && make -j$(nproc) install \
    && cd /tmp && rm -rf aws-ofi-nccl

ENV LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:${LD_LIBRARY_PATH}

# Example 3: custom benchmarking / monitoring tools
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       perf-tools-unstable strace tcpdump \
    && rm -rf /var/lib/apt/lists/*
```

Build, ship to every node, relaunch:

```bash
docker build -t rccl-mn:custom -f Dockerfile.custom .

# distribute (no shared registry):
for h in node-b node-c; do
    docker save rccl-mn:custom | ssh "$h" docker load
done

# launch the new image everywhere:
for h in $(awk '{print $1}' ~/.mpi_hostfile); do
    ssh "$h" "cd $(pwd) && IMAGE=rccl-mn:custom ./launch.sh"
done
```

This pattern is **reproducible** (the Dockerfile is the source of truth)
and keeps `Dockerfile`, `launch.sh`, and `entrypoint.sh` untouched.

### Pattern B — host-aware setup at launch time

Some setup can't go in the image because it depends on the host: NIC IRQ
affinity, `mlxconfig` writes to a specific PCI address, copying a
licence file off the host, mounting a per-node scratch path, etc.
For these, ship a directory of scripts to every node and run them once
the container is up.

**1. Create a setup directory** with one or more scripts (do this on
your orchestrator host; copy or `rsync` the directory to every node, or
keep it in a shared `$HOME` if you have one):

```bash
mkdir -p ~/docker-mn-init
cat > ~/docker-mn-init/configure-nic.sh <<'EOF'
#!/bin/bash
# Tune every Mellanox HCA inside the container for low-latency RCCL.
set -e
for hca in /sys/class/infiniband/mlx5_*; do
    name=$(basename "$hca")
    echo 1 > "$hca/device/numa_node" 2>/dev/null || true
    echo "  tuned $name"
done
# Bump the OOB shared-memory limit (helps NCCL bootstrap on some kernels)
sysctl -w kernel.shmmax=$((64 << 30)) 2>/dev/null || true
echo "[ok] NIC tuning complete on $(hostname)"
EOF
chmod +x ~/docker-mn-init/configure-nic.sh

# (optional) replicate to every node if $HOME isn't shared
for h in node-b node-c; do
    rsync -a ~/docker-mn-init/ "$h:docker-mn-init/"
done
```

**2. Bind-mount the directory** when launching containers — `launch.sh`
already supports this through `EXTRA_VOLUMES`:

```bash
for h in $(awk '{print $1}' ~/.mpi_hostfile); do
    ssh "$h" "cd $(pwd) && \
        EXTRA_VOLUMES='-v \$HOME/docker-mn-init:/opt/init:ro' \
        ./launch.sh"
done
```

**3. Fan out execution** of the setup script to every container:

```bash
for h in $(awk '{print $1}' ~/.mpi_hostfile); do
    ssh "$h" "docker exec --privileged rccl-mn /opt/init/configure-nic.sh" &
done
wait
```

> The `--privileged` flag is needed only if your script writes to
> `/sys`, `/proc/sys`, or otherwise needs root capabilities the
> standard launch doesn't grant.  Most user-space tweaks don't need it.

### Pattern C — running any one-off across the whole cluster

The same fan-out idiom works for ad-hoc commands (smoke-tests, log
collection, environment dumps):

```bash
# parallel docker exec to every node, with output prefixed by hostname
for h in $(awk '{print $1}' ~/.mpi_hostfile); do
    ssh "$h" "docker exec rccl-mn ibv_devinfo" 2>&1 | sed "s|^|[$h] |" &
done
wait
```

For very large clusters you'll outgrow this loop — `pdsh -f 64`,
`parallel-ssh`, or Ansible all drop in as direct replacements.

---

## 10. Troubleshooting

| Symptom                                                          | Cause / fix                                                                                                                                |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `Permission denied (publickey)` between containers               | The same `id_rsa` is not on every node. Re-run `setup_ssh.sh`, then `rsync ~/.docker-ssh-keys/` to every node (Method 2 Step B).           |
| `mpirun` hangs at startup with no output                         | The container sshd isn't listening: `docker logs rccl-mn` should end with `=== Ready ===`. If not, check `docker logs` for entrypoint errors. |
| `mpirun` errors with `ssh: connect to host ... port 22: ...`     | You forgot `--mca plm_rsh_agent "ssh -p 2224 ..."` — by default OpenMPI uses port 22.                                                      |
| NCCL bootstrap fails: `Connect to 169.254.x.x failed`            | NCCL picked a link-local interface (e.g. `usb0`) that has the same IP on every node. Pass `-x NCCL_SOCKET_IFNAME=$NIC` and `--mca {oob,btl}_tcp_if_include $NIC`, see §7. |
| `cannot find verbs library` / `no IB devices`                    | `/dev/infiniband` isn't passed in. Verify `ls /dev/infiniband` on the host and that `launch.sh` added `--device=/dev/infiniband`.          |
| `RCCL: WARN Failed to open device ... Operation not permitted`   | GPU devices not accessible. Check `/dev/kfd` and `/dev/dri/render*` perms on the host (`ls -l`), and that you're in the `render` group.    |
| Container exits immediately                                      | `docker logs rccl-mn` will show the entrypoint error. Most common: `~/.docker-ssh-keys/` missing or unreadable.                            |
| Want to start over                                               | `./stop.sh && rm -rf ~/.docker-ssh-keys && docker rmi rccl-mn:standalone`                                                                  |
| Anything else                                                    | `docker logs rccl-mn` and `docker exec rccl-mn ip a / ibv_devinfo / mpirun --version` are your friends.                                    |
