# Using RCCL/RCCL-Tests in a docker environment

## Docker build

Assuming you have docker installed on your system:

### To build the docker image :

By default, the given Dockerfile uses `docker.io/rocm/dev-ubuntu-22.04:latest` as the base docker image, and then installs RCCL (develop branch) and RCCL-Tests (develop branch), targetting `gfx942` GPUs.
```shell
$ docker build -t rccl-tests -f Dockerfile.ubuntu --pull .
```

The base docker image, rccl repo, rccl-tests repo, and GPU targets can be modified using `--build-args` in the `docker build` command above. E.g., to use a different base docker image for the MI250 GPU:
```shell
$ docker build -t rccl-tests -f Dockerfile.ubuntu --build-arg="ROCM_IMAGE_NAME=rocm/dev-ubuntu-20.04" --build-arg="ROCM_IMAGE_TAG=6.2" --build-arg="GPU_TARGETS=gfx90a" --pull .
```

### To start an interactive docker container on a system with AMD GPUs :

```shell
$ docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --network=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -it rccl-tests /bin/bash
```

### To run rccl-tests (all\_reduce\_perf) on 8 AMD GPUs (inside the docker container) :

If using ROCm 6.3.x or earlier
```shell
$ mpirun --allow-run-as-root -np 8 --mca pml ucx --mca btl ^openib -x NCCL_DEBUG=VERSION -x HSA_NO_SCRATCH_RECLAIM=1 /workspace/rccl-tests/build/all_reduce_perf -b 1 -e 16G -f 2 -g 1
```

If using ROCm 6.4.0 or later
```shell
$ mpirun --allow-run-as-root -np 8 --mca pml ucx --mca btl ^openib -x NCCL_DEBUG=VERSION /workspace/rccl-tests/build/all_reduce_perf -b 1 -e 16G -f 2 -g 1
```

For more information on rccl-tests options, refer to the [Usage](https://github.com/ROCm/rccl-tests#usage) section of rccl-tests.


---

## Multi-Node Setup

For running RCCL/MPI workloads across multiple nodes, use the `mnctl`
Python tool with `Dockerfile.Multinode.Ubuntu` (or
`Dockerfile.Multinode.ALinux3`). It layers SSH, user management, and
shared-directory infrastructure on top of any ROCm base image, and
orchestrates the build/launch across every host in your hostfile.

`mnctl` is implemented as a standard-library-only Python 3.6+ package
under `mnctl/`; the thin `run_mnctl.py` wrapper exists for users who
prefer not to type `python3 -m mnctl`.

### Files

| File | Purpose |
|---|---|
| `Dockerfile.Multinode.Ubuntu` | Lightweight multi-node image (SSH, user, PATH hooks, build tools) |
| `Dockerfile.Multinode.ALinux3` | Same as above for AlmaLinux-3 based ROCm images |
| `mnctl/` | Host setup + image build + shared deps (UCX/MPI) + container launch + orchestration |
| `run_mnctl.py` | Thin shim equivalent to `python3 -m mnctl` |
| `entrypoint.sh` | Container entrypoint (UID remap, SSH keys, sshd, post-setup hook) |
| `versions.env` | Pinned UCX / OpenMPI versions consumed by `mnctl` |
| `post-setup/` | Post-setup hook examples (AINIC, Mellanox, or custom) — see [post-setup/README.md](post-setup/README.md) |

Run `python3 -m mnctl --help` for the full option list.

### 2-Node Example (16 GPUs)

```bash
cd projects/rccl/docker

# Step 1: Create a hostfile (once)
cat > ~/.mnctl_hostfile << 'EOF'
node-a slots=8
node-b slots=8
EOF

# Step 2: Build image + launch containers on all nodes (single command)
#   - Builds the image if it doesn't exist (skips if already built)
#   - Launches containers on every node (skips nodes already running)
#   Use --ssh KEY_PATH for an existing key pair, or --ssh to auto-generate
python3 -m mnctl --launch-all --ssh ~/.ssh/id_rsa

# Step 3: Verify and run
python3 -m mnctl --verify

docker exec -it rccl-mn bash
MPI_HOME=/opt/shared/ompi
$MPI_HOME/bin/mpirun -np 16 \
  --hostfile ~/.mnctl_hostfile --map-by slot \
  --mca plm_rsh_agent "ssh -p 2224 -o StrictHostKeyChecking=no -q" \
  --allow-run-as-root \
  -mca pml ^ucx -mca osc ^ucx -mca btl ^openib \
  -x NCCL_SOCKET_IFNAME=eth1,eth0 \
  /opt/rccl-tests/build/all_reduce_perf -b 1 -e 16G -f 2 -g 1
```

### 4-Node Example (32 GPUs)

```bash
cd projects/rccl/docker

# Step 1: Create a hostfile with all 4 nodes
cat > ~/.mnctl_hostfile << 'EOF'
node-a slots=8
node-b slots=8
node-c slots=8
node-d slots=8
EOF

# Step 2: Build + launch everywhere (with auto-generated SSH keys)
python3 -m mnctl --launch-all --ssh

# Step 3: Verify and run
python3 -m mnctl --verify

docker exec -it rccl-mn bash
MPI_HOME=/opt/shared/ompi
$MPI_HOME/bin/mpirun -np 32 \
  --hostfile ~/.mnctl_hostfile --map-by slot \
  --mca plm_rsh_agent "ssh -p 2224 -o StrictHostKeyChecking=no -q" \
  --allow-run-as-root \
  -mca pml ^ucx -mca osc ^ucx -mca btl ^openib \
  -x NCCL_SOCKET_IFNAME=eth1,eth0 \
  /opt/rccl-tests/build/all_reduce_perf -b 1 -e 16G -f 2 -g 1
```

### Extra options

```bash
# Build shared deps only (UCX/OpenMPI) — once, shared via NFS
python3 -m mnctl --setup-deps

# Force rebuild image, shared deps, and replace all containers
python3 -m mnctl --launch-all --rebuild

# Use a specific ROCm image
python3 -m mnctl --launch-all rocm/dev-ubuntu-24.04:latest

# Mount RCCL source for development
python3 -m mnctl --launch-all \
    --volume "$HOME/rocm-systems/projects/rccl:/media/rccl"

# Run a post-setup script (NIC drivers, custom tools, etc.)
python3 -m mnctl --launch-all --post-setup ./post-setup/ainic

# AINIC with driver source mounted
python3 -m mnctl --launch-all \
    --post-setup ./post-setup/ainic \
    --volume /path/to/drivers-linux:/opt/nic-drivers:ro

# Use a different Dockerfile (e.g. AlmaLinux base)
python3 -m mnctl --launch-all --dockerfile Dockerfile.Multinode.ALinux3 \
    rocm/dev-almalinux-8:latest

# Validate configuration without launching
python3 -m mnctl --launch-all --dry-run

# Debug a failing setup
python3 -m mnctl --launch-all --verbose

# Teardown all containers across nodes
python3 -m mnctl --stop-all

# Launch / teardown a single node only
python3 -m mnctl --run --name rccl-mn
docker stop rccl-mn && docker rm rccl-mn
```

## Copyright

All modifications are copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
