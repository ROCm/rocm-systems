# RCCL Smoke Test

A standalone program that verifies RCCL collective functionality across all
primary collectives and three message sizes.

Designed for system bring-up teams (and QA / CI) who need a quick,
single-command sanity check that RCCL is correctly installed and functional.
Supports both single-node and multi-node testing without MPI.

---

## What it tests

| Collective              | Datatype          | 1 KB | 4 MB | 1 GB |
|-------------------------|-------------------|------|------|------|
| AllReduce               | float32           | ✓    | ✓    | ✓    |
| AllGather               | float32           | ✓    | ✓    | ✓    |
| ReduceScatter           | float32           | ✓    | ✓    | ✓    |
| AllToAll                | float32           | ✓    | ✓    | ✓    |
| Broadcast               | float32           | ✓    | ✓    | ✓    |
| AllReduce               | bfloat16          | ✓    | ✓    | ✓    |
| ReduceScatter           | bfloat16          | ✓    | ✓    | ✓    |

Reduction collectives (`AllReduce`, `ReduceScatter`) are tested in both datatypes
because the `ncclSum` accumulation path differs by dtype. Data-movement collectives
have no reduction step, so one dtype is sufficient.

- **Reduction op:** `ncclSum` (where applicable)
- **Validation:** GPU-side correctness check after every collective
- **No performance metrics** — pass / fail only

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| ROCm ≥ 5.7  | Provides HIP runtime and `hip::device` CMake target |
| RCCL        | Defaults to `/opt/rocm/lib/librccl.so` |
| CMake ≥ 3.16 | Build system |
| `amdclang++` | Bundled with ROCm; required for HIP device code |

---

## Build

```bash
cd projects/rccl/test/smoketest
cmake -B build
cmake --build build -j$(nproc)
```

### Custom RCCL install path

```bash
cmake -B build -DRCCL_PATH=/path/to/custom/rccl/install
cmake --build build -j$(nproc)
```

The binary is placed at `build/RcclSmokeTest`.

---

## Running — single-node

Auto-detects all local GPUs. No arguments needed. One process manages all GPUs.

```bash
./build/RcclSmokeTest
```

Example output (8 GPUs, all passing):

```
Single-node smoke test: 8 GPU(s)

  RCCL Smoke Test   Ranks: 8
  Legend: P=Pass  F=Fail

  Collective                 |   1 KB |   4 MB |   1 GB |
  --------------------------+---------+--------+--------+
  AllReduce (float32)        |   P   |   P   |   P   |
  AllGather                  |   P   |   P   |   P   |
  ReduceScatter (float32)    |   P   |   P   |   P   |
  AllToAll                   |   P   |   P   |   P   |
  Broadcast                  |   P   |   P   |   P   |
  AllReduce (bfloat16)       |   P   |   P   |   P   |
  ReduceScatter (bfloat16)   |   P   |   P   |   P   |

  RESULT: ALL TESTS PASSED
  Test Time: 35.2 s
```

---

## Running — multi-node

Multi-node mode runs **one process per GPU** across all nodes. No MPI required.
The `NCCL_COMM_ID` environment variable provides the bootstrap address that RCCL
uses to exchange the communicator ID across processes.

### Arguments

| Argument | Description |
|----------|-------------|
| `--nranks N` | Total number of ranks (GPUs) across all nodes |
| `--rank R` | This process's global rank (0 … N-1) |
| `--device D` | Local GPU index to use on this node (default: 0) |

### Choosing a bootstrap port

Pick any unused port in 10000–65535 on the root node:

```bash
shuf -i 10000-65535 -n 1
```

### Direct invocation (2 nodes × 4 GPUs = 8 ranks)

On **node0** (ranks 0–3):
```bash
NCCL_COMM_ID=node0:54321 ./build/RcclSmokeTest --nranks 8 --rank 0 --device 0 &
NCCL_COMM_ID=node0:54321 ./build/RcclSmokeTest --nranks 8 --rank 1 --device 1 &
NCCL_COMM_ID=node0:54321 ./build/RcclSmokeTest --nranks 8 --rank 2 --device 2 &
NCCL_COMM_ID=node0:54321 ./build/RcclSmokeTest --nranks 8 --rank 3 --device 3 &
```

On **node1** (ranks 4–7), simultaneously:
```bash
NCCL_COMM_ID=node0:54321 ./build/RcclSmokeTest --nranks 8 --rank 4 --device 0 &
NCCL_COMM_ID=node0:54321 ./build/RcclSmokeTest --nranks 8 --rank 5 --device 1 &
NCCL_COMM_ID=node0:54321 ./build/RcclSmokeTest --nranks 8 --rank 6 --device 2 &
NCCL_COMM_ID=node0:54321 ./build/RcclSmokeTest --nranks 8 --rank 7 --device 3
```

Rank 0 prints the full pass/fail table. All ranks exit with code 0 (pass) or 1 (fail).

### Via launch script

`LaunchSmokeTest.sh` automates the above: it picks a free port, ssh-launches the
right process on each node, and waits for all to finish.

```bash
# Create a hostfile — one hostname per line, root node first
cat > hosts.txt <<EOF
node0
node1
EOF

# Launch: 2 nodes × 4 GPUs per node = 8 ranks
./LaunchSmokeTest.sh hosts.txt 4

# Optionally specify a port explicitly
./LaunchSmokeTest.sh hosts.txt 4 54321
```

SSH key-based authentication is required (no password prompts). The binary must
be at the same path on all nodes, or on a shared filesystem.

### Testing multi-node code path on a single node

You can exercise the `NCCL_COMM_ID` + `ncclCommInitRank` path on a single
8-GPU machine without a second node. RCCL will still use xGMI/PCIe for
intra-node communication — only the bootstrap mechanism differs from single-node mode.

```bash
echo "localhost" > hosts.txt
./LaunchSmokeTest.sh hosts.txt 8
```

---

## Verbose / debug output

```bash
# Full RCCL initialization and channel debug logs
NCCL_DEBUG=INFO ./build/RcclSmokeTest

# Suppress RCCL banner (just test output)
NCCL_DEBUG=WARN ./build/RcclSmokeTest
```

---

## Exit codes

| Code | Meaning |
|------|---------|
| `0`  | All tests passed |
| `1`  | One or more tests failed, or a fatal error occurred |

---

## Known limitations

- **No RDMA / InfiniBand testing:** Targets MI450 scale-up (xGMI / PCIe) workloads.
  IB transport support is out of scope for this version.

- **No Symmetric / CE collectives:** Planned for a future version.
