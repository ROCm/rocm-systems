# NCCL M2N

NCCL M2N is a standalone NCCL-based library for cross-group GPU data
movement. It provides the **reshard** functionality: redistribute a global
tensor between two disjoint groups of GPU processes (the source group holds
one sharding / replication layout, the destination group holds another),
with a single API call from the application side.

The library uses copy/staging-backed reshard transports, with two entry
points: `ncclReshard`, and `ncclReshardWithWindow` for callers that supply a
user-registered NCCL window. Both use the transport selected by
`NCCL_RESHARD_COPY_ALGORITHM`, whose default is `PACK`. The shared library
is installed as `libnccl_m2n.so`; the public header is `nccl_m2n.h`.

## Maintainers

| GitHub | Areas |
|--------|-------|
| @kaushik-ks | Design, kernels, performance |
| @spotluri | Design, kernels, performance |
| @kingchc | APIs, integration |
| @kwen2501 | APIs, integration |

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
  - [Build](#build)
  - [Install](#install)
  - [Usage](#usage)
- [Public API Reference](#public-api-reference)
- [Build Reference](#build-reference)
- [Benchmarks](#benchmarks)
- [Runtime environment variables](#runtime-environment-variables)
- [Contributing](#contributing)
- [Third-party software](#third-party-software)
- [License](#license)

---

## Overview

A typical caller has two disjoint sets of ranks inside one NCCL
communicator: a source group and a destination group (e.g. trainer ranks
and inference / generator ranks). Both sides hold the same logical tensor —
each rank owns a local tile of it — but distributed under a different
layout. `ncclReshard` transfers the tensor from the source group to the
destination group with a single call, reshaping the tile on every
destination rank to match the destination layout.

A layout is described with two abstractions:

- **Mesh** (`ncclMesh_t`) — one side's rank topology only (no per-tensor
  placement), matching PyTorch DTensor's `DeviceMesh` / JAX's `Mesh`.
- **Distributed tensor** (`ncclDistTensor_t`) — the tensor descriptor for
  that side: shape, dtype, a pointer to its mesh, and **placement**
  (`NCCL_RESHARD_REPLICATE` / `NCCL_RESHARD_SHARD(d)`) per mesh axis,
  describing how each axis maps onto the tensor.

Every rank in the communicator — including ranks outside both the source
and destination meshes — must pass **both** the source and destination mesh
and tensor descriptors to the call. A rank that doesn't participate on a
given side still provides that side's descriptor with `dataPtr = NULL`,
since the library reads both meshes everywhere to derive the transfer
geometry.

See [Public API Reference](#public-api-reference) for the full mesh,
placement, and tensor-descriptor semantics, including rank roles and
same-dim vs. cross-dim sharding.

Both mesh axes may independently use `REPLICATE` or `SHARD(d)`, where
`0 <= d < tensor.ndims`. This includes `(SHARD(0), SHARD(1))`, swapped
placements, and repeated sharding such as `(SHARD(0), SHARD(0))`.
Mesh axes are applied in placement order: axis 0 partitions the global
tensor, then axis 1 partitions each resulting piece. For a length-16 tensor
on a `2 x 2` mesh with `(SHARD(0), SHARD(0))`, mesh coordinates `(0,0)`,
`(0,1)`, `(1,0)`, `(1,1)` own `[0,4)`, `[4,8)`, `[8,12)`, `[12,16)`.

Each rank holds a contiguous, equally sized local tile. For each tensor
dimension `d`, its global extent is `localShape[d]` times the product of
mesh-axis sizes placed at `SHARD(d)`. Source and destination global shapes
must match. Fully replicated layouts `(REPLICATE, REPLICATE)` are supported.
Public meshes have one or two axes, and tensors may have 1–3 dimensions.
These placement semantics apply to both `ncclReshard` and
`ncclReshardWithWindow`, including grouped calls and Python bindings.

**Reshard geometry** — each source tile sends its intersection with each
destination tile, including when either side shards multiple dimensions.
Replicas reuse the existing replication load balancing and forwarding paths.

---

## Quick Start

Prerequisites: CUDA and an MPI runtime for the benchmarks. NCCL is vendored
as a git submodule (`third_party/nccl`); `NCCL_HOME` defaults to its build
output automatically. To use a different NCCL build instead, set
`NCCL_HOME` explicitly (see below).

```bash
git clone <repo-url> nccl-m2n
cd nccl-m2n

# One-time: initialize and build the vendored NCCL submodule (skip if you
# set NCCL_HOME below to point at your own build)
make nccl-submodule

# Point at the NCCL build. Make defaults to the submodule's build output
# above, so this is optional for Make and required for CMake.
export NCCL_HOME=$PWD/third_party/nccl/build
```

### Build

Two build paths are shipped side-by-side — pick either. Make defaults to
`build/`; CMake writes to the directory passed with `-B`.

**Make**

```bash
make -C nccl_m2n                           # → build/lib/libnccl_m2n.so
make -C nccl_m2n reshard                   # → build/bin/reshard_bench (also a worked example)
```

`make help` lists all targets. `make` (no target) builds only the library.

**CMake**

```bash
cmake -S nccl_m2n -B build -DNCCL_HOME="$NCCL_HOME" \
      -DNCCL_M2N_BUILD_BENCH=ON
cmake --build build -j

# Library only (faster):
# cmake -S nccl_m2n -B build -DNCCL_HOME="$NCCL_HOME" && cmake --build build -j
```

See [Build Reference](#build-reference) for the full target list and
required/optional environment variables.

### Install

```bash
# Make — copies lib + nccl_m2n.h to $PREFIX (default /usr/local)
make -C nccl_m2n install

# CMake — copies lib + headers to CMAKE_INSTALL_PREFIX (default /usr/local)
cmake --build build --target install
```

### Usage

Initialize the runtime. The config struct lets you cap CTA count; other
tuning knobs are env-driven (see [Runtime environment variables](#runtime-environment-variables)):

```cpp
#include "nccl_m2n.h"

ncclM2nHandle_t m2nHandle = nullptr;
ncclM2nConfig_t cfg = NCCL_M2N_CONFIG_INITIALIZER;
cfg.maxCta = 8;
ncclM2nInit(&m2nHandle, &cfg);
```

Describe each side's mesh and tensor descriptor. Example: 8 ranks split 4
source / 4 destination, both native 1-D meshes sharding the outer tensor
dim. `dataPtr` is `NULL` on the side a rank doesn't participate in:

```cpp
int srcMeshDims[] = {4};
ncclMesh_t srcMesh = NCCL_M2N_MESH_INITIALIZER;
srcMesh.ndims = 1;
srcMesh.dims = srcMeshDims;
srcMesh.startRank = 0;

int dstMeshDims[] = {4};
ncclMesh_t dstMesh = NCCL_M2N_MESH_INITIALIZER;
dstMesh.ndims = 1;
dstMesh.dims = dstMeshDims;
dstMesh.startRank = 4;

int srcPlacements[] = {NCCL_RESHARD_SHARD(0)};
size_t srcLocalShape[] = {256, 1024};
ncclDistTensor_t src = NCCL_M2N_DIST_TENSOR_INITIALIZER;
src.dataPtr = is_source ? buffer : nullptr;
src.localShape = srcLocalShape;
src.ndims = 2;
src.dtype = ncclFloat32;
src.mesh = &srcMesh;
src.placements = srcPlacements;

int dstPlacements[] = {NCCL_RESHARD_SHARD(0)};
size_t dstLocalShape[] = {256, 1024};
ncclDistTensor_t dst = NCCL_M2N_DIST_TENSOR_INITIALIZER;
dst.dataPtr = is_dest ? buffer : nullptr;
dst.localShape = dstLocalShape;
dst.ndims = 2;
dst.dtype = ncclFloat32;
dst.mesh = &dstMesh;
dst.placements = dstPlacements;
```

Issue the reshard, then complete and finalize. Reshard is asynchronous, so
synchronize the stream before releasing its resources:

```cpp
ncclReshard(m2nHandle, comm, &src, &dst, stream);

cudaStreamSynchronize(stream);
ncclM2nFinalize(m2nHandle);
```

Callers that have an NCCL window registered can use `ncclReshardWithWindow`
instead, passing the window alongside the same `src`/`dst` descriptors.

---

## Public API Reference

### C API

```c
#include "nccl_m2n.h"
```

| Function | Purpose |
|---|---|
| `ncclM2nInit(handle, config)` | Initialize an explicit handle from an optional `ncclM2nConfig_t` (`NULL` config = all defaults). |
| `ncclM2nFinalize(handle)` | Release a handle — or `NULL` for the internal default handle — once its M2N work has completed. |
| `ncclM2nGetLastError()` | Return detail for the most recent M2N error on the calling thread. |
| `ncclReshard(handle, comm, src, dst, stream)` | Primary reshard entry point; copy/staging transport selected by `NCCL_RESHARD_COPY_ALGORITHM`. |
| `ncclReshardWithWindow(handle, comm, window, src, dst, stream)` | Reshard entry point for callers that supply a user-registered NCCL window. |
| `ncclM2nGroupStart()` | Begin recording `ncclReshard`/`ncclReshardWithWindow` calls on the calling thread instead of issuing them immediately. |
| `ncclM2nGroupEnd()` | Close the outermost group and issue its recorded calls. |
| `ncclM2nGroupAbort()` | Discard an active group's recorded calls without issuing them. |

Layouts are described with two descriptors (both in
[`src/nccl_m2n.h`](src/nccl_m2n.h), with static initializers
`NCCL_M2N_MESH_INITIALIZER` / `NCCL_M2N_DIST_TENSOR_INITIALIZER` /
`NCCL_M2N_CONFIG_INITIALIZER`):

| Descriptor | Fields |
|---|---|
| `ncclMesh_t` | `ndims`, `dims[]`, `startRank` — one side's rank topology only. |
| `ncclDistTensor_t` | `dataPtr`, `localShape[]`, `ndims`, `dtype`, `mesh`, `placements[]` — the per-rank tile plus its shape/dtype, a pointer to its mesh, and per-axis placement (`NCCL_RESHARD_REPLICATE` / `NCCL_RESHARD_SHARD(d)`). |

See [`src/nccl_m2n.h`](src/nccl_m2n.h) for the full contract on every
function and field, including preconditions, error codes, and
group-submission semantics.

### Python API

```python
import nccl.m2n as m2n
```

| Call | Purpose |
|---|---|
| `m2n.reshard(src, dst, comm, stream=..., src_mesh=..., src_placements=..., dst_mesh=..., dst_placements=...)` | Primary reshard entry point (staging-backed). |
| `m2n.reshard_with_window(...)` | Reshard entry point for callers that supply a user-registered NCCL window. |
| `m2n.Mesh(dims, start_rank=...)` | Rank-topology descriptor — flat list (1-D) or nested list (2-D). |
| `m2n.Shard(dim)` | Placement helper for a sharded mesh axis. |
| `m2n.init()` | Context manager yielding a handle exposing `handle.reshard(comm, src, dst, stream=...)`. |
| `m2n.group()` | Defines a grouped submission boundary for reshard calls. |
| `m2n.group_start()` / `m2n.group_end()` / `m2n.group_abort()` | Explicit group control for non-context-manager code. |

See the [M2N Python guide](../python/nccl/m2n/README.md) for install steps
and full usage examples.

---

## Build Reference

Both Make and CMake are supported; pick the one that fits your toolchain.

### Make targets

| Target | Output | Notes |
|---|---|---|
| `make` / `make lib`             | `build/lib/libnccl_m2n.so`                  | Library only; no MPI link. |
| `make reshard`                  | `build/bin/reshard_bench`                | Single-layer bench (links MPI). |
| `make reshard_batch_user_window` | `build/bin/reshard_batch_bench_user_window` | Batched/concurrent comm sweep. |
| `make reshard_model`            | `build/bin/reshard_model_bench`           | Config-driven model transfer bench (links MPI). |
| `make bench`                    | All bench binaries above                    | |
| `make bench reshard`            | Equivalent to `make reshard`                | Sub-name picker, see `make help`. |
| `make install`                  | Copies `lib` + `nccl_m2n.h` to `$PREFIX`  | Defaults `PREFIX=/usr/local`. |
| `make clean`                    | Removes M2N artifacts from `build/`         | Preserves other libraries' artifacts. |

### CMake targets

```bash
cmake -S nccl_m2n -B build -DNCCL_HOME="$NCCL_HOME" \
      [-DNCCL_M2N_BUILD_BENCH=ON]
cmake --build build -j [--target <name>]
```

| `--target` | Output | Notes |
|---|---|---|
| *(default)*           | `build/lib/libnccl_m2n.{so,a}`            | Builds all configured targets. |
| `nccl_m2n_shared`    | `build/lib/libnccl_m2n.so`                | Library only. |
| `nccl_m2n_static`    | `build/lib/libnccl_m2n.a`                 | Static archive. |
| `reshard_bench` *etc.* | `build/bin/<name>`                        | Requires `-DNCCL_M2N_BUILD_BENCH=ON`. |
| `install`             | Copies `lib` + headers to `CMAKE_INSTALL_PREFIX` | Defaults `/usr/local`. |

### Required environment

| Variable | Default | Purpose |
|---|---|---|
| `NCCL_HOME` | Make: `third_party/nccl/build` (vendored submodule); CMake: required at configure time (no default) | Path to a from-source NCCL build (`$NCCL_HOME/include/nccl_device.h` must exist). Make reads the env var directly; CMake configure fails when `NCCL_HOME` is unset. Pass `-DNCCL_HOME=...` or set the environment variable. Override to point at your own build. |

### Optional environment / cache vars

| Make var | CMake equivalent | Default | Purpose |
|---|---|---|---|
| `CUDA_HOME` | auto-detected by `find_package(CUDAToolkit)` | `/usr/local/cuda` | CUDA install. CUDA ≥ 12.8 is needed for the default `sm_100` arch. |
| `MPI_HOME` | auto-detected by `find_package(MPI)` | system MPI | Used by benchmarks only (the library does not link MPI). |
| `PREFIX` | `CMAKE_INSTALL_PREFIX` | `/usr/local` | `install` destination. |
| `NVCC_GENCODE` | `CMAKE_CUDA_ARCHITECTURES` | `sm_80, sm_90, sm_100` (Make) / `80;90;100` (CMake) | Target GPU arch. |
| `DEBUG=1` / `DEBUG=full` | `-DCMAKE_BUILD_TYPE=Debug` (+ `-DCMAKE_CUDA_FLAGS_DEBUG=...`) | unset / Release | Line info / device debug. |
| `BUILDDIR` | `cmake -B <dir>` | `build/` | Make output directory. |

---

## Benchmarks

All benches link MPI for NCCL bootstrap; rank 0 broadcasts the unique ID.

### Single layer — `reshard_bench`

Drives one reshard with the configuration given on the command line; runs
warmup + timed iterations, optionally validates the byte pattern.

```bash
# 8 GPUs, 2-D, same-dim sharding
mpirun -np 8 ./build/bin/reshard_bench \
    --src-mesh-dims 1,4 --dst-mesh-dims 1,4 \
    --tensor-dims 1024,1024 \
    --src-shard-dim 0 --dst-shard-dim 0 \
    --algorithm ring --validate

# 3-D cross-dim
mpirun -np 8 ./build/bin/reshard_bench \
    --src-mesh-dims 1,4 --dst-mesh-dims 1,4 \
    --tensor-dims 256,128,64 \
    --src-shard-dim 0 --dst-shard-dim 2 \
    --algorithm ring --validate
```

`--help` lists all flags (`--lb-mode`, `--print-all-ranks`,
`--verbose`, ...).

### Model transfer — `reshard_model_bench`

A config-driven benchmark that measures disaggregated model resharding using
real model parameter shapes, dtypes, and parallelism configs. Unlike the
synthetic `reshard_bench` (single-layer), this benchmark reads HuggingFace-style model config
and system config JSON files, automatically computing placement rules, expert
grouping, PP stage mapping, and deduplication.

#### How It Works

- Parses the model config (per-parameter shapes/dtypes) and system config
  (train/gen parallelism: TP, CP, EP, DP, PP) into a transfer descriptor
  per parameter.
- Groups per-expert weights into 3D tensors and deduplicates repeated
  PP-stage patterns by default (disable with `--no-dedup`).
- Infers each parameter's placement (column-/row-/expert-parallel, or
  replicated) from its name.
- Creates one NCCL communicator and CUDA stream per (train_stage,
  gen_stage) pair, and allocates a device buffer per transfer descriptor.
- Runs a warmup pass followed by timed iterations (with optional
  `--validate` correctness checks), reporting per-pattern and aggregate
  bandwidth/latency.

#### Input File Formats

**Model config JSON** (generated by `hf_converter.py` from a HuggingFace repo):

```json
{
  "lm_head.weight": { "shape": [129280, 7168], "dtype": "BF16" },
  "model.layers.0.self_attn.q_proj.weight": { "shape": [7168, 7168], "dtype": "F8_E4M3" },
  "model.layers.0.input_layernorm.weight": { "shape": [7168], "dtype": "BF16" }
}
```

**System config JSON**:

```json
{
  "train": {
    "num_gpus": 32,
    "tp_size": 1,
    "cp_size": 1,
    "ep_size": 16,
    "dp_size": 1,
    "pp_size": 2
  },
  "generation": {
    "num_gpus": 32,
    "tp_size": 8,
    "cp_size": 1,
    "ep_size": 1,
    "dp_size": 4,
    "pp_size": 1
  }
}
```

Total MPI world size must equal `train.num_gpus + generation.num_gpus`.
Ranks `[0, train.num_gpus)` are trainers; ranks
`[train.num_gpus, total)` are generators.

#### Building

```bash
make bench reshard_model
make bench reshard_model NVCC_GENCODE="-gencode=arch=compute_100,code=sm_100"
```

#### Running

```bash
mpirun -np <total_gpus> ./build/bin/reshard_model_bench \
    --model-config benchmarks/configs/model_configs/<model>.json \
    --system-config benchmarks/configs/system_configs/<system>.json \
    --iterations 10 --warmup 2 \
    --algorithm auto --lb-mode node \
    --validate
```

`--help` lists all flags and defaults.

#### Shipped configs

This tree ships a toy DeepSeek-style model config at
`benchmarks/configs/model_configs/dsv3-toy.model.json` plus system config
examples under `benchmarks/configs/system_configs/`:

- `dsv3-128gpus-gb200.json`
- `dsv3-256gpus-gb200.json`
- `dsv3-pp-64gpus-gb200.json`
- `qwen235b-TP2CP2PP4EP16-128gpus-gb200.json`

---

## Runtime environment variables

### Generic

Apply regardless of the selected copy algorithm.

| Variable | Effect |
|---|---|
| `NCCL_RESHARD_LOG_LEVEL`      | One of `NONE`, `WARN` (default), `INFO`, `DEBUG`, `TRACE`. |
| `NCCL_RESHARD_COPY_ALGORITHM` | Copy transport for both entry points: `PACK` (default), `DIRECT`, or `PIPE` (beta). |
| `NCCL_RESHARD_NUM_CTAS`       | Directly overrides the resolved CTA count; invalid values are ignored. |
| `NCCL_RESHARD_DEVCOMM_CACHE_CAPACITY` | Maximum cached DevComms per process (default 64). A positive integer, captured at runtime initialization; host storage is allocated on first use and DevComms are created lazily. No runtime eviction or automatic growth. Exhaustion fails locally before new DevComm creation; abort the distributed job on error, increase the limit, and restart. No cross-rank admission collective is performed, so peers may otherwise remain waiting in setup. Does not change split-communicator or persistent-control limits. |

### Algorithm-specific

| Variable | Algorithm | Effect |
|---|---|---|
| `NCCL_RESHARD_PACK_BUFFSIZES` | PACK | Bounded staging pool as comma-separated `size[:slots]` buckets (default `2147483648:4`). Sizes accept bytes or binary `k`/`m` suffixes; omitted slots default to one. Invalid profiles retain the built-in default. Multiple size buckets disable split comm and use the parent DevComm path; one bucket with multiple slots retains split support. |
| `NCCL_RESHARD_STAGING_NUM_CHANNELS` | DIRECT, PIPE | Channel count for the channelized staging pool (separate from PACK's pool above); default 4, or 8 for PIPE host-RMA. |
| `NCCL_RESHARD_STAGING_CHANNEL_SIZE` | DIRECT, PIPE | Per-channel data capacity for that pool; default 8 MiB, or 128 MiB for PIPE host-RMA. |
| `NCCL_RESHARD_STAGING_CHUNK_SIZE` | DIRECT, PIPE | Chunk size within a staging channel; default 1 MiB, or 32 MiB for PIPE host-RMA. |
| `NCCL_RESHARD_PIPE_NET_MODE` | PIPE (beta) | `DEVICE` (default, persistent CUDA kernel) or `HOST_RMA` (host-initiated NCCL one-sided put/signal). `HOST_RMA` requires NCCL 2.31.2 or newer (public tag `v2.31.2-1`); older NCCL can hang when a rank has no PIPE edges, so `ncclReshard` returns `ncclInvalidUsage` instead. |
| `NCCL_RESHARD_GEN_COMM_CONCURRENCY` | PACK, PIPE split-comm | Positive execution-concurrency limit, maximum 64, set before runtime initialization. PACK: an explicit value caps active physical slots in each bucket to `min(configured slots, limit)`; commB uses the same effective slot count. If unset, PACK preserves its configured profile, including profiles with more than four slots. PIPE: lane count, default 4; lane = agreed persistent namespace ID modulo concurrency. Parents sharing a lane are event-ordered but retain separate persistent signals/counters, and must be submitted in consistent relative order across generator ranks. No additional collective. |
| `NCCL_RESHARD_GEN_COMM_PARENT_CAPACITY` | PIPE split-comm | Reserved persistent parent namespaces per shared generator commB (default 8, maximum 64), independent of concurrency. Set before runtime initialization. A parent reuses its namespace across calls; IDs never wrap or recycle during the runtime epoch. Exceeding capacity fails collectively during parent setup. Signals/counters are reserved for all namespaces up front; execution contexts are sized by concurrency. Does not affect PACK. |

---

## Contributing

Issues and merge requests are tracked on this same GitLab project. When
reporting a bug, include:

- Cluster / GPU model and NCCL version.
- The mesh shape, tensor shape, shard dims, algorithm, and load-balance mode
  used.
- The minimal `reshard_bench` command line that reproduces the issue.

See `RELEASE.md` for release history.

---

## Third-party software

This product includes `nlohmann/json` (JSON for Modern C++), which is used
for JSON parsing and serialization.

| Component | Version | Source | License and copyright | Used by |
|---|---|---|---|---|
| [JSON for Modern C++](https://github.com/nlohmann/json) (`nlohmann/json`) | 3.12.0 | `json.hpp` from `nlohmann/json`, for example upstream `single_include/nlohmann/json.hpp`; bundled here as `third_party/nlohmann/json.hpp` | MIT License; Copyright (c) 2013-2025 Niels Lohmann | `benchmarks/reshard_model_bench.cu` parses model and system configuration JSON files. |

The applicable third-party copyright and license notice is included in this
product's [nlohmann/json MIT license file](third_party/nlohmann/LICENSE.MIT).

---

## License

Apache-2.0 in the NCCL contrib drop, inherited from the parent `nccl/nccl`
`LICENSE.txt`. The third-party license for the vendored `nlohmann/json`
dependency is in [`third_party/nlohmann/LICENSE.MIT`](third_party/nlohmann/LICENSE.MIT).
© NVIDIA Corporation.
