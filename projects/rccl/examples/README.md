<!--
  SPDX-License-Identifier: Apache-2.0
  Ported from the NVIDIA NCCL examples (https://github.com/NVIDIA/nccl/tree/master/docs/examples).
-->

# RCCL Library Examples

This directory contains RCCL (ROCm Communication Collectives Library) examples,
ported from the upstream
[NCCL examples](https://github.com/NVIDIA/nccl/tree/master/docs/examples).
Because RCCL exposes the same `ncclXxx` API surface as NCCL, the application
code is largely identical — only the underlying GPU runtime calls have been
translated from CUDA to HIP, and the build system retargeted to `hipcc` and
`librccl`.

The examples progress from basic concepts to advanced usage patterns, each with
its own README. See the
[RCCL documentation](https://rocm.docs.amd.com/projects/rccl/en/latest/) for a
full reference.

## Categories

| Folder | What it shows |
| --- | --- |
| `01_communicators` | Communicator creation: single-process multi-GPU, one device per pthread, one device per MPI process |
| `02_point_to_point` | Send/Recv in a simple ring pattern |
| `03_collectives` | Basic AllReduce |
| `04_user_buffer_registration` | `ncclCommRegister` to pre-register user buffers for faster collectives |
| `05_symmetric_memory` | `ncclCommWindowRegister` symmetric memory windows |
| `06_device_api` | Device-side API (kept as the RCCL-specific examples already present in the project) |

## Prerequisites

- ROCm (default: `/opt/rocm`). Override with `ROCM_PATH=...`.
- RCCL headers and library (shipped with ROCm; override location with `RCCL_HOME=...`).
- `hipcc` in `$ROCM_PATH/bin`.
- Optional: an MPI implementation for the MPI example.

## Build

From this directory:

```bash
make GPU_TARGETS="gfx942"            # build all 01..05 categories
make -C 03_collectives/01_allreduce  # build one example
make MPI=1 -C 01_communicators/03_one_device_per_process_mpi
```

Useful variables:

- `GPU_TARGETS` — space-separated AMD GPU archs, e.g. `"gfx90a gfx942 gfx1100"`. Defaults to those three.
- `ROCM_PATH` — ROCm install root (default `/opt/rocm`).
- `RCCL_HOME` — RCCL install root (default `$(ROCM_PATH)`).
- `MPI=1` — build with MPI support. Optionally set `MPI_HOME`.
- `PREFIX` — install prefix for `make install` (default `/usr/local`).

The build uses `hipcc` for both compilation and linking. For the MPI example,
`OMPI_CXX`/`MPICH_CXX` are exported so the MPI wrapper invokes `hipcc`.

## Run

```bash
./03_collectives/01_allreduce/allreduce
mpirun -np 2 ./01_communicators/03_one_device_per_process_mpi/one_device_per_process_mpi
```

By default the threaded/single-process examples use every visible GPU; restrict
with `HIP_VISIBLE_DEVICES=0,1,2,3`.

## Environment Variables (runtime)

- `HIP_VISIBLE_DEVICES` — comma-separated list of GPUs visible to the application.
- `NCCL_DEBUG=INFO` — verbose RCCL logging (RCCL honors the `NCCL_*` env vars).
- All other RCCL environment variables apply.

## Troubleshooting

- Use `NCCL_DEBUG=INFO` for detailed RCCL logging.
- Ensure your GPU's `gfx` arch is included in `GPU_TARGETS` (check with `rocminfo`).
- For MPI builds, confirm `mpicxx`/`mpirun` are on `PATH` or set `MPI_HOME`.
