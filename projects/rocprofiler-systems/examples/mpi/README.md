# MPI

## Overview

This example benchmarks various MPI communication patterns including point-to-point (`MPI_Send`/`MPI_Recv`) and collective operations (`MPI_Alltoall`, `MPI_Allgather`, `MPI_Bcast`, `MPI_Reduce`, `MPI_Scatter`/`MPI_Gather`, `MPI_Allreduce`). The main C++ program exercises send/recv and all-to-all with different data types across duplicated and split communicators, while individual C programs each demonstrate a specific MPI collective. This suite is useful for profiling MPI interposition (`ROCPROFSYS_USE_MPIP`) and understanding communication overhead in distributed workloads.

## Source Files

- `mpi.cpp` - Main benchmark with `all2all<Tp,N>()` and `send_recv<Tp,N>()` templated over data types, plus communicator duplication and splitting by prime/non-prime rank.
- `allgather.c` - MPI_Allgather example
- `allreduce.c` - MPI_Allreduce example
- `all2all.c` - MPI_Alltoall example
- `bcast.c` - MPI_Bcast example
- `reduce.c` - MPI_Reduce example
- `scatter-gather.c` - MPI_Scatter/MPI_Gather example
- `send-recv.c` - MPI_Send/MPI_Recv example

## Prerequisites

- CMake 3.21+
- MPI implementation (OpenMPI, MPICH, etc.)
- C and C++ compilers

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/mpi
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/
cmake --build <build_dir> --target mpi-example
```

**Targets:**

| Target | Description |
|--------|-------------|
| `mpi-example` | C++ benchmark (send/recv + all2all with communicator ops) |
| `mpi-allgather` | C allgather example |
| `mpi-allreduce` | C allreduce example |
| `mpi-all2all` | C all-to-all example |
| `mpi-bcast` | C broadcast example |
| `mpi-reduce` | C reduce example |
| `mpi-scatter-gather` | C scatter/gather example |
| `mpi-send-recv` | C send/recv example |

## Running

```bash
# Run with 2 MPI processes
mpirun -np 2 ./mpi-example

# Run individual collective tests
mpirun -np 4 ./mpi-allreduce
mpirun -np 4 ./mpi-scatter-gather
```

**Arguments (mpi-example):**

| Position | Description | Default |
|----------|-------------|---------|
| 1 | Number of iterations | 1 |

## Profiling with rocprofiler-systems

```bash
mpirun -np 2 rocprof-sys-run -- ./mpi-example
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_USE_MPIP` | `ON` | Enable MPI profiling interposition |
| `ROCPROFSYS_USE_MPI` | `ON` | Enable MPI-aware output merging |
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile |
| `ROCPROFSYS_USE_SAMPLING` | `ON` | Enable sampling |

```bash
mpirun -np 2 rocprof-sys-run \
    -e ROCPROFSYS_USE_MPIP=ON \
    -e ROCPROFSYS_TRACE=true \
    -- ./mpi-example
```

For flat profiling with collapsed processes/threads:

```bash
mpirun -np 2 rocprof-sys-run \
    -e ROCPROFSYS_USE_MPIP=ON \
    -e ROCPROFSYS_FLAT_PROFILE=ON \
    -e ROCPROFSYS_COLLAPSE_PROCESSES=ON \
    -e ROCPROFSYS_COLLAPSE_THREADS=ON \
    -- ./mpi-example
```
