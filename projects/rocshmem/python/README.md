# rocshmem4py: Python Bindings for rocSHMEM

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

`rocshmem4py` provides Python bindings for the ROCm Shared Memory (rocSHMEM) library, enabling high-performance communication for GPU-accelerated applications on AMD ROCm platforms.

## Features

- **Core rocSHMEM API**: Memory management, data transfer, atomics, synchronization
- **PyTorch Integration**: `init_with_torch()` / `finalize_with_torch()` for seamless PyTorch workflows
- **Symmetric Tensor Helpers**: `rocshmem_create_tensor`, `symm_rocshmem_tensor`, `rocshmem_create_tensor_list_intra_node` matching the Triton-distributed `pyrocshmem` API
- **`__cuda_array_interface__`**: Zero-copy interop between `SymmetricBuffer` and PyTorch tensors
- **MPI Integration**: `init_with_mpi()` for existing MPI applications

## Prerequisites

- AMD ROCm 6.0+ with HIP
- rocSHMEM library (built with RO, IPC, or GDA backend)
- Python 3.7+
- CMake 3.16+
- pybind11 2.6.0+
- OpenMPI with UCX support (required for the RO backend)

## Installation

```bash
export ROCM_PATH=/opt/rocm
export ROCSHMEM_HOME=/path/to/rocshmem/build
export LD_LIBRARY_PATH=$ROCSHMEM_HOME/lib:$ROCM_PATH/lib:$LD_LIBRARY_PATH

pip install pybind11 cmake
pip install -e .
```

## Quick Start

### With PyTorch (recommended)

```python
import torch
import rocshmem4py

# Launch: mpirun -n 2 python my_script.py
rocshmem4py.init_with_torch()

my_pe = rocshmem4py.rocshmem_my_pe()
n_pes = rocshmem4py.rocshmem_n_pes()

# Allocate a symmetric tensor (backed by rocshmem_malloc)
t = rocshmem4py.rocshmem_create_tensor((64,), torch.float32)
t.fill_(float(my_pe))
torch.cuda.synchronize()

# Transfer data to the next PE
peer = (my_pe + 1) % n_pes
rocshmem4py.rocshmem_barrier_all()
rocshmem4py.rocshmem_putmem(
    t.data_ptr(), t.data_ptr(), t.nbytes, peer)
rocshmem4py.rocshmem_quiet()
rocshmem4py.rocshmem_barrier_all()

rocshmem4py.finalize_with_torch()
```

### With MPI

```python
from mpi4py import MPI
import rocshmem4py

rocshmem4py.init_with_mpi(MPI.COMM_WORLD)

my_pe = rocshmem4py.rocshmem_my_pe()
buf = rocshmem4py.SymmetricBuffer(1024)
rocshmem4py.rocshmem_barrier_all()

buf.free()
rocshmem4py.rocshmem_finalize()
```

## API Reference

### Initialization / Finalization

| Function | Description |
|---|---|
| `init_with_torch(group=None)` | Init rocSHMEM via torch.distributed (recommended) |
| `finalize_with_torch()` | Synchronized teardown of rocSHMEM + torch.distributed |
| `init_with_mpi(comm)` | Init rocSHMEM via mpi4py |
| `init_rocshmem_by_uniqueid(group)` | Low-level init with a torch process group |
| `rocshmem_init()` | Raw rocSHMEM init (rarely needed directly) |
| `rocshmem_finalize()` | Raw rocSHMEM finalize |
| `rocshmem_init_attr(rank, nranks, uid)` | Init with unique ID |
| `rocshmem_get_uniqueid()` | Get a unique ID for init_attr |

### PE Queries

| Function | Description |
|---|---|
| `rocshmem_my_pe()` | PE number of the calling process |
| `rocshmem_n_pes()` | Total number of PEs |
| `rocshmem_team_my_pe(team)` | PE number within a team |
| `rocshmem_team_n_pes(team)` | Number of PEs in a team |

### Memory Management

| Function | Description |
|---|---|
| `rocshmem_malloc(size)` | Allocate symmetric memory (returns pointer) |
| `rocshmem_free(ptr)` | Free symmetric memory |
| `rocshmem_ptr(dest, pe)` | Get remote symmetric pointer |
| `SymmetricBuffer(size)` | RAII wrapper with `__cuda_array_interface__` |
| `rocshmem_create_tensor(shape, dtype)` | Allocate symmetric memory as a PyTorch tensor |
| `symm_rocshmem_tensor(tensor, peer)` | View a symmetric tensor on a remote PE |
| `rocshmem_create_tensor_list_intra_node(shape, dtype)` | Symmetric tensor views for all PEs |

### Data Transfer

| Function | Description |
|---|---|
| `rocshmem_putmem(dest, src, nbytes, pe)` | Blocking put |
| `rocshmem_getmem(dest, src, nbytes, pe)` | Blocking get |
| `rocshmem_putmem_nbi(dest, src, nbytes, pe)` | Non-blocking put |
| `rocshmem_getmem_nbi(dest, src, nbytes, pe)` | Non-blocking get |
| `rocshmem_putmem_on_stream(dest, src, nbytes, pe, stream)` | Stream-ordered put |
| `rocshmem_getmem_on_stream(dest, src, nbytes, pe, stream)` | Stream-ordered get |
| `rocshmem_putmem_signal_on_stream(...)` | Stream-ordered put with signal |
| `rocshmem_signal_wait_until_on_stream(...)` | Stream-ordered signal wait |

### Synchronization

| Function | Description |
|---|---|
| `rocshmem_barrier_all()` | Barrier across all PEs |
| `rocshmem_barrier_all_on_stream(stream)` | Stream-ordered barrier |
| `rocshmem_fence()` | Ordering fence |
| `rocshmem_quiet()` | Wait for all outstanding operations |

### Atomic Operations

| Function | Description |
|---|---|
| `rocshmem_int_atomic_fetch_add(dest, value, pe)` | Atomic int fetch-and-add |
| `rocshmem_long_atomic_fetch_add(dest, value, pe)` | Atomic long fetch-and-add |
| `rocshmem_int_atomic_compare_swap(dest, cond, value, pe)` | Atomic int CAS |

### Constants

| Constant | Value | Description |
|---|---|---|
| `ROCSHMEM_TEAM_WORLD` | `0` | Team containing all PEs |
| `ROCSHMEM_TEAM_INVALID` | `-1` | Invalid team identifier |
| `ROCSHMEM_SUCCESS` | `0` | Success status code |

## Testing

Tests require MPI-launched processes (the RO backend needs MPI). Use `launch_test.sh` or invoke `mpirun` directly:

```bash
# Using the launch script
./launch_test.sh -n 2 -c "pytest tests/ -v"

# Direct mpirun
mpirun --allow-run-as-root -n 2 \
  -mca pml ucx -mca osc ucx \
  -x ROCSHMEM_HEAP_SIZE=536870912 \
  -x LD_LIBRARY_PATH \
  -x WORLD_SIZE=2 \
  python3 -m pytest tests/ -v
```

### Test Files

| File | Scope |
|---|---|
| `test_basic.py` | Single-PE: constants, SymmetricBuffer, tensor helpers, sync |
| `test_collective.py` | Multi-PE: putmem, getmem, atomics, stream ops (data-verified) |

## Troubleshooting

**ImportError**: Ensure rocSHMEM libraries are in `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=$ROCSHMEM_HOME/lib:$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

**CMake cannot find rocSHMEM**: Set `ROCSHMEM_HOME`:
```bash
export ROCSHMEM_HOME=/path/to/rocshmem/build
```

**MPI / UCX issues**: Ensure OpenMPI is built with UCX support and `OMPI_DIR` points to it.

## License

MIT License. See [LICENSE](LICENSE) for details.
