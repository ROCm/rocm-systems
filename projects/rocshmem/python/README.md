# rocshmem4py: Python Bindings for rocSHMEM

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

`rocshmem4py` provides Python bindings for the ROCm OpenSHMEM (rocSHMEM) runtime library, enabling GPU-centric networking through an OpenSHMEM-like interface on AMD ROCm platforms.

## Features

- **Core rocSHMEM API**: Memory management, data transfer, atomics, synchronization
- **PyTorch Integration**: `init_with_torch()` / `finalize_with_torch()` for seamless PyTorch workflows
- **Symmetric Tensor Helpers**: `rocshmem_create_tensor`, `symm_rocshmem_tensor`, `rocshmem_create_tensor_list_intra_node` matching the Triton-distributed `pyrocshmem` API
- **`__cuda_array_interface__`**: Zero-copy interop between `SymmetricBuffer` and PyTorch tensors
- **MPI Integration**: `init_with_mpi()` for existing MPI applications

## API Coverage

`rocshmem4py` currently exposes a focused host-side subset of rocSHMEM APIs:
initialization/finalization, PE/team queries, memory allocation, put/get
(blocking/non-blocking/stream variants), selected atomics, and key constants.
It does not yet expose every rocSHMEM host API.

Host-facing symbols are exported directly from the compiled extension module
(`_rocshmem4py`) to avoid drift between Python imports and C++ bindings.

## Prerequisites

- AMD ROCm 6.0+ with HIP
- rocSHMEM library (built with RO, IPC, or GDA backend)
- Python 3.8+
- CMake 3.20+
- pybind11 2.13.1+
- OpenMPI with UCX support (required for the RO backend)
- `mpi4py` (only required when using `init_with_mpi()`)

## Installation

```bash
export ROCM_PATH=/opt/rocm
export ROCSHMEM_HOME=/path/to/rocshmem/build
export LD_LIBRARY_PATH=$ROCSHMEM_HOME/lib:$ROCM_PATH/lib:$LD_LIBRARY_PATH

pip install pybind11 cmake
pip install -e .
```

> Note: this package is not published to PyPI yet; install from source only.

## Quick Start

### Backend launch matrix

| Backend | Requires MPI runtime | Recommended launcher | Example |
|---|---|---|---|
| RO      | Yes | `mpirun`   | `mpirun -n 2 python my_script.py` |
| IPC/GDA | No  | `torchrun` | `torchrun --standalone --nproc_per_node=2 my_script.py` |

`init_with_torch()` works under both launchers. `init_with_mpi()` requires an
`mpirun`-launched process environment.

### With PyTorch coordination (recommended)

```python
import torch
import rocshmem4py

# RO backend:   mpirun -n 2 python my_script.py
# IPC/GDA:      torchrun --standalone --nproc_per_node=2 my_script.py
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

### With MPI coordination

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
| `ROCSHMEM_SIGNAL_SET` | impl-defined | Signal set op enum |
| `ROCSHMEM_SIGNAL_ADD` | impl-defined | Signal add op enum |
| `ROCSHMEM_CMP_EQ/NE/GT/GE/LT/LE` | impl-defined | Signal wait compare enums |

## Testing

Canonical test sources live in `python/tests/`.
When `BUILD_PYTHON_TESTS=ON`, CMake installs those test assets into the test package.

The RO backend **must be launched with `mpirun`** — UCX/OMPI runtime must be
present. Within an `mpirun` launch you may choose either `init_with_torch()`
(uses `torch.distributed` for the rocSHMEM unique-id exchange) or
`init_with_mpi()` (uses `mpi4py`). `torchrun` alone is **not** sufficient for
the RO backend.

Non-RO backends (IPC / GDA) have no MPI runtime dependency and should be
launched with `torchrun`.

Use `launch_test.sh` (backend-aware) or invoke the launcher directly:

```bash
# RO backend (default, via mpirun)
./launch_test.sh -n 2 -c "pytest tests/ -v"

# IPC / GDA backend (via torchrun)
./launch_test.sh -l torchrun -n 2 -c "pytest tests/ -v"

# Direct mpirun (RO)
mpirun --allow-run-as-root -n 2 \
  -mca pml ucx -mca osc ucx \
  -x ROCSHMEM_HEAP_SIZE=536870912 \
  -x LD_LIBRARY_PATH \
  -x WORLD_SIZE=2 \
  python3 -m pytest tests/ -v

# Direct torchrun (IPC / GDA)
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m pytest tests/ -v
```

If you hit rendezvous port conflicts under `torchrun`, set `MASTER_PORT`
(or `ROCSHMEM_MASTER_PORT`) explicitly.

### Test Files

| File | Scope |
|---|---|
| `test_basic.py` | Single-PE: constants, SymmetricBuffer, tensor helpers, barrier |
| `test_collective.py` | Multi-PE: stream-based put/get, stream barriers, peer views (data-verified) |

### Design: the wheel and its tests are backend-agnostic

`_rocshmem4py` statically links `librocshmem.a` at build time against a
specific rocSHMEM install, but its **source** contains no backend-specific
`#ifdef`s — the same extension source builds cleanly against any rocSHMEM
backend (RO / IPC / GDA). No compile-time backend macros leak into the
Python API.

The tests follow the same principle: they exercise the **binding layer**
(argument passing, tensor / `__cuda_array_interface__` round-trips,
`SymmetricBuffer` RAII, PyTorch integration) through the portable rocSHMEM
surface that every backend implements:

- `rocshmem_barrier_all` / `rocshmem_barrier_all_on_stream`
- `rocshmem_putmem_on_stream` / `rocshmem_getmem_on_stream`
- `rocshmem_putmem_signal_on_stream` / `rocshmem_signal_wait_until_on_stream`
- `rocshmem_ptr` and symmetric tensor helpers

## Troubleshooting

**ImportError**: Ensure rocSHMEM libraries are in `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=$ROCSHMEM_HOME/lib:$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

**CMake cannot find rocSHMEM**: Set `ROCSHMEM_HOME`:
```bash
export ROCSHMEM_HOME=/path/to/rocshmem/build
```

**Link error mentions `recompile with -fPIC`**: Build `rocshmem` with PIC enabled
when linking its static archive into `_rocshmem4py`:
```bash
cmake -S /path/to/rocshmem -B /path/to/rocshmem/build \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build /path/to/rocshmem/build --parallel
cmake --install /path/to/rocshmem/build
```

**MPI / UCX issues**: Ensure OpenMPI is built with UCX support and `OMPI_DIR` points to it.

## License

MIT License. See [LICENSE](LICENSE) for details.
