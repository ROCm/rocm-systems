# rocshmem4py: Python Bindings for rocSHMEM

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

`rocshmem4py` provides Python bindings for the ROCm OpenSHMEM (rocSHMEM) runtime library, enabling GPU-centric networking through an OpenSHMEM-like interface on AMD ROCm platforms.

## Features

- **Core rocSHMEM API**: Memory management, data transfer, atomics, synchronization
- **Framework-agnostic allocation**: `rocshmem_create_buffer` / `rocshmem_get_peer_buffer` work without any ML framework
- **`__cuda_array_interface__`**: Zero-copy interop between `SymmetricBuffer` and PyTorch tensors
- **PyTorch interop submodule**: `rocshmem4py.interop.torch` for tensor allocation and tensor-aware RMA (`put`, `get`, `barrier_all`)
- **PyTorch coordination**: `init_with_torch()` / `finalize_with_torch()` for torch.distributed-based init
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
from rocshmem4py.interop import torch as rshmem_torch

# RO backend:   mpirun -n 2 python my_script.py
# IPC/GDA:      torchrun --standalone --nproc_per_node=2 my_script.py
rocshmem4py.init_with_torch()

my_pe = rocshmem4py.rocshmem_my_pe()
n_pes = rocshmem4py.rocshmem_n_pes()

# Allocate a symmetric tensor (backed by rocshmem_malloc)
src = rshmem_torch.create_tensor((64,), torch.float32)
dst = rshmem_torch.create_tensor((64,), torch.float32)
src.fill_(float(my_pe))
dst.fill_(-1.0)
torch.cuda.synchronize()

# Stream-ordered transfer to the next PE (portable across all backends)
peer = (my_pe + 1) % n_pes
rshmem_torch.barrier_all()
rshmem_torch.put(dst, src, peer)
rshmem_torch.barrier_all()
torch.cuda.synchronize()

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

#### Framework-agnostic (`rocshmem4py`)

| Function | Description |
|---|---|
| `rocshmem_malloc(size)` | Allocate symmetric memory (returns raw pointer) |
| `rocshmem_free(ptr)` | Free symmetric memory |
| `rocshmem_ptr(dest, pe)` | Get remote symmetric pointer (IPC backends only) |
| `SymmetricBuffer(size)` | RAII wrapper; exposes `__cuda_array_interface__` for zero-copy torch interop |
| `rocshmem_create_buffer(nbytes)` | Collective allocation returning a `SymmetricBuffer` |
| `rocshmem_get_peer_buffer(buf, peer)` | Non-owning `SymmetricBuffer` view of a peer's buffer (IPC only) |

#### PyTorch interop (`rocshmem4py.interop.torch`)

| Function | Description |
|---|---|
| `create_tensor(shape, dtype)` | Collective allocation returning a symmetric `torch.Tensor` |
| `get_peer_tensor(tensor, peer)` | Zero-copy tensor view of a peer's symmetric tensor (IPC only) |
| `free_tensor(tensor)` | Explicit collective-safe deallocation |
| `put(dst, src, peer, stream=None)` | Stream-ordered put (all backends) |
| `get(dst, src, peer, stream=None)` | Stream-ordered get (all backends) |
| `barrier_all(stream=None)` | Stream-ordered collective barrier |

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

## Running the tests

```bash
# IPC / GDA backend (any AMD multi-GPU node)
torchrun --standalone --nproc_per_node=2 -m pytest tests/ -v

# RO backend (must use mpirun)
./launch_test.sh -n 2 -c "pytest tests/ -v"
```

Test source layout, the full launcher &times; backend &times; init-tier
matrix used by `conftest.py`, and CI-author guidance live in the test
suite's own README:
<https://github.com/ROCm/rocm-systems/blob/develop/projects/rocshmem/python/tests/README.md>.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: _rocshmem4py` | rocSHMEM not on the loader path | `export LD_LIBRARY_PATH=$ROCSHMEM_HOME/lib:$ROCM_PATH/lib:$LD_LIBRARY_PATH` |
| CMake cannot find rocSHMEM at build time | `ROCSHMEM_HOME` unset | `export ROCSHMEM_HOME=/path/to/rocshmem/build` before `pip install -e .` |
| Link error: `recompile with -fPIC` | `librocshmem.a` built without PIC | Rebuild rocSHMEM with `-DCMAKE_POSITION_INDEPENDENT_CODE=ON` |
| MPI / UCX import or runtime errors | OpenMPI not built with UCX | Use an OpenMPI install with UCX and point `OMPI_DIR` at it |
| `Unsupported configuration to initialize rocSHMEM. Please initialize the MPI library using MPI_Init first` | RO backend launched under `torchrun` &mdash; see *"RO backend requires `mpirun`"* below | Launch with `mpirun` (you can keep `init_with_torch()`), or build an IPC/GDA-only rocSHMEM if you don't need inter-node RDMA |
| Rendezvous / port conflict under `torchrun` | Default `MASTER_PORT=29500` already taken | Set `MASTER_PORT` (or `ROCSHMEM_MASTER_PORT`) to a free port |

### RO backend requires `mpirun`

This is a structural property of rocSHMEM's C library, not a packaging
gap in `rocshmem4py`: `library_init_subcomm` (in `src/rocshmem.cpp`)
requires either `MPI_Initialized()` to be true or the OpenMPI launcher
env var `OMPI_COMM_WORLD_SIZE` to be set. `mpirun`/`prterun` exports
those env vars; `torchrun` does not. Pre-importing `mpi4py` is **not** a
workaround &mdash; it makes `MPI_Initialized()` return true, which then
routes the C library through a subgroup-creation path
(`MPI_Group_incl` + `MPI_Comm_create_group`) across processes that live
in disjoint singleton MPI universes, and that crashes inside OMPI's PMIx
wireup. Use `mpirun` for RO; `init_with_torch()` itself still works
under `mpirun`, so you keep the `torch.distributed` unique-id exchange
and only the launcher changes.

### Diagnosing which rocSHMEM backend you actually linked

A pre-built `_rocshmem4py.so` (or any wheel you might pick up later)
statically links *one* rocSHMEM backend. When initialization fails in
ways that look backend-specific, two checks pin down which one:

```bash
# What backend does the rocSHMEM install at $ROCSHMEM_HOME advertise?
"${ROCSHMEM_HOME}/bin/rocshmem_info" | grep "Vendor String"

# Which backend's code is actually linked into the loaded extension?
nm -D --defined-only "$(python -c 'import _rocshmem4py; print(_rocshmem4py.__file__)')" \
  | grep -E " T _ZN8rocshmem(9RO|10IPC|10GDA)Backend" | head
```

The auto-detect order is IPC &rarr; GDA &rarr; RO; force a specific one
with `ROCSHMEM_BACKEND=ipc` if the build supports it. The same
`_rocshmem4py` source builds against any backend &mdash; there are no
backend `#ifdef`s in the binding layer &mdash; so two wheels with
identical Python APIs can behave differently at runtime depending on
how the C library was configured.

## License

MIT License. See [LICENSE](LICENSE) for details.
