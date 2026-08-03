# rocshmem4py test suite

Audience: contributors and CI maintainers running the `rocshmem4py`
test suite. End users of the package should start at the top-level
README:
<https://github.com/ROCm/rocm-systems/blob/develop/python/rocshmem/README.md>.

## Layout

| File | Scope |
|---|---|
| `test_smoke.py` | Single- and multi-PE torch-free tests via ctypes + HIP: `rocshmem_create_buffer`, `SymmetricBuffer` lifecycle, H2D/D2H roundtrip, stream-based put/get, `rocshmem_get_peer_buffer` |
| `test_basic.py` | Single-PE: constants, `SymmetricBuffer`, `interop.torch` tensor helpers, barrier |
| `test_collective.py` | Multi-PE: stream-based put/get, stream barriers, peer views, `interop.torch` RMA wrappers (data-verified) |
| `test_memory.py` | Single-PE torch-free tests via ctypes + HIP: `rocshmem_calloc` (zero-init verified), `rocshmem_align` (alignment + invalid-arg), `rocshmem_buffer_register` / `rocshmem_buffer_unregister` / `rocshmem_buffer_unregister_all` |
| `conftest.py` | Three-tier init ladder, `requires_torch` / `requires_multi_pe` markers |

## Running the tests

The tests are plain `pytest` — `conftest.py` handles rocSHMEM init/finalize and
PE detection, so all that's needed is to launch `pytest` across the desired
number of PEs. Build + install the binding first (see the top-level README),
run from the environment whose `python` / `torchrun` you intend to use, and put
the ROCm runtime on the loader path
(`export LD_LIBRARY_PATH=$CMAKE_PREFIX_PATH/lib:$ROCM_PATH/lib:$LD_LIBRARY_PATH`):

```bash
# IPC / GDA backend via torchrun (torch present)
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m pytest tests/ -v

# RO backend via a UCX-enabled Open MPI (must use mpirun)
mpirun --allow-run-as-root -n 2 \
  -mca pml ucx -mca osc ucx \
  -x ROCSHMEM_HEAP_SIZE=536870912 \
  -x LD_LIBRARY_PATH \
  -x WORLD_SIZE=2 \
  python3 -m pytest tests/ -v

# A specific test file: just name it
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m pytest tests/test_teams.py -v
```

To build + validate the wheel across *all* rocSHMEM backend configurations in
isolated venvs, use `validate_configs.py` in the package root (it rebuilds the
C++ library per config and checks the wheel's import, version tag, GPU arches,
and transitive deps; pass `--run-tests` to also run `test_api_compat.py`).

## Three-tier init ladder

`conftest.py` walks `init_with_torch` &rarr; `init_with_mpi` &rarr; raw
`rocshmem_init()` so the suite still runs on bare CI images that have
neither `torch` nor `mpi4py`. Verified on gfx942:

| Backend | `mpirun` + `init_with_torch` | `mpirun` + `init_with_mpi` | `mpirun` + raw `rocshmem_init()` | `torchrun` + `init_with_torch` |
|---|---|---|---|---|
| `ipc_single` / `ipc_vmm` (FINEGRAIN) | yes | yes | yes | yes |
| `all_backends` (GDA+RO+IPC, FINEGRAIN) | yes | yes | yes | yes |
| `ro_net` (RO only) | yes | yes | yes | no &mdash; RO requires `mpirun`, see top-level README |
| `ipc_vmm` with `USE_HEAP_DEVICE_VMM_POSIX=ON` | yes | yes | no &mdash; VMM-POSIX guard | yes |

The third tier (raw `rocshmem_init()`) is incompatible with
`-DUSE_HEAP_DEVICE_VMM_POSIX=ON` builds by source-level design
(`src/rocshmem.cpp:161-165` `LOG_ERROR_EXIT`). The VMM-POSIX
allocator's `pidfd_getfd`-based IPC handshake cannot survive the
MPI-Comm bootstrap order; only the unique-ID `TcpBootstrap` path is
safe (which `init_with_torch` and `init_with_mpi` both take). The
conftest detects VMM-POSIX at runtime and skips tier 3 only on those
builds &mdash; end users don't hit this because the public init helpers
never use the raw path.

## Tests are backend-agnostic

The tests exercise the **binding layer** (argument passing, tensor /
`__cuda_array_interface__` round-trips, `SymmetricBuffer` RAII, PyTorch
integration) through the portable rocSHMEM surface that every backend
implements:

- `rocshmem_barrier_all` / `rocshmem_barrier_all_on_stream`
- `rocshmem_putmem_on_stream` / `rocshmem_getmem_on_stream`
- `rocshmem_putmem_signal_on_stream` / `rocshmem_signal_wait_until_on_stream`
- `rocshmem_ptr` / `rocshmem_create_buffer` / `rocshmem_get_peer_buffer`

This matches the wheel itself: `_rocshmem4py.cc` has no
backend-specific `#ifdef`s.
