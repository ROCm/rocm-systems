# nccl4py: Python Bindings for RCCL

Python bindings for the [ROCm Communication Collectives Library
(RCCL)](https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl), AMD's drop-in replacement for
NVIDIA NCCL on ROCm. This package is a fork of the upstream NVIDIA
[nccl4py](https://github.com/NVIDIA/nccl/tree/master/bindings/nccl4py)
v0.2.0 and provides both low-level Cython bindings and a high-level
Pythonic API for RCCL collective operations.

## Highlights

- **Drop-in API:** keeps the upstream `nccl.bindings` /
  `nccl.core` import paths, so applications written against NCCL
  on CUDA work unchanged on RCCL on ROCm.
- **HIP-backed `cuda.core` shim:** ROCm hosts get
  `from cuda.core import ...` resolved via
  `nccl._hip_compat.cuda_core_shim` (backed by `hip-python`)
  without pulling NVIDIA `cuda-bindings` / `cuda-core`. The shim
  is registered in `sys.modules` on `import nccl`, so `from
  cuda.core import ...` resolves to the HIP backend after
  `import nccl` has run; it is **not** shipped as a top-level
  `cuda` package on disk to avoid shadowing co-installed
  distributions that contribute to the `cuda.*` namespace
  (`cuda-bindings` pulled transitively by PyTorch / RAPIDS /
  Triton / nsight tooling).
- **RCCL-only collectives:** wrappers for
  `ncclAllReduceWithBias` and `ncclAllToAllv` (which exist in
  `librccl.so` but have no upstream NCCL equivalent) are exposed
  via [`nccl.bindings.rocm_extensions`](nccl/bindings/rocm_extensions.pyx).
- **No CUDA toolchain at build time:** Cython extensions treat
  `cudaStream_t` as an opaque `void*`; `librccl.so` is resolved at
  runtime via `dlopen`. `pip install .` works on a ROCm host
  without `CUDA_HOME`.

## Requirements

- **Runtime:** `librccl.so` resolved at first use. Discovery order:
  1. `NCCL_LIBRARY=<path>` (explicit override; path or SONAME).
  2. `ctypes.util.find_library("rccl")` (the loader's `ldconfig`
     cache).
  3. `$ROCM_PATH/lib/librccl.so`, then `$HIP_PATH/lib/librccl.so`.
  4. Bare `librccl.so`, resolved via `LD_LIBRARY_PATH` / the standard
     loader search paths.
- **Python:** 3.10 or later.
- **Python deps:** `hip-python`, `numpy`, `packaging` (resolved
  automatically by `pip`).

## Installation

`pip install .` is the supported install path on ROCm:

```bash
pip install .
```

> **Note:** The `Makefile` and `CMakeLists.txt` shipped in this
> directory are inherited from upstream NVIDIA `nccl4py` and target
> NVIDIA hardware (they require `CUDA_HOME` / `nvcc` and pull
> CUDA-only `torch` / `cupy` wheels). They are out of scope for the
> ROCm bring-up and tracked separately under release engineering;
> use `pip install` on ROCm hosts.

For a development environment with the test extras:

```bash
pip install -e .
pip install pytest pytest-mpi pytest-cov mpi4py ml-dtypes
```

## Tests

The unit-style files (`tests/test_loader_stubs.py`,
`tests/test_shim_surface.py`, `tests/test_rocm_extensions.py`) run
under a single rank without any GPU and are useful as smoke tests:

```bash
pytest tests/test_loader_stubs.py tests/test_shim_surface.py tests/test_rocm_extensions.py
```

The collective-correctness and PyTorch / CuPy interop tests are
marked `@pytest.mark.mpi` and require `mpirun -np N` plus visible
GPUs:

```bash
mpirun -np 8 pytest -m mpi tests/
```

`tests/conftest.py` skips MPI-tests gracefully on hosts without
`mpi4py`, `torch`, or any visible GPU; the 8-rank tests `xfail`
when fewer than 8 GPUs are visible.

## Examples

[examples/01_basic/](examples/01_basic/) contains minimal,
self-contained scripts:

```bash
mpirun -np 4 python examples/01_basic/01_allreduce.py
mpirun -np 2 python examples/01_basic/02_send_recv.py
```

## Layout

- [`nccl/bindings/`](nccl/bindings/) - Cython bindings to the
  RCCL/NCCL C ABI; vendored from upstream `nccl4py` with the
  patches needed to drop the CUDA driver-types include and to
  resolve `librccl.so` via `dlopen`.
- [`nccl/_hip_compat/`](nccl/_hip_compat/) - HIP/ROCm-only
  compatibility layer; not a public API. Hosts the
  `cuda_core_shim` package registered in `sys.modules` as
  `cuda.core` by `nccl/__init__.py`.
- [`nccl/core/`](nccl/core/) - the high-level Pythonic surface
  (Communicator, Buffer, Group, ...).

## References

- [RCCL](https://github.com/ROCm/rccl)
- [RCCL documentation](https://rocm.docs.amd.com/projects/rccl/en/latest/)
- [Upstream NVIDIA nccl4py](https://github.com/NVIDIA/nccl/tree/master/bindings/nccl4py)
