# RCCL nccl4py Build-Smoke Tests

## Description

This directory contains a pytest harness that validates the **supported ROCm
install path** for nccl4py:

```bash
cd bindings/nccl4py && pip install -e .
```

The harness mirrors `test/ir-device`: it builds the package on demand from a
session-scoped fixture and **skips with a clear reason** (rather than failing)
when prerequisites are missing.

It exercises the CPU-only smoke modules in `bindings/nccl4py/tests/`:

| Module | What it checks |
|--------|----------------|
| `test_rocm_extensions.py` | RCCL-only wrappers (`all_reduce_with_bias`, `all_to_all_v`) fail controlled |
| `test_shim_surface.py` | HIP `cuda.core` shim (optional; self-skips without visible GPUs) |

The harness also runs an in-process ``import nccl.bindings`` check after
``pip install -e .``. The ``test_loader_stubs.py`` module under
``bindings/nccl4py/tests`` is RCCL-version-specific (it expects certain
symbols to be absent) and is not part of this build-smoke suite.

The upstream NVIDIA `bindings/nccl4py/CMakeLists.txt` (CUDA + `uv` wheel
build) is **not** what this suite validates. On ROCm, `pip install` is the
supported path.

## Prerequisites

Build RCCL once so `librccl.so` exists:

```bash
cmake -B build/release -DBUILD_TESTS=ON .
cmake --build build/release --target rccl
```

## Configuration (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `RCCL_DIR` | repo root (derived) | RCCL source root |
| `RCCL_BUILD` | `$RCCL_DIR/build/release` | RCCL CMake build dir (`librccl.so`) |
| `ROCM_PATH` | `/opt/rocm` | ROCm install root |
| `NCCL4PY_DIR` | `$RCCL_DIR/bindings/nccl4py` | nccl4py source tree |
| `NCCL_LIBRARY` | (auto) | Explicit path to `librccl.so` |

## Running

```bash
cd test/nccl4py
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Build nccl4py via pip + run smoke modules
pytest -v --cache-clear

# CPU-only smoke (no GPU required)
pytest -v -m nccl4py_cpu
```

Build and per-module logs are written to `logs/`.

## CI / test_runner integration

Register in a `test_runner` JSON config like `ir_device`:

```json
"nccl4py_build_smoke": {
  "extends": "default",
  "is_pytest": true,
  "setup_venv": true,
  "test_dir": "test/nccl4py",
  "num_ranks": 1,
  "num_nodes": 1,
  "num_gpus": 0,
  "timeout": 600,
  "tests": [
    {
      "name": "NCCL4Py_BuildSmoke_All",
      "description": "pip install -e nccl4py + CPU smoke pytest modules",
      "test_filter": "ALL"
    }
  ]
}
```

On runners without a prior RCCL build, every case auto-skips.
