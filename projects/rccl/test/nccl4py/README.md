# RCCL nccl4py Build-Smoke Tests

## Description

This directory contains a pytest harness that validates the **uv / CMake
build path** for nccl4py -- the same path NCCL 2.29.7 shipped as
*"Added CMake for NCCL4Py build"*:

```bash
cmake -DBUILD_NCCL4PY=ON <rccl-build-dir>
cmake --build <rccl-build-dir> --target nccl4py   # -> uv build -> wheel
```

The harness mirrors `test/ir-device`: it drives the CMake target on demand
from a session-scoped fixture and **skips with a clear reason** (rather than
failing) when prerequisites are missing.

What it does, per session:

1. `cmake -DBUILD_NCCL4PY=ON <RCCL_BUILD>` -- (re)configure the existing RCCL
   build tree so the `nccl4py` target is reachable (exercises the
   root-`CMakeLists.txt` `BUILD_NCCL4PY` wiring added for RCCL).
2. `cmake --build <RCCL_BUILD> --target nccl4py` -- build the wheel via
   `uv build` into `<RCCL_BUILD>/dist`.
3. `uv venv` + `uv pip install <wheel> pytest` -- stand up an isolated
   Python 3.13 environment matching the wheel ABI.
4. Run the CPU/GPU smoke modules in `bindings/nccl4py/tests/` with that
   environment's interpreter.

It exercises:

| Module | What it checks |
|--------|----------------|
| (wheel artifact) | `cmake --build --target nccl4py` produced a `.whl` |
| `test_rocm_extensions.py` | RCCL-only wrappers (`all_reduce_with_bias`, `all_to_all_v`) fail controlled |
| `test_shim_surface.py` | HIP `cuda.core` shim (optional; self-skips without visible GPUs) |

The harness also runs an out-of-process `import nccl.bindings` check against
the uv env after the wheel is installed. The `test_loader_stubs.py` module
under `bindings/nccl4py/tests` is RCCL-version-specific (it expects certain
symbols to be absent) and is not part of this build-smoke suite.

## Prerequisites

Build RCCL once so `librccl.so` exists **and** `<RCCL_BUILD>` is a configured
CMake tree:

```bash
cmake -B build/release -DBUILD_TESTS=ON .
cmake --build build/release --target rccl
```

`uv` must be installed and on `PATH` (the `nccl4py` CMake target is
uv-driven): <https://docs.astral.sh/uv/getting-started/installation/>.

## Configuration (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `RCCL_DIR` | repo root (derived) | RCCL source root |
| `RCCL_BUILD` | `$RCCL_DIR/build/release` | RCCL CMake build dir (`librccl.so`, dist/) |
| `ROCM_PATH` | `/opt/rocm` | ROCm install root |
| `NCCL4PY_DIR` | `$RCCL_DIR/bindings/nccl4py` | nccl4py source tree |
| `NCCL4PY_VENV` | `<workdir>/nccl4py_uvenv` | uv venv the wheel is installed into |
| `UV_PYTHON` | `3.13` | Python the uv venv/wheel target uses |
| `CMAKE` | `cmake` | cmake executable |
| `UV` | `uv` | uv executable |
| `NCCL_LIBRARY` | (auto) | Explicit path to `librccl.so` |

## Running

```bash
cd test/nccl4py
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Build nccl4py via uv/CMake + run smoke modules
pytest -v --cache-clear

# CPU-only smoke (no GPU required)
pytest -v -m nccl4py_cpu
```

Build, env, and per-module logs are written to `logs/`.

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
      "description": "uv/CMake nccl4py wheel build + CPU smoke pytest modules",
      "test_filter": "ALL"
    }
  ]
}
```

On runners without a prior RCCL build (or without `uv`), every case
auto-skips.
