# HotSwap

HotSwap is a **load-time ISA rewriter** for AMD GPUs. It lets a workload built
for one GPU architecture run on a different physical GPU by rewriting device
code as it is loaded — for example running a `gfx1250` (RDNA-class) workload on
a `gfx942` (MI300) or `gfx950` (MI350) host GPU.

Unlike [`rocjitsu`](../rocjitsu/) — which is a *software* simulator that decodes
the ISA in an event-driven core — HotSwap runs on **real hardware** and only
rewrites the parts of the program that the host GPU cannot execute natively.
Because the rewrite happens once, at code-object load time, it is much faster
than a full simulator while still letting you exercise code for an architecture
you don't physically have.

## What HotSwap actually is

HotSwap is **not** a single `libhsa-hotswap.so`. It is a set of co-installed
artifacts (the same ones the reference Docker recipe produces), staged into one
tree:

```
<hotswap-root>/
  lib/        libhotswap_intercept.so   # HIP intercept, LD_PRELOADed
              libhsa-runtime64.so.1     # HotSwap-patched ROCR runtime
              libamd_comgr.so           # COMGR transpiler
  llvm-tools/ llc llvm-mc lld ld.lld    # the transpiler shells out to these
  runtime/hotswap_py/                   # python adapter runtime
```

mirage wires these into a workload through the **HotSwap env contract**: the
patched ROCR + COMGR shadow the system copies via `LD_LIBRARY_PATH`, the HIP
intercept is `LD_PRELOAD`ed, and a few `HOTSWAP_*` variables select the source
target and adapter policy.

## Status

By default mirage does **not** build HotSwap — it *finds* and *uses* an existing
install. An opt-in CMake flag (`MIRAGE_BUILD_HOTSWAP`, see below) can build it
from source as part of the mirage build.

## Usage with mirage

Once a HotSwap tree is installed somewhere mirage can find it (see
[Where mirage looks](#where-mirage-looks)):

```sh
# Create a profile that uses HotSwap, then run under it:
mirage profile create rdna --emulator hotswap
mirage run --profile rdna -- ./my-rocm-app --flag
```

mirage discovers the install, sets `HOTSWAP_ENABLE`, `HOTSWAP_LIB_DIR`,
`HOTSWAP_SOURCE_TARGET` (default `gfx1250:32`), `HOTSWAP_ADAPTER_POLICY`
(default `compile`) and `HOTSWAP_PY_DIR`, points `LD_LIBRARY_PATH` at the lib
dir, and `LD_PRELOAD`s the intercept. Any of these can be overridden from the
exec environment.

If mirage can't find a HotSwap install, `mirage profile create --emulator
hotswap` fails with guidance describing exactly which locations were searched
and how to make it discoverable. mirage also fails loudly at run time rather
than silently running the workload unemulated.

### Where mirage looks

Discovery is anchored on `libhotswap_intercept.so`; the directory that contains
it is the HotSwap lib dir. mirage searches in the following locations, in order
(the first match wins):

1. `$HOTSWAP_LIB` / `$HSA_TOOLS_LIB` — an explicit path straight to the
   intercept `.so`.
2. `$HOTSWAP_LIB_DIR` — the HotSwap lib dir directly.
3. Any directory on `$LD_LIBRARY_PATH`.
4. `$ROCM_HOME` / `$ROCM_PATH` — the ROCm install root (`<root>/lib`).
5. `../lib` relative to the `mirage` binary.
6. Standard system / ROCm library directories: `/opt/rocm/lib`,
   `/usr/local/lib`, `/usr/lib`, `/usr/lib/x86_64-linux-gnu`.

The `llvm-tools/` and `runtime/hotswap_py/` directories are resolved relative to
the lib dir's parent (or via `$HOTSWAP_PY_DIR`).

## Installation

### Install a prebuilt tree

Place the `lib/`, `llvm-tools/` and `runtime/hotswap_py/` directories under any
location from [Where mirage looks](#where-mirage-looks), e.g.:

```sh
export HOTSWAP_LIB_DIR=/abs/path/to/hotswap/lib
# (llvm-tools/ and runtime/hotswap_py/ are expected next to that lib dir)
```

### Build from source (opt-in, via mirage's CMake)

HotSwap is built by a dedicated, opt-in CMake path. It is a full LLVM + COMGR +
ROCR source build, so it is **off by default**:

```sh
cmake -S . -B build -DMIRAGE_BUILD_HOTSWAP=ON
cmake --build build --target hotswap
```

This mirrors the reference Docker recipe and produces three artifact sets,
staged under `target/` (the lib lands in `target/lib`, which discovery searches
automatically):

1. **COMGR transpiler** (`libamd_comgr.so`) + the LLVM tools, from the
   `llvm-project` HotSwap fork. The in-tree
   [`../llvm-project-hotswap`](../llvm-project-hotswap) checkout is used as the
   source by default to avoid re-cloning llvm-project.
2. **HotSwap-patched ROCR** (`libhsa-runtime64.so.1`), from the `rocm-systems`
   HotSwap fork (built with `ROCR_ENABLE_HOTSWAP_COMGR_ADAPTER=ON`).
3. **HIP intercept** (`libhotswap_intercept.so`) + the python runtime, from the
   HotSwap testing repo.

Useful cache variables (see [`cmake/Hotswap.cmake`](cmake/Hotswap.cmake)
for the full list):

| Variable | Purpose | Default |
| --- | --- | --- |
| `MIRAGE_HOTSWAP_STAGE` | Where artifacts are staged | `target` |
| `MIRAGE_HOTSWAP_LLVM_SRC` | Existing llvm-project (HotSwap fork) checkout | `llvm-project-hotswap` |
| `MIRAGE_HOTSWAP_ROCR_REPO` / `_REF` | ROCR fork URL / ref | `martin-luecke/rocm-systems` @ `users/mluecke/hotswap-compatibility` |
| `MIRAGE_HOTSWAP_TESTING_REPO` / `_REF` | intercept repo URL / ref | `harsh-amd/rocm-hotswap-testing` @ `mluecke/hotswap-env-contract` |
| `MIRAGE_HOTSWAP_ROCM_PATH` | ROCm prefix for the ROCR build | `$ROCM_PATH` or `/opt/rocm` |
| `MIRAGE_HOTSWAP_JOBS` | Parallel build jobs | host CPU count |

The ROCR and intercept sources are private forks: clone over HTTPS using
whatever credentials git is configured with (a token in the URL, a credential
helper, or an SSH rewrite).

## How it works (under the hood)

```mermaid
flowchart TD
    ENV["HOTSWAP_* env contract<br/>LD_PRELOAD=libhotswap_intercept.so<br/>LD_LIBRARY_PATH=…/hotswap/lib"] -->|set by mirage| APP
    APP["ROCm app"] -->|HIP calls intercepted| INT["libhotswap_intercept.so"]
    INT -->|patched runtime shadows system| ROCR["libhsa-runtime64.so.1<br/>(HotSwap-patched ROCR)"]
    ROCR -->|transpiles gfx1250 → gfx942/gfx950<br/>at code-object load| COMGR["libamd_comgr.so<br/>(COMGR transpiler)"]
    COMGR -->|shells out to| TOOLS["llc / llvm-mc / lld"]
    ROCR -->|dispatches rewritten kernels| GPU["real GPU (gfx942 / gfx950)"]
```

mirage's job is discovery + wiring: it finds the HotSwap tree and sets the env
contract. The rewriting itself lives in the patched ROCR + COMGR.

## See also

- [`../rocjitsu/`](../rocjitsu/) — the software GPU simulator backend.
- [`cmake/Hotswap.cmake`](cmake/Hotswap.cmake) — the source build.
- [`../README.md`](../README.md) — the mirage CLI/dashboard overview.
- [HotSwap Design & Brainstorming Hub][confluence] (internal).

[confluence]: https://amd.atlassian.net/wiki/spaces/MLSE/pages/1620425029/HotSwap+Design+Brainstorming+Hub
