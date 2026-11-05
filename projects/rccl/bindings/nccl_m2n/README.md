# nccl.m2n on ROCm

`nccl.m2n` (DTensor-style reshard between disjoint rank groups) from upstream
[NVIDIA/nccl-extensions](https://github.com/NVIDIA/nccl-extensions) `python/`, `nccl_m2n` part only, at
the commit in `UPSTREAM_VERSION`. It builds on the RCCL port of nccl4py (`../nccl4py`, whose
`nccl.bindings.cynccl` the Cython bindings use) and loads the `libnccl_m2n.so` that
`projects/rccl/contrib/nccl_m2n/Makefile.rocm` builds. `scripts/build.sh` stages it as
`nccl/m2n/lib/rocm/libnccl_m2n.so` in the wheel (`NCCL_EXTENSIONS_REQUIRE_NATIVE_LIBS=1` fails the build
without it); `NCCL_M2N_LIBRARY` overrides it. ROCm mode is chosen by the installed nccl4py (the RCCL
port), not by probing the GPU runtime.

## Why the extra environment (workaround)

With upstream on CUDA, `pip install` is enough: torch, nccl4py and `libnccl_m2n` share one libnccl. Here
torch from TheRock ships its own `librccl.so.1` (RCCL 2.28.9), which lacks `ncclPutSignal`, `ncclSignal`,
`ncclWaitSignal`, `ncclGetLsaDevicePointer` and `ncclCommQueryProperties`. Loading `nccl.m2n` next to it
maps two RCCLs, or binds `libnccl_m2n` to the old one. So the validation setup loads the RCCL built from this
tree first: `LD_PRELOAD`, `NCCL_M2N_RCCL_LIBRARY` and `rocm/sitecustomize.py` on `PYTHONPATH`
(`contrib/nccl_m2n/scripts/py_env.sh`). `sitecustomize.py` is test-only: it writes the private
`rocm_sdk._ALL_CDLLS`, which a TheRock update can change without notice. The real fix is a TheRock build
whose RCCL has these APIs, so `libnccl_m2n` uses the RCCL torch already loads; then this goes away.

On load, `nccl.m2n` checks the library (`nccl/_extensions/_rocm_check.py`): every visible GPU has an arch
listed in `gpu_archs.txt` next to `libnccl_m2n.so` (skipped if the file is absent, e.g. for an external
`NCCL_M2N_LIBRARY`), `ncclGetVersion()` is at least 2.30.5 (the `nccl_m2n.h` minimum), the five symbols
above resolve, and `/proc/self/maps` has a single `librccl`. Otherwise it raises with the details.

There is no version pin in `pyproject.toml`: RCCL is not a pip package of its own. TheRock ships it inside
`rocm-sdk-libraries`, versioned by ROCm (`7.15.0a…`), not by RCCL, and no published build has the APIs
yet. Once one does, add `rocm-sdk-libraries>=<that build>` as a dependency (or a `therock` extra); until
then the load-time check is the guard.

## Upstream files and resync

29 files are upstream `python/` as is or with the edits in `rocm-port.patch`; the rest is ours (below). The
patch holds every change to upstream files (6 files): `setup.py`/`pyproject.toml`/`MANIFEST.in` (ROCm build,
no `nccl.ep`, staged `lib/rocm`, cibuildwheel/auditwheel), `_runtime.py` (`is_rocm()`, lazy `cuda.*`), and the
two cybind-generated files `cynccl_m2n.pxd` (opaque stream typedef) and `_internal/nccl_m2n_linux.pyx`
(no `cuda.pathfinder` on ROCm, load-time check). A resync overwrites generated files, so never edit
upstream files without refreshing the patch.

The C library in `projects/rccl/contrib/nccl_m2n` comes from the same upstream commit
(`contrib/nccl_m2n/UPSTREAM_VERSION`); resync both together, the bindings follow `nccl_m2n.h`:

```bash
git -C <nccl-extensions> checkout <new commit>
./resync.sh <nccl-extensions>           # copy the upstream subset over this tree, then patch -p1 rocm-port.patch
# on rejects: fix the files by hand (*.rej), then
./resync.sh --refresh <nccl-extensions> # rewrite rocm-port.patch from the tree
# update commit/date in UPSTREAM_VERSION
```

After editing any upstream file here, run `resync.sh --refresh` against a checkout at the `UPSTREAM_VERSION`
commit and commit the patch with the change.

Ours (not in upstream): `README.md`, `UPSTREAM_VERSION`, `resync.sh`, `rocm-port.patch`,
`nccl/_extensions/_rocm_check.py` (load-time checks), `rocm/sitecustomize.py` (test-only, keeps one `librccl`
per process when torch is loaded; active when `rocm/` is on `PYTHONPATH`) and
`tests/rocm/xdtensor_reshard_check.py` (DTensor reshard against a torch reference, communicator from
`nccl.core.Communicator` or `ProcessGroupNCCL._comm_ptr()`).

Intentionally not part of this port: `nccl.ep`, `nccl/m2n/examples/` (incl. the JAX example referenced by
`nccl/m2n/README.md`), `build_assets/`, the CUDA-only `ci/verify_m2n_package.sh`, and the `nccl.ep` tests.

Build and checks: `projects/rccl/contrib/nccl_m2n/scripts/README.md`.
