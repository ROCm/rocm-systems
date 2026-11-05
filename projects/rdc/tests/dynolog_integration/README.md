# dynolog &harr; RDC integration CI helpers

**dynolog** is a system-level telemetry daemon that runs RDC in-process. dynolog does
not spawn `rdcd`; it embeds RDC in-process through a thin C++ shim,
[`RdcWrapper`](https://github.com/facebookincubator/dynolog/tree/main/dynolog/src/gpumon/amd),
which `#include`s `rdc/rdc.h`, links `librdc`, and creates one GPU group + field
group to watch a fixed set of `RDC_FI_*` fields.

These helpers back the [`rdc-dynolog-ci.yml`](../../../../.github/workflows/rdc-dynolog-ci.yml)
workflow. They build that shim (and its `RdcWrapperExample`) against an RDC that
was just built from `projects/rdc`, so CI catches RDC changes that would break
the downstream dynolog integration **before** they ship.

```
RdcWrapperExample (C++)
   -> dynolog RdcWrapper        (dynolog's integration shim)
   -> librdc / librdc_bootstrap (embedded RDC, built from projects/rdc)
   -> libamd_smi.so             -> /dev/kfd
```

The **primary signal is the shim compile + link** against the freshly-installed
`rdc/rdc.h` and `librdc.so`: an RDC API/ABI/build change that the dynolog
integration cannot tolerate fails the build. Running the example on a GPU is an
additional runtime check that soft-passes when no GPU metrics are available.

## Subcommands

Run any phase locally (from `projects/rdc/tests`) with:

```bash
python3 -m dynolog_integration <subcommand> [options]
```

| Subcommand | Purpose |
| --- | --- |
| `clone` | Shallow-fetch dynolog at a pinned ref (branch, tag, or 40-char SHA). |
| `build-wrapper` | Configure + build `dynolog/src/gpumon/amd` (shim + example) against an installed RDC. Auto-detects `ROCM_VERSION` from `<rocm-dir>/.info/version`. |
| `run-example` | Run `RdcWrapperExample` for a bounded window and verify it initialized RDC, discovered GPU entities, and collected metrics. |

Example end-to-end run against an RDC already installed under `/opt/rocm`:

```bash
cd projects/rdc/tests
python3 -m dynolog_integration clone \
    --ref b301ce05b3b25e2f48e4525bfc52a0ba6fe77446 --dest /tmp/dynolog
python3 -m dynolog_integration build-wrapper \
    --dynolog-dir /tmp/dynolog --rocm-dir /opt/rocm
python3 -m dynolog_integration run-example \
    --binary /tmp/dynolog/dynolog/src/gpumon/amd/build/examples/RdcWrapperExample \
    --rocm-dir /opt/rocm \
    --ld-library-path /opt/rocm/lib:/opt/rocm/lib/rdc
```

## No gRPC is needed

gRPC is an RDC *standalone*-mode dependency: only `rdcd`, `rdci` and
`librdc_client.so` link it, and `rdc_libs/CMakeLists.txt` installs
`librdc_client.so` into the exported target set only under `BUILD_STANDALONE`.
dynolog uses embedded mode and links the `rdc` target, whose exported interface
is `rdc_bootstrap;pthread;amd_smi;cap`.

So the workflow builds RDC with `-DBUILD_STANDALONE=OFF`: `librdc.so`,
`librdc_bootstrap.so` and `librdc_rocp.so` are unchanged (nothing under
`rdc_libs/` other than the `rdc_client` gate reads that option), and the
generated `rdcTargets.cmake` contains no `gRPC::grpc++` reference at all. RDC's
standalone/gRPC surface is `rdc-ci.yml`'s job, not this workflow's.

Avoiding gRPC here also avoids pinning a second gRPC version: TheRock builds
RDC against `THEROCK_GRPC_VERSION` (`third-party/grpc/`, currently 1.78.1,
statically linked), which is also RDC's own `GRPC_DESIRED_VERSION` default. A
hand-rolled gRPC in this workflow would drift from what actually ships.

## Prerequisites for the build

The shim's standalone CMake requires an RDC SDK + `amd_smi` (normally under
`/opt/rocm`), plus `glog` and `gflags`. On a stock Ubuntu host:

```bash
apt-get install -y ninja-build cmake libgoogle-glog-dev libgflags-dev
```

`libcap` is additionally needed to *build RDC itself* (`find_library(LIB_CAP)`),
and is pulled in transitively when linking the shim against `librdc.so`.

> **CI note:** the ROCm dev container the workflow runs in
> (`rocm/dev-ubuntu-22.04`) cannot install `libgoogle-glog-dev` — its
> `libunwind-dev` dependency is unavailable there — so the workflow builds
> **glog v0.6.0 from source** (`-DWITH_UNWIND=OFF`) into `/usr/local` instead.
> See the workflow's *Build and install glog* step.

## Tests

Stdlib-only unit tests cover the ROCm-version math and the example-output
verification (no GPU required):

```bash
cd projects/rdc/tests
python3 -m pytest dynolog_integration          # or:
python3 -m unittest dynolog_integration.test_build dynolog_integration.test_run_example
```

## CI configuration

The workflow runs on a **self-hosted AMD GPU runner** and needs these repo/org
settings (the same GPU runner convention the DME integration CI uses):

| Setting | Required | Purpose |
| --- | --- | --- |
| `vars.RUNNER_TYPE` | **Yes** | Second `runs-on` label selecting the self-hosted GPU runner. If unset, the job never schedules and eventually times out. |
| `vars.GPU_TARGETS` | No (default `gfx942`) | `GPU_TARGETS` for the RDC build; set to the runner's arch (e.g. `gfx90a`, `gfx950`) so profiler kernels load. |
| `vars.RDC_DOCKER_IMAGE` | No (default `rocm/dev-ubuntu-22.04:7.2`) | ROCm build container image. |

The runtime `RdcWrapperExample` step is intentionally lenient: it **soft-passes**
(with a `::warning::`) when a started example collects no metrics or throws on an
unsupported field, since those depend on the runner's GPU arch rather than the
RDC build/link surface. The build step is the primary gate. Pass `--strict` to
`run-example` to gate on metric collection on a known-good runner.

It still hard-fails on the cases no GPU arch explains: the binary not starting
at all (runtime link/load break), a non-capability fatal from `RdcWrapper`, the
process exiting on its own (the poll loop is meant to run until terminated), and
setup never completing within the window (a hang).

## Keeping the dynolog pin current

Automatic runs build dynolog at the immutable SHA pinned in `DYNOLOG_REF`
(`.github/workflows/rdc-dynolog-ci.yml`). Because the shim is built against
`develop`'s RDC, a **deliberate** RDC API change that the pinned dynolog has not
yet adopted will (correctly) fail the build until the integration is updated. To
resolve such drift:

* re-run against a newer dynolog via **Actions -> Run workflow** and set the
  `dynolog_ref` input, then bump `DYNOLOG_REF` to that ref; or
* coordinate the corresponding change upstream in
  `facebookincubator/dynolog` first.
