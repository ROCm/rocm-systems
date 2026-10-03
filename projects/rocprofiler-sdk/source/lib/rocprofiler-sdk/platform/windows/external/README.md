# Native-Windows platform external dependencies

This directory holds the third-party dependencies required by the native
Windows (`D3DKMT` + `wkmi`) agent-topology enumerator.

## `d3dkmt/`

Vendored Microsoft D3DKMT thunk headers (see `d3dkmt/README.md`). Always
present in the tree.

## `wkmi` (AMD-private, prebuilt only)

The AMD-private Windows Kernel Mode Interface library provides the detailed
(KFD-parity) GPU topology fields. rocprofiler-sdk does **not** build wkmi from
source: doing so requires the libhsakmt/hsa-runtime build context, which this
project's standalone build never sets up. Instead it links a prebuilt
`wkmi.lib` drop.

### Build behavior (see ../CMakeLists.txt)

When `ROCPROFILER_WINDOWS_USE_WKMI=ON` (the default), CMake links a prebuilt
`wkmi.lib`:

- **Default source of the drop:** rocr-runtime's (libhsakmt's) committed copy,
  a sibling of this project in the rocm-systems monorepo:

      projects/rocr-runtime/libhsakmt/src/dxg/wkmi/{wkmi.h,win/dbg/wkmi.lib,win/rel/wkmi.lib}

  `ROCPROFILER_WKMI_ROOT` defaults to that directory; the per-config
  (debug/release) `win/{dbg,rel}/wkmi.lib` is selected automatically.
- **Overrides:** point `ROCPROFILER_WKMI_ROOT` at another drop (e.g. an
  installed layout), or set `WKMI_PREBUILT_LIB` to an explicit `wkmi.lib` (its
  header is still resolved from `ROCPROFILER_WKMI_ROOT`, so set both when
  pointing elsewhere).

If no prebuilt library is found (or `ROCPROFILER_WINDOWS_USE_WKMI=OFF`), the
wkmi bridge compiles to a stub that returns `false`, so the tree still builds and
the enumerator degrades to D3DKMT-only basic fields.

> **CRT/compiler match:** the prebuilt `wkmi.lib` CRT/runtime and compiler MUST
> match the rocprofiler-sdk Windows build (e.g. `/MT`) to avoid CRT-mismatch
> crashes at runtime. The build does not (and cannot here) verify this — it is
> the packager's responsibility to point at the copy that matches this build.
