# Build rocprofiler-sdk

Builds rocprofiler-sdk — the primary ROCm profiling framework.
Depends on an installed hsa-runtime64 and rocprofiler-register.

## Prerequisites
- `hsa-runtime64` installed (from hipamd build → `C:/github-emu/install/`)
- `amdhip64.dll` built with `-DROCCLR_ENABLE_GPUOPEN=ON` for RGP capture support

## Configure + Build
```bash
mkdir -p /c/github-emu/rocm-systems/projects/rocprofiler-sdk/build
cd /c/github-emu/rocm-systems/projects/rocprofiler-sdk/build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/c/github-emu/install \
  -DROCM_PATH=/c/opt/rocm \
  -Dhsa-runtime64_DIR=/c/github-emu/install/lib/cmake/hsa-runtime64 \
  -DROCPROFILER_BUILD_TESTS=OFF \
  -DROCPROFILER_BUILD_SAMPLES=OFF 2>&1 | tail -20
cmake --build . --config Release -j 6 2>&1 | tail -30
```

## RGP / SQTT-relevant source paths
- `source/lib/rocprofiler-sdk/` — core library
- `source/lib/rocprofiler-sdk/counters/` — hardware counter collection
- `source/lib/rocprofiler-sdk/tracing/` — dispatch/API tracing
- `cmake/Modules/rocprofiler-sdk-utilities.cmake` — `rocprofiler_sdk_sqtt_triple_buffer_disabled()` and SQTT config

## Dependency chain for full RGP support
```
amdhip64.dll  (CLR + ROCCLR_ENABLE_GPUOPEN=ON)
    └── hsa-runtime64.lib  (rocr-runtime, includes AQLprofile interface)
            └── rocprofiler-sdk  (SQTT dispatch tracing, counter collection)
                        └── RGP tool reads .rgp via RDF chunks written by CLR
```

## Key cmake flags for SQTT/RGP
| Flag | Effect |
|------|--------|
| `-DROCPROFILER_BUILD_TESTS=ON` | Builds dispatch validation tests |
| `-DROCPROFILER_BUILD_SAMPLES=ON` | Builds sample tools |
| `-DCMAKE_BUILD_TYPE=Debug` | Enable assertions in SQTT paths |
