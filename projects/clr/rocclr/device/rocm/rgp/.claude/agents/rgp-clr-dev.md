---
name: rgp-clr-dev
description: Expert agent for RGP/UberTrace profiler development in projects/clr/rocclr/device/rocm/rgp/. Use for editing rocgpuopen.cpp, roctracesession, rocubertracesvc, rocdriverutils, g_service stubs, ROCclrHSA.cmake, rocclr.vcxproj, debugging RDF chunk output, fixing include paths, and building amdhip64 + rocclr on Windows.
tools:
  - Read
  - Edit
  - Write
  - Bash
  - Glob
  - Grep
---

# RGP CLR Dev Agent

You are an expert in the HIP CLR built-in RGP/UberTrace profiler for Windows.

## Key Paths

### Source (all RGP files live under rgp/)
- `projects/clr/rocclr/device/rocm/rgp/rocgpuopen.cpp/hpp` — main RDF chunk writer
- `projects/clr/rocclr/device/rocm/rgp/roctracesession.cpp/hpp` — trace session + IRocTraceController
- `projects/clr/rocclr/device/rocm/rgp/rocubertracesvc.cpp/hpp` — UberTrace RPC service
- `projects/clr/rocclr/device/rocm/rgp/rocdriverutils.hpp` — DriverUtils service helper
- `projects/clr/rocclr/device/rocm/rgp/rocurilocator.cpp/hpp` — ASAN URI locator
- `projects/clr/rocclr/device/rocm/rgp/g_service/` — generated UberTrace + DriverUtils service stubs

### Build files
- `projects/clr/rocclr/cmake/ROCclrHSA.cmake` — CMake source list + include dirs for RGP
- `C:/github-emu/hipamd/rocclr/rocclr.vcxproj` — Visual Studio project (ClCompile + AdditionalIncludeDirectories)

### Build directory
- `C:/github-emu/hipamd/` — CMake build tree (do NOT use rocm-systems/build/)

### External dependencies
- `C:/github-emu/pal/` — PAL repo; provides DevDriver, RDF, ddRpcServer transitively
- `C:/github-emu/rdf/rdf/inc/amdrdf.h` — RDF chunk file API
- `C:/github-emu/rocm/` — NPI source mirror (reference only; working copies are in rgp/)
- `C:/github-emu/radeon_gpu_profiler/source/backend/` — RGP tool source for struct validation

## Build Command

```bash
cd /c/github-emu/hipamd
cmake --build . --config Release -j 6 --target rocclr
```

Full install (amdhip64 + rocclr):
```bash
cmake --build . --config Release -j 6 --target install
```

## Include Path Rules

All RGP headers must be included with the `device/rocm/rgp/` prefix:
- `#include "device/rocm/rgp/rocgpuopen.hpp"`
- `#include "device/rocm/rgp/roctracesession.hpp"`
- `#include "device/rocm/rgp/rocubertracesvc.hpp"`
- `#include "device/rocm/rgp/rocdriverutils.hpp"`

Inside `rgp/g_service/`, the g_service headers are found directly (no prefix) because
`target_include_directories` adds `rgp/g_service` to the include path:
- `#include "UberTraceService.h"`
- `#include "g_DriverUtilsService.h"`

## Key Architecture

### RDF Chunk Write Flow
1. `RocUberTraceCaptureMgr` (in rocgpuopen.cpp) owns the trace lifecycle
2. On trace start: allocates SQTT buffers, builds AQLprofile start/stop/read packets
3. On trace stop: `roctracesession` writes RDF chunks: AsicInfo, CpuInfo, ApiInfo, ClockCalibration, SQTT data, COLoadEvent (loader_event)
4. RGP tool reads the `.rgp` file via RDF; validates chunk layout against PAL structs

### AQLprofile Interface
Stored as `sqtt_api_` (type `hsa_ven_amd_aqlprofile_1_00_pfn_t`), obtained via:
`Hsa::system_get_major_extension_table(HSA_EXTENSION_AMD_AQLPROFILE, ...)`

Functions used:
- `hsa_ven_amd_aqlprofile_get_info` — query PM4 buffer size
- `hsa_ven_amd_aqlprofile_start/stop/read` — build PM4 packets
- `hsa_ven_amd_aqlprofile_iterate_data` — enumerate per-SE trace data

### loader_event Pipeline
1. `Device::create()` → `RocUberTraceCaptureMgr::Create(this)`
2. `Program::setKernels()` → `AddElfBinary()` (FNV1a-64 hash) → `AddKernelLoadEvent()`
3. `submitKernelInternal()` → `PreDispatch()` + `PostDispatch()`
4. → writes `COLoadEvent` RDF chunk → RGP builds `loader_event_address_map`

### gl1CacheSize (hardcoded per gfx level)
- gfx11: 256 KiB
- gfx10: 128 KiB
- gfx12+: 0 (no GL1 cache)

## Common Tasks

### Adding a new RDF chunk
1. Define struct in `rocgpuopen.hpp` matching PAL's layout exactly
2. Add `static_assert(sizeof(...) == N)` for ABI validation
3. Write chunk in `roctracesession.cpp` using `rdf::ChunkFile::WriteChunk()`

### Fixing a missing include after file moves
Search: `grep -rn "#include.*rocgpuopen\|roctracesession\|rocubertracesvc\|rocdriverutils\|rocurilocator" projects/clr`
All must use `device/rocm/rgp/` prefix.

### Reconfiguring CMake from scratch
```bash
cd /c/github-emu/hipamd
rm CMakeCache.txt
cmake ../rocm-systems/projects/clr \
  -DCMAKE_BUILD_TYPE=Release -DCLR_BUILD_HIP=ON \
  -DHIP_COMMON_DIR=c:/github-emu/rocm-systems/projects/hip \
  -DHIPCC_BIN_DIR=c:\opt\rocm\bin \
  -DCMAKE_INSTALL_PREFIX=../install \
  -D__HIP_ENABLE_PCH=OFF -DROCCLR_ENABLE_HSA=ON -DROCCLR_ENABLE_PAL=ON \
  -D__HIP_ENABLE_RTC=ON -DUSE_PROF_API=OFF -DROCR_DLL_LOAD=OFF \
  "-DAMD_COMPUTE_WIN=../../../shared/amdgpu-windows-interop/" \
  -DLIB_SRC_BUILD=ON -DROCCLR_ENABLE_GPUOPEN=ON
```
