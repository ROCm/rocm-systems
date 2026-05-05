# HRR In-Tree Capture Layer — Agent Instructions

## Status (2026-04-30): IMPLEMENTATION COMPLETE, BUILDS CLEAN
All capture and playback files exist and build. Read `DESIGN.md` for architecture.

**INDEPENDENCE RULE: Zero dependency on `hipamd/src/profiler/`. No profiler headers. All structs in `hip_capture.h`.**

## Repo Context
- **Repo root:** `C:/MIGraphX/rocm-systems` (branch: `develop`)
- **Old HRR reference:** `C:/MIGraphX/rocm-systems_hrr` (branch: `hrr`) — reference only
- **Reference profiler (analysis only):** `C:/profiler/rocm-systems/projects/clr/hipamd/src/profiler/`
- Do NOT edit `hip_hrr.cpp` or out-of-tree proxy; those are superseded.

## File Map

### Capture Layer (`hipamd/src/hrr/`)
```
hip_capture.h           — public API, EventHeader (VERSION=3, 32 bytes), BlobHash struct
hip_capture.cpp         — 12 manual shims (NON-static, extern'd by generated), init/shutdown
                          D2H blob capture for hipMemcpy and hipMemcpyAsync (with stream sync)
hip_capture_writer.h    — write_event_raw(), write_blob(), hrr_cap_next_seq() declarations
hip_capture_writer.cpp  — streaming events.bin writer; hrr_cap_next_seq() is extern "C"
hip_capture_handles.h   — stream/event/module ID maps declarations
hip_capture_handles.cpp — mutex-guarded unordered_map implementations
hip_capture_generated.cpp — AUTO-GENERATED (run gen_hrr_api_args.py to regenerate)
                            517 shims + hip_capture_build_table()
CMakeLists.txt          — adds hrr/ sources to amdhip64 target
```

### Playback System (`utils/playback/`)
```
hrr_reader.h/.cpp           — archive loader, v3 format
hip_playback.h              — PlaybackContext, PlaybackFn typedef, dispatch table decl
hip_playback.cpp            — 26 manual playback implementations (including all graph APIs)
hip_playback_generated.cpp  — AUTO-GENERATED: 504 shims + hrr_playback_dispatch[]
hrr_playback.cpp            — hrr-playback tool (replay + D2H validation + --info)
gen_hrr_api_args.py         — generator for all 3 auto-generated files
```

### Generated Header (`utils/include/`)
```
hrr_api_args.h  — AUTO-GENERATED: 529 hrr_args_* structs + hrr_api_id_t enum (529 entries)
```

## Key Files to Know

### Hook Points — Two Dispatch Tables
- `../include/hip/amd_detail/hip_api_trace.hpp` — `HipDispatchTable` + `HipCompilerDispatchTable`
  `GetHipDispatchTable()` / `GetHipCompilerDispatchTable()` return **const ptr** — use `const_cast`
- `../hip_table_interface.cpp` — routes every `extern "C" hipFoo()` through dispatch table

### Runtime Internals (kernel arg introspection)
- `../../rocclr/device/devkernel.hpp` — `KernelParameterDescriptor`: `type_`, `offset_`, `size_`, `info_.hidden_`
- `../../opencl/amdocl/cl_kernel.h` — `T_POINTER`, `T_INT`, etc.
- `amd::Kernel* k = hip::DeviceFunc::asFunction(f)->kernel()`
- `k->signature()` → `amd::KernelSignature`: `numParameters()`, `at(i)`
- `kernel->name()` returns `std::string` — call `.c_str()` for C-string functions
- `sig.numParametersAll()` includes hidden args — check `desc.info_.hidden_` and skip

## Critical Implementation Notes

### Payload layout (CORRECT as of 2026-05-05)
`payload` in `PlaybackFn` points to the FULL `hrr_args_*` struct (header + fields).
`hrr_args_*` struct layout: `hdr(32) + ret(4) + fields...`
**Cast `payload` directly:** `const auto* a = reinterpret_cast<const hrr_args_foo*>(payload);`
Then access fields: `a->deviceId`, `a->ptr`, etc. — no offset arithmetic needed.
Exception: `replay_kernel_launch` is variable-length; skip the header manually:
`const uint8_t* p = pl + sizeof(hrr_event_header); const uint8_t* end = pl + hdr->payload_length;`
`rd32`/`rd64` helpers are still defined but unused — do not add new uses of them.

### Dispatch Table Globals (non-static, extern'd by generated file)
```cpp
HipDispatchTable         g_real_table{};
HipDispatchTable         g_cap_table{};
std::atomic<bool>        g_installed{false};
HipCompilerDispatchTable g_real_compiler_table{};
std::atomic<bool>        g_compiler_installed{false};
```

### D2H Blob Capture (hip_capture.cpp)
After a DeviceToHost memcpy, the host buffer holds GPU result — capture it as a blob:
- `hipMemcpy` (sync): call real function, then `write_blob(dst, sizeBytes)` immediately
- `hipMemcpyAsync` (async): after real call, `g_real_table.hipStreamSynchronize_fn(stream)`, then `write_blob(dst, sizeBytes)`

### `<<<>>>` Launch Path
- `__hipPushCallConfiguration` → save grid/block/shared/stream in thread_local TLS
- `hipLaunchByPtr` → read from TLS, record kernel launch (no arg introspection on this path)
- `__hipPushCallConfiguration` is in `HipCompilerDispatchTable`; `hipLaunchByPtr` is in `HipDispatchTable`

### MANUAL_CAPTURE_APIS (12)
kernel launches (4), H2D memcpy (4), module load (3), `__hipRegisterFatBinary`

### MANUAL_PLAYBACK_APIS (26)
kernel launches (4), H2D memcpy (4), module load (3), hipModuleGetFunction,
hipMalloc/MallocAsync/MallocFromPoolAsync/MallocManaged/Free/FreeAsync,
hipStreamCreate×3/Destroy, hipEventCreate×2/Destroy,
hipStreamBeginCapture, hipStreamEndCapture, hipGraphInstantiate, hipGraphLaunch

### HIP Graph Replay Chain
`hipStreamBeginCapture` → kernels → `hipStreamEndCapture` (returns `hipGraph_t*`) →
`hipGraphInstantiate` (use `hipGraphInstantiateWithFlags(exec, graph, 0)`) →
`hipGraphLaunch(exec, stream)`
All four are MANUAL. Validated: 276 kernels, 32 graph launches, all PASS.

### Fat Binary / Module Resolution
- `__hipRegisterFatBinary` fires at static-init before shims install — retroactive sweep via `ForEachFatBinaryBlob` at `hip_capture_init()`
- At playback: loaded via `hipModuleLoadData`, stored in `ctx.co_modules`; kernel names found by scanning `module_map` (co_hash always 0 in kernel launch events)

### `hrr_cap_next_seq()` Linkage
Must be `extern "C"` in both definition (hip_capture_writer.cpp) and declaration sites
to avoid MSVC C/C++ mangling mismatch.

### Include Paths (from hrr/)
- Use `"../hip_global.hpp"` (parent-relative) for hipamd/src/ headers
- `utils/include/` is on the include path via CMakeLists

## Generator Script
```
hipamd/src/hrr/gen_hrr_api_args.py
```
Run from `hipamd/src/hrr/`: `C:/Users/gandryey/AppData/Local/Programs/Python/Python312/python.exe gen_hrr_api_args.py`
Generates: `hrr_api_args.h`, `hip_capture_generated.cpp`, `hip_playback_generated.cpp`
Note: `.pyc` / `__pycache__/` are auto-generated cache files — add to `.gitignore`, do not commit.

### Generator Bug Fixes (already applied — do not revert)
- No `#include "utils/flags.hpp"` in generated file (causes MSVC C4430 cascade)
- Strip `struct`/`enum`/`class`/`union` keywords before `_NON_CASTABLE_TYPES` lookup
- `hipChannelFormatDesc` in `_NON_CASTABLE_TYPES` (struct param, can't cast to u64)
- `hipDevice_t*`: `static_cast<uint64_t>(static_cast<int>(*device))`
- `hipCreateChannelDesc`: returns `hipChannelFormatDesc` (not hipError_t) — emit simple forward
- `hrr_cap_next_seq` is `extern "C"` in writer and all declaration sites

## Wiring (complete)
- `hipamd/src/CMakeLists.txt` — `add_subdirectory(hrr)` after hiprtc
- `hipamd/src/hip_context.cpp` — `hip_capture_init()` after `PlatformState::Instance().Init()`
- `rocclr/utils/flags.hpp` — `HIP_HRR_CAPTURE_OUTPUT` cstring flag at end of `RUNTIME_FLAGS`

## Build System

### Build Tree
- **Build dir:** `C:/MIGraphX/amdhip` (Visual Studio 18 2026, Release)
- **Source dir:** `C:/MIGraphX/rocm-systems/projects/clr`
- **Install dir:** `C:/MIGraphX/install`
- **Playback test build:** `C:/MIGraphX/test/build2`

### Full Reconfigure (from `C:/MIGraphX/amdhip`)
```
cmake ../rocm-systems/projects/clr -DCMAKE_BUILD_TYPE=Release -DCLR_BUILD_HIP=ON \
  -DHIP_COMMON_DIR=c:/github-emu/rocm-systems/projects/hip \
  -DHIPCC_BIN_DIR=c:/opt/rocm/bin \
  -DCMAKE_INSTALL_PREFIX=../install -D__HIP_ENABLE_PCH=OFF \
  -DROCCLR_ENABLE_HSA=ON -DROCCLR_ENABLE_PAL=ON -D__HIP_ENABLE_RTC=ON \
  -DUSE_PROF_API=OFF -DROCR_DLL_LOAD=OFF \
  -DAMD_COMPUTE_WIN=../../../shared/amdgpu-windows-interop/ -DLIB_SRC_BUILD=ON \
  -DClang_BIN=c:/opt/rocm/lib/llvm/bin
```

### Incremental Build (from `C:/MIGraphX/amdhip`)
```bash
# Capture layer (amdhip64 DLL):
cmake --build . --config Release -j 6 --target amdhip64 > /tmp/hip_build.txt 2>&1

# Full build + install:
cmake --build . --config Release -j 6 --target install > /tmp/hip_build.txt 2>&1
```
**Always redirect to file** — piping to grep kills the build. Always use `--config Release`.

### Build Key Notes
- `GetHipDispatchTable()` returns `const` ptr — use `const_cast` to write
- MUST match `--config Release` to `CMAKE_BUILD_TYPE=Release` — `macros.hpp` enforces with `#error`
- clang.exe is at `c:/opt/rocm/lib/llvm/bin/clang.exe` (NOT `bin/`)
- ROCm layout: `c:/opt/rocm/bin/` — hipcc.exe, amdhip64_7.dll; `c:/opt/rocm/lib/llvm/bin/` — clang.exe

## Capture Usage
```bash
HIP_HRR_CAPTURE_OUTPUT=./capture_out ./my_hip_app
```
Produces `./capture_out/` directory with `events.bin`, `blobs/`, `code_objects/`.

## Playback Usage
```bash
hrr-playback capture_out [--verbose] [--skip-device-sync]
hrr-playback capture_out --info [--events]
```
Exits 0 = all D2H checks pass (or no blobs), 1 = any failure.

## Environment
- Windows 11, ROCm SDK 6.4 at `C:/opt/rocm` and `C:/Program Files/AMD/ROCm/6.4/`
- Python 3.12: `C:/Users/gandryey/AppData/Local/Programs/Python/Python312/python.exe`
- Test: `C:/MIGraphX/test/` — hip_raw_trace + hrr-validate; build at `C:/MIGraphX/test/build2`
