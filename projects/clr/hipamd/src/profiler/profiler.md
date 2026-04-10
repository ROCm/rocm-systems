# HIP CLR Profiler Project — Key Knowledge

## Project root
`C:/profiler/rocm-systems/` (git repo, branch: develop)

## Build
```bash
cd C:/profiler/build
cmake --build . --config Release -j 6 --target install
```
Installs to `C:/profiler/install/`. Copy DLL to test dir manually:
```bash
cp C:/profiler/install/bin/amdhip64_7.dll C:/profiler/test/amdhip64_7.dll
```

## Test
```bash
# Compile (must specify --offload-arch=gfx1100 for W7900; default is gfx906 which fails at runtime)
cd C:/profiler/test && "C:/profiler/install/bin/hipcc.exe" vectoradd.cpp -o vectoradd.exe -std=c++17 --offload-arch=gfx1100

# Run with profiler
GPU_CLR_PROFILE=1 GPU_ENABLE_PAL=0 ./vectoradd.exe
# Produces: C:/profiler/test/hip_clr_trace.json
```
- Do NOT set AMD_LOG_LEVEL when running tests.
- `GPU_CLR_PROFILE=1` → writes to `hip_clr_trace.json` (value treated as flag if no `/`, `\`, or `.`).
- `GPU_CLR_PROFILE=/path/to/out.json` → writes to that path.
- **ALWAYS pass `--offload-arch=gfx1100`** — missing this causes silent `hipMemcpy` failures and kernel launch error "device kernel image is invalid".

## Key files
- `projects/clr/hipamd/src/hip_intercept.cpp:38` — `hipRegisterTracerCallback`
- `projects/clr/rocclr/platform/prof_protocol.h` — `activity_domain_t`, `activity_record_t`
- `projects/clr/rocclr/platform/activity.hpp` — `amd::activity_prof::report_activity`, `correlation_id` TLS
- `projects/clr/hipamd/src/hip_context.cpp:29` — `hip::init()` — calls `HipClrProfilerInit()`
- `projects/clr/hipamd/src/CMakeLists.txt:94` — `target_sources(amdhip64 PRIVATE ...)`
- `projects/rocr-runtime/libhsakmt/src/dxg/time.cpp` — `hsaKmtGetClockCounters` Windows/WSL impl
- `projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp` — `TranslateTime()`, `SyncClocks()`

## GPU/CPU clock fix (Windows)

### Root cause
DXG `time.cpp` was setting `SystemClockCounter = TimeNanos()` (nanoseconds) and
`SystemClockFrequencyHz = 1e9`. `TranslateTime()` used these as the reference clock,
then `rocvirtual` multiplied by `ticksToTime_ = 100` (ns/QPC-tick) → 100× too large.

### Fix location: `libhsakmt/src/dxg/time.cpp`
```cpp
#if defined(_WIN32)
  // CPUClockCounter from D3DKMTQueryClockCalibration = QPC ticks, atomically
  // correlated with GPUClockCounter. Using it makes TranslateTime() return
  // QPC ticks → rocvirtual * ticksToTime_ = correct nanoseconds.
  Counters->SystemClockCounter = Counters->CPUClockCounter;
  Counters->SystemClockFrequencyHz = rocr::os::SystemClockFrequency();
#else
  // WSL uses __linux__; original behavior preserved.
  Counters->SystemClockCounter = rocr::os::TimeNanos();
  Counters->SystemClockFrequencyHz = 1000000000;
#endif
```
`amd_gpu_agent.cpp` has NO adjustment code — fix is entirely in the DXG layer.

## Implemented: HIP CLR Profiling Layer

### New files created
- `projects/clr/hipamd/src/profiler/hip_clr_profiler.hpp` — structs + internal API
- `projects/clr/hipamd/src/profiler/hip_clr_profiler.cpp` — core implementation
- `projects/clr/hipamd/src/profiler/hip_clr_dispatch_wrappers.cpp` — generated; 511 *Layer wrappers
- `projects/clr/hipamd/include/hip/amd_detail/hip_clr_profiler_ext.h` — public C API
- `projects/clr/hipamd/src/profiler/generate_wrappers.py` — wrapper generator (run manually when HIP API changes)

### Files modified
- `hip_context.cpp` — `HipClrProfilerInit()` at end of `hip::init()`
- `hip_profile.cpp` — `hipProfilerStart/Stop` call `HipClrProfilerEnable/Disable()`
- `src/CMakeLists.txt` — both profiler .cpp files in `target_sources(amdhip64)`

### Wrapper regeneration
Only needed when `hip_api_trace.hpp` or `hip_api_trace.cpp` change (new HIP APIs).
Not part of the cmake build — run manually from `projects/clr/hipamd/src/profiler/`:
```bash
"C:/Users/gandryey/AppData/Local/Programs/Python/Python312/python.exe" generate_wrappers.py
```

### Architecture
**CPU timing** — dispatch table wrappers:
```cpp
auto* _rec = HipClrGetActiveRecord(api_id);  // alloc slot N, set correlation_id TLS=N
auto _r = g_next.hipFoo_fn(...);             // GPU command inherits correlation_id N
if (_rec) _rec->end_ = Clock::now();
```
**GPU timing** — `HipClrActivityCallback` (ACTIVITY_DOMAIN_HIP_OPS):
- `ar->correlation_id == slot` → direct array index, no map/pending table

### JSON output (Chrome Trace Event format)
- `displayTimeUnit: "us"` — all `ts`/`dur` values in microseconds
- CPU `ts` = `duration_cast<microseconds>(start_.time_since_epoch()).count()`
- GPU `ts` = `begin_ns / 1000`
- Thread ids: compact sequential integers (raw hash → mapped to 0,1,2... to stay within JS safe integer range)
- GPU process name = `gcnArchName` from `amd::Device::isa().targetId()` (e.g. `"gfx1100"`)
- CPU process pid=1024, GPU process pid=device_id (from activity_record_t)

### Public API
- `hipClrProfilerEnable()` / `hipClrProfilerDisable()`
- `hipClrProfilerGetRecords(records, count)`
- `hipClrProfilerReset()`
- `hipClrProfilerWriteJson(filepath)`

## Reference tracer
`C:/Work/hipicd/Tracer/hip_tracer_core.cpp` — standalone ICD tracer (reference for JSON format)
