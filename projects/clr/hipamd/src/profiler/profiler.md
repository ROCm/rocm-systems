# HIP CLR Profiler

Built-in profiling layer for the HIP runtime. Captures CPU API timings and GPU
activity (dispatch, copy, barrier) and writes a Chrome Trace Event JSON file.

---

## Build

```bash
cd C:/profiler/build
cmake --build . --config Release -j 6 --target install
cp C:/profiler/install/bin/amdhip64_7.dll C:/profiler/test/amdhip64_7.dll
```

## Test

```bash
# Always pass --offload-arch=gfx1100 for W7900; missing it causes silent kernel failures.
cd C:/profiler/test
"C:/profiler/install/bin/hipcc.exe" vectoradd.cpp -o vectoradd.exe -std=c++17 --offload-arch=gfx1100

# Debug build
"C:/profiler/install/bin/hipcc.exe" vectoradd.cpp -o vectoradd_dbg.exe -g -O0 --offload-arch=gfx1100 \
  -I/c/profiler/rocm-systems/projects/clr/hipamd/include \
  -L/c/profiler/build/hipamd/lib/Release -lamdhip64

# Run
GPU_CLR_PROFILE_OUTPUT=hip_clr_trace.json ./vectoradd.exe
```

---

## Activation

### Environment variable

```bash
GPU_CLR_PROFILE_OUTPUT=<path>   # enable at startup; writes JSON to <path> on exit
```

Defined in `projects/clr/rocclr/utils/flags.hpp` via the CLR flag system.

### Programmatic API (`hip/amd_detail/hip_profiler_ext.h`) — BETA v0.1.0

> **Beta:** structures, signatures, and enum values may change without notice.
> Do not use in production code.

```c
hipError_t hipProfilerEnableExt(void);
hipError_t hipProfilerDisableExt(void);
hipError_t hipProfilerResetExt(void);
hipError_t hipProfilerGetRecordsExt(const HipApiRecordExt* const** chunks,
                                     size_t* chunk_count,
                                     size_t* chunk_size,
                                     size_t* total_count);
```

`hipProfilerStart` / `hipProfilerStop` (standard HIP API) also map to enable/disable.

---

## Record structures (`hip/amd_detail/hip_profiler_ext.h`)

```c
/* GPU activity — 128 bytes */
typedef struct HipGpuActivityExt {
  uint64_t op        : 3;   /* HipGpuOpExt */
  uint64_t copy_kind : 4;   /* HipCopyKindExt; valid when op==HIP_OP_COPY_EXT */
  uint64_t device_id : 16;
  uint64_t queue_id  : 16;
  uint64_t begin_ns;
  uint64_t end_ns;
  union { uint64_t bytes; const char* kernel_name; };
  uint32_t gpu_op_count;    /* >1 for graph launches with multiple nodes */
  const struct HipGpuActivityExt* gpu_ops;
} HipGpuActivityExt;

/* Per-API record — 256 bytes */
typedef struct {
  const char*       api_name;
  uint64_t          has_gpu_activity : 1;
  uint64_t          thread_id;
  uint64_t          start_ns;
  uint64_t          end_ns;
  hipStream_t       stream;
  HipGpuActivityExt gpu;   /* valid when has_gpu_activity != 0 */
} HipApiRecordExt;
```

### Op types (`HipGpuOpExt`)

| Value | Constant | Meaning |
|-------|----------|---------|
| 0 | `HIP_OP_DISPATCH_EXT` | Kernel dispatch |
| 1 | `HIP_OP_COPY_EXT` | Memory copy |
| 2 | `HIP_OP_BARRIER_EXT` | Barrier / fence |

### Copy directions (`HipCopyKindExt`, valid when `op == HIP_OP_COPY_EXT`)

| Constant | Direction |
|----------|-----------|
| `HIP_COPY_KIND_H2D_EXT` | Host → device (SDMA) |
| `HIP_COPY_KIND_D2H_EXT` | Device → host (SDMA) |
| `HIP_COPY_KIND_D2D_EXT` | Device → device (blit) |
| `HIP_COPY_KIND_FILL_EXT` | Device fill |
| _(rect / image variants)_ | See header for full list |

`hipCopyKindIsSDMAExt(kind)` — returns non-zero for H2D / D2H kinds.

### Iterating records

```c
const HipApiRecordExt* const* chunks;
size_t chunk_count, chunk_size, total;
hipProfilerGetRecordsExt(&chunks, &chunk_count, &chunk_size, &total);
for (size_t c = 0; c < chunk_count; ++c) {
  size_t n = (total - c * chunk_size < chunk_size) ? total - c * chunk_size : chunk_size;
  for (size_t i = 0; i < n; ++i) {
    const HipApiRecordExt* r = &chunks[c][i];
    /* r->api_name, r->start_ns, r->end_ns, r->gpu, ... */
  }
}
```

---

## Key files

| File | Role |
|------|------|
| `src/profiler/hip_clr_profiler.hpp` | Internal structs and API declarations |
| `src/profiler/hip_clr_profiler.cpp` | Core implementation |
| `src/profiler/hip_clr_dispatch_wrappers.cpp` | Generated 511 `*Layer` wrappers |
| `src/profiler/generate_wrappers.py` | Wrapper generator — run manually when HIP API changes |
| `include/hip/amd_detail/hip_profiler_ext.h` | Public C API header |
| `rocclr/utils/flags.hpp` | `GPU_CLR_PROFILE_OUTPUT` flag definition |
| `src/hip_context.cpp` | Calls `HipProfilerInitExt()` at end of `hip::init()` |
| `src/hip_profile.cpp` | `hipProfilerStart/Stop` → `HipProfilerEnableExt/DisableExt` |
| `src/CMakeLists.txt` | Both profiler `.cpp` files added to `amdhip64` |

### Wrapper regeneration

Only needed when `hip_api_trace.hpp` or `hip_api_trace.cpp` gain new entries:

```bash
"C:/Users/gandryey/AppData/Local/Programs/Python/Python312/python.exe" generate_wrappers.py
```

