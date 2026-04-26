# LTTng curated parameter-capture tracepoints — Design Spec

**Date:** 2026-04-26
**Author:** bewelton (with brainstorming session)
**Branch:** `users/bewelton/lttng`
**Folds into:** PR [#5475](https://github.com/ROCm/rocm-systems/pull/5475)
**Status:** Draft, awaiting user review

---

## 1. Goal

Add per-API typed parameter-capture LTTng tracepoints for ~82 curated HIP and HSA APIs (final count is whatever ends up in `curated_apis.yaml`). These augment the existing generic `hip_api_enter` / `hip_api_exit_*` and `hsa_api_enter` / `hsa_api_exit_*` events without replacing them. Consumers that need parameter visibility (kernel launch dims, memcpy sizes, stream/event handles, etc.) can subscribe to per-API typed events; consumers that only need call boundaries continue to use the generic events at the same cost as today.

## 2. Non-goals

- **Auto-instrument all HIP / HSA APIs.** Only ~82 high-value APIs are curated. The other ~450 wrappers continue to emit only generic `api_enter` / `api_exit_*`.
- **Struct-field exploration.** Parameters typed `T*` for complex structs (`hipMemcpy3DParms*`, `hipKernelNodeParams*`) capture the pointer address only. No struct walking in v1.
- **`void**` deref.** `hipLaunchKernel`'s kernel argument array captures the outer pointer only.
- **RCCL coverage.** RCCL is a separate library at `projects/rccl/`; its API surface is a future follow-up.
- **Symbolic enum names.** Enums captured as their underlying integer; consumers map to symbolic names from `hip_runtime_api.h` themselves.
- **Replace generic enter/exit events.** Generic events keep their current schema and emit-cost.

## 3. Architecture

### 3.1 Per-call event flow (curated API)

```
1. wrapper enters
   -> tracepoint: rocm_hip:hip_api_enter        (corr_id, parent_corr_id)  [generic, unchanged]
2. function body runs (IN-params held in C locals across the body)
3. just before return:
   -> tracepoint: rocm_hip:<api>_args           (corr_id, IN-params..., OUT-params...) [NEW typed]
   -> tracepoint: rocm_hip:hip_api_exit_status  (corr_id, status)           [generic, unchanged]
```

Three events per curated call (vs. 2 today). Consumers that don't subscribe to `<api>_args` pay the standard `lttng_ust_tracepoint_enabled()` short-circuit cost (one atomic load + unlikely branch).

### 3.2 Source-of-truth and toolchain

```
projects/clr/hipamd/scripts/curated_apis.yaml                        DSL (HIP)
projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml  DSL (HSA, mirror)
        |
        v
lttng_curated_codegen.py    (YAML only, no libclang)
        |
        v
projects/clr/hipamd/src/lttng/rocm_hip_curated_tp.h        (generated, checked in)
projects/clr/hipamd/src/lttng/rocm_trace_emit_curated.h    (generated, checked in)
projects/rocr-runtime/.../lttng/rocm_hsa_curated_tp.h       (generated, checked in)
projects/rocr-runtime/.../lttng/rocm_trace_emit_curated.h   (generated, checked in)

lttng_curated_verify.py     (libclang vs YAML — separate CI gate; per-ROCm-version drift detection)
```

Generated files are **checked in**. The build does not invoke the codegen by default — it consumes the checked-in headers directly. Codegen runs only when explicitly requested (developer regeneration, or CI drift gate). See §9 for the exact build/CI dependency contract.

### 3.3 Exception path

If the wrapper body throws, `<api>_args` is **not** emitted. The existing `CATCH` macro continues to fire `hip_api_exit_status` with the error code. Rationale: OUT params haven't been written; the dominant exception is allocation failure where IN-params alone don't add much beyond `api_name` from `hip_api_enter`.

## 4. DSL — `curated_apis.yaml`

Top-level: list of API entries. Per entry:

| Field | Required | Description |
|---|---|---|
| `api` | yes | C function name (must match HIP/HSA header exactly). |
| `category` | yes | `streams` \| `events` \| `kernel_launch` \| `memory` \| `graphs` \| `module` \| `hsa_queues` \| `hsa_signals` \| `hsa_memory`. |
| `args` | yes | List of arg entries (may be empty). |
| `since` | no | Informational, e.g. `"6.0"`. |
| `notes` | no | Informational. |

Per `args` entry:

| Field | Required | Description |
|---|---|---|
| `name` | yes | Parameter name. **Must exactly match the corresponding parameter name in the HIP/HSA header declaration** — the migrator binds YAML entries to source parameters by name (positional binding is intentionally not used because parameter reorderings would silently produce wrong-typed captures). The verify script enforces this as a hard error (see §8.3). |
| `type` | yes | DSL type from the vocabulary below. |
| `dir` | yes | `IN` \| `OUT` \| `INOUT`. |

### 4.1 Type vocabulary

| DSL type | LTTng field | Maps from C type |
|---|---|---|
| `handle` | `ctf_integer_hex(uint64_t)` | All opaque handles: `hipStream_t`, `hipEvent_t`, `hipFunction_t`, `hipGraph_t`, `hipGraphExec_t`, `hipGraphNode_t`, `hipModule_t`, `hsa_signal_t`, `hsa_queue_t*`, `hsa_agent_t`. |
| `ptr` | `ctf_integer_hex(uint64_t)` | `void*`, `T*`, `const T*` — captured as address only. |
| `device_ptr` | `ctf_integer_hex(uint64_t)` | `hipDeviceptr_t`. |
| `size` | `ctf_integer(uint64_t)` | `size_t`. |
| `int32` / `uint32` | `ctf_integer(int32_t)` / `ctf_integer(uint32_t)` | `int`, `unsigned int`, `int32_t`, `uint32_t`. |
| `int64` / `uint64` | `ctf_integer(int64_t)` / `ctf_integer(uint64_t)` | `int64_t`, `uint64_t`. |
| `float` | `ctf_float(float)` | `float`. |
| `enum` | `ctf_integer(int32_t)` | `hipMemcpyKind`, `hipMemoryAdvise`, `hsa_queue_type32_t`, etc. |
| `dim3` | 3 × `ctf_integer(uint32_t)` | `dim3`. Auto-expands to `<name>_x`, `<name>_y`, `<name>_z`. |
| `dim3_packed` | 1 × `ctf_integer_hex(uint64_t)` | `dim3`. Three 16-bit lanes packed into one uint64 (`x | y<<16 | z<<32`). Used to stay under field-budget for high-arity APIs (see §4.4). |
| `cstring` | `ctf_string` | `const char*`. NULL-safe (NULL → empty string). Helper guards on `tracepoint_enabled()`. |

### 4.4 Field-budget rule (LTTng-UST 10-field / 20-arg limit)

Every generated `<api>_args` event MUST stay within LTTng-UST's documented limit of **10 LTTng fields** (which corresponds to 20 `LTTNG_UST_TP_ARGS` slots, since each field is 2 args). The mandatory `corr_id` field counts as 1 of the 10. This means each curated API may declare at most **9 fields** of payload after `corr_id`.

Note: `dim3` expands to **3** fields, not 1. Codegen MUST count expanded fields, not DSL entries, when validating against the limit.

When a curated API exceeds 9 payload fields under the natural mapping, the YAML author MUST apply one of the following mitigations, in this order of preference:

1. **Use `dim3_packed`** in place of `dim3` (saves 2 fields per dim3).
2. **Drop low-value fields** (e.g. captured pointer values for opaque blob args like `kernelParams`, `extra`) and document the omission in the API's YAML `notes:`.
3. **Split into two events** — `<api>_args` and `<api>_args_ext` — both keyed on the same `corr_id`. Codegen emits both from a single helper. Reserved as a last resort; consumer-side join logic costs.

Codegen enforces this at generation time: it computes the expanded field count and aborts with a clear error if any API exceeds 9 payload fields without a `pack:` or `split:` directive in its YAML entry. Verify script (`lttng_curated_verify.py`) re-checks at CI time.

**Known high-arity APIs that require mitigation:**

| API | Natural fields | Mitigation in v1 |
|---|---|---|
| `hipModuleLaunchKernel` | 12 (corr_id + f + 3 grid + 3 block + sharedMem + stream + kernelParams + extra) | `pack: [gridDim, blockDim]` using `dim3_packed` → 8 fields. |
| `hipExtModuleLaunchKernel` | 14 (adds 2 global grid dims) | `pack: [gridDim, blockDim, globalGridDim]` using `dim3_packed` → 9 fields. |
| `hsa_queue_create` | 10 (corr_id + agent + size + type + callback + data + private_seg + group_seg + queue) | At limit; no mitigation needed (9 payload fields). |
| `hsa_amd_memory_async_copy` | 10 (corr_id + dst + dst_agent + src + src_agent + size + num_dep + dep_signals + completion_signal) | At limit; no mitigation needed. |
| `hsa_amd_memory_async_copy_on_engine` | 11 (adds engine_id) | Drop `dep_signals` pointer field → 10 (9 payload). |

Any future API additions must pass the 9-payload-field budget at codegen time.

### 4.2 Direction semantics

- `IN` — captured at wrapper entry into a `__rocm_in_<name>` C local; emitted at exit.
- `OUT` — captured at exit by deref'ing the original out-pointer parameter. Only when `status == hipSuccess` (HIP) or `HSA_STATUS_SUCCESS` (HSA); on error path emitted as 0/empty.
- `INOUT` — both. IN-side captured into local; OUT-side re-read at exit and emitted as `<name>_out` field.

### 4.3 YAML example

```yaml
- api: hipMemcpyAsync
  category: memory
  args:
    - {name: dst,       type: ptr,    dir: IN}
    - {name: src,       type: ptr,    dir: IN}
    - {name: sizeBytes, type: size,   dir: IN}
    - {name: kind,      type: enum,   dir: IN}
    - {name: stream,    type: handle, dir: IN}

- api: hipMalloc
  category: memory
  args:
    - {name: ptr,  type: ptr,  dir: OUT}
    - {name: size, type: size, dir: IN}

- api: hipLaunchKernel
  category: kernel_launch
  args:
    - {name: function_address, type: ptr,    dir: IN}
    - {name: numBlocks,        type: dim3,   dir: IN}
    - {name: dimBlocks,        type: dim3,   dir: IN}
    - {name: args,             type: ptr,    dir: IN}
    - {name: sharedMemBytes,   type: size,   dir: IN}
    - {name: stream,           type: handle, dir: IN}
```

## 5. Codegen output

### 5.1 `rocm_hip_curated_tp.h` (generated, checked in)

Included from `rocm_hip_tp.h` near the bottom (one new `#include` line). Contains one `LTTNG_UST_TRACEPOINT_EVENT` per curated API:

```c
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hip, hipMemcpyAsync_args,
    LTTNG_UST_TP_ARGS(uint64_t, corr_id,
                      uint64_t, dst, uint64_t, src, uint64_t, sizeBytes,
                      int32_t, kind, uint64_t, stream),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_integer(uint64_t, corr_id, corr_id)
        lttng_ust_field_integer_hex(uint64_t, dst, dst)
        lttng_ust_field_integer_hex(uint64_t, src, src)
        lttng_ust_field_integer(uint64_t, sizeBytes, sizeBytes)
        lttng_ust_field_integer(int32_t, kind, kind)
        lttng_ust_field_integer_hex(uint64_t, stream, stream)
    )
)
```

Header includes a `SHA256(curated_apis.yaml)` comment for reviewer / CI verification.

### 5.2 `rocm_trace_emit_curated.h` (generated, checked in)

Included from `rocm_trace_emit.h`. One `static inline` helper per API.

All-IN example (`hipMemcpyAsync`):

```c
static inline void rocm_trace_emit_hipMemcpyAsync_args(
    uint64_t corr_id,
    void* dst, const void* src, size_t sizeBytes,
    hipMemcpyKind kind, hipStream_t stream) {
    if (rocm_trace_disabled()) return;
    if (lttng_ust_tracepoint_enabled(rocm_hip, hipMemcpyAsync_args)) {
        lttng_ust_do_tracepoint(rocm_hip, hipMemcpyAsync_args, corr_id,
            (uint64_t)(uintptr_t)dst, (uint64_t)(uintptr_t)src,
            (uint64_t)sizeBytes, (int32_t)kind,
            (uint64_t)(uintptr_t)stream);
    }
}
```

OUT-param example (`hipMalloc`):

```c
static inline void rocm_trace_emit_hipMalloc_args(
    uint64_t corr_id,
    void** ptr_out, size_t size, hipError_t status) {
    if (rocm_trace_disabled()) return;
    if (lttng_ust_tracepoint_enabled(rocm_hip, hipMalloc_args)) {
        const uint64_t ptr_val =
            (status == hipSuccess && ptr_out != nullptr)
                ? (uint64_t)(uintptr_t)(*ptr_out) : 0ULL;
        lttng_ust_do_tracepoint(rocm_hip, hipMalloc_args, corr_id,
            ptr_val, (uint64_t)size);
    }
}
```

`dim3` expansion (`hipLaunchKernel`):

```c
static inline void rocm_trace_emit_hipLaunchKernel_args(
    uint64_t corr_id,
    const void* function_address, dim3 numBlocks, dim3 dimBlocks,
    void** args, size_t sharedMemBytes, hipStream_t stream) {
    if (rocm_trace_disabled()) return;
    if (lttng_ust_tracepoint_enabled(rocm_hip, hipLaunchKernel_args)) {
        lttng_ust_do_tracepoint(rocm_hip, hipLaunchKernel_args, corr_id,
            (uint64_t)(uintptr_t)function_address,
            (uint32_t)numBlocks.x, (uint32_t)numBlocks.y, (uint32_t)numBlocks.z,
            (uint32_t)dimBlocks.x, (uint32_t)dimBlocks.y, (uint32_t)dimBlocks.z,
            (uint64_t)(uintptr_t)args, (uint64_t)sharedMemBytes,
            (uint64_t)(uintptr_t)stream);
    }
}
```

When `HIP_ENABLE_LTTNG_UST=0`, the `#else` branch provides empty no-op definitions for all 78 helpers — preserves zero-cost OFF mode.

## 6. Migrator changes

`lttng_migrate.py` (HIP + HSA) gains `--curated-yaml <path>`. For wrappers in the curated set, three transforms are applied in addition to the existing generic-enter/exit injection:

### 6.1 IN-param capture into stable C locals

Right after `__rocm_corr` allocation:

```c
// ---- BEGIN curated-args IN capture (auto-inserted) ----
void* const          __rocm_in_dst       = dst;
const void* const    __rocm_in_src       = src;
const size_t         __rocm_in_sizeBytes = sizeBytes;
const hipMemcpyKind  __rocm_in_kind      = kind;
hipStream_t const    __rocm_in_stream    = stream;
// ---- END curated-args IN capture ----
```

For OUT-only args, no IN-local; the wrapper's original out-pointer parameter is used at exit. For INOUT, the IN-local captures the in-value; the OUT side reads from the original pointer at exit.

**Curated marker (independent of IN locals).** Because some curated APIs have only OUT args (e.g. `hipStreamCreate`) or no args at all (e.g. `hipDeviceSynchronize`), `__rocm_in_*` is not a reliable indicator that a wrapper was migrated as curated. The migrator therefore inserts a dedicated single-line sentinel comment as the first statement of every curated wrapper, immediately after the existing `__rocm_corr` allocation:

```c
/* __ROCM_CURATED__: <api_name> */
```

The sentinel is a stable, comment-only marker emitted unconditionally for every curated wrapper, regardless of arg shape. Idempotency and the coverage gate both key off this sentinel (see §6.3 and §8.2).

### 6.2 New `_CURATED` macros

Three new variants of the existing `ROCM_TRACE_RET_*` macros:

```c
#define ROCM_TRACE_RET_STATUS_CURATED(api, expr, ...) \
    do {                                              \
        const hipError_t __rocm_status = (expr);      \
        rocm_trace_emit_##api##_args(__VA_ARGS__);    \
        rocm_trace_emit_hip_api_exit_status(__func__, \
            __rocm_corr, (int32_t)__rocm_status);     \
        return __rocm_status;                         \
    } while (0)

/* Similar for _PTR_CURATED and _VOID_CURATED. */
```

The migrator rewrites `return <expr>;` to `ROCM_TRACE_RET_STATUS_CURATED(<api>, <expr>, __rocm_corr, <captured-args>);`.

### 6.3 Idempotency

Re-running the migrator on a curated wrapper is a no-op when the `/* __ROCM_CURATED__: <api> */` sentinel (see §6.1) is already present in the body. The sentinel is emitted for every curated wrapper, including OUT-only and no-arg APIs, so this rule applies uniformly.

### 6.4 Ordering

The args event fires **after the call returns** and **before** the generic exit event. Consumers join args event payload with the matching `hip_api_enter` timestamp via `corr_id` if they need entry-time framing. We deliberately do not emit args at entry — the saving from one-event-with-everything outweighs the small loss of timeline precision.

### 6.5 HSA mirror

Identical scheme in HSA. `_CURATED_HSA` macro variants substitute `hsa_status_t` and `HSA_STATUS_SUCCESS`.

## 7. Edge cases

### 7.1 `void**` parameters
Captured as outer-pointer-only (`uint64_t` hex). No deref. Documented in DSL: `{name: args, type: ptr, dir: IN}`.

### 7.2 Struct-pointer parameters (`hipMemcpy3DParms*`, etc.)
Captured as pointer-only. No struct field exploration. If a future need emerges, add a `struct_field` DSL type then.

### 7.3 `dim3` capture
Migrator captures the `dim3` IN-param as a single `dim3 const __rocm_in_<name>` C local (preserving the type). Field expansion to `.x`/`.y`/`.z` happens inside the emit helper.

### 7.4 Enum capture
As `int32_t`. Symbolic-name resolution is consumer-side.

### 7.5 `cstring` (kernel name)
NULL-safe wrapper before `lttng_ust_field_string`: `kname ? kname : ""`. Strict `tracepoint_enabled()` guard because string copy is non-trivial.

### 7.6 Exception path
CATCH does NOT emit `<api>_args`. Documented in schema header. If consumers need IN-params on the failure path, they reconstruct from a nearby successful args event.

## 8. Tests

### 8.1 New test scripts
- `test_hip_curated_args_payload.sh` — picks ~8 representative APIs across categories; asserts payload field correctness via `babeltrace2` + grep.
- `test_hip_curated_args_coverage.sh` — runs a generated helper that calls every curated API; asserts every API's `_args` event appears with matching `corr_id` linkage to generic enter/exit.
- HSA mirrors of both.

### 8.2 Coverage gate update (`lttng_coverage_gate.sh`)
- Verifies every API in `curated_apis.yaml` is present in the migration inventory.
- Body-content scan: each curated wrapper contains the `/* __ROCM_CURATED__: <api> */` sentinel AND a `_CURATED(` macro invocation.
- Additionally, when the API's YAML entry has at least one IN or INOUT arg, the wrapper body must contain at least one `__rocm_in_` local. APIs with only OUT args or no args are exempt from the `__rocm_in_` check.

### 8.3 `lttng_curated_verify.py` (separate CI gate)
- Loads YAML.
- For each API, looks up declaration in `/opt/rocm/include` via libclang.
- Verifies arg count + per-arg type match per the type-vocabulary mapping. Mismatch → **hard error**.
- Verifies every YAML `name` exists as a parameter in the header declaration. Mismatch → **hard error** (this is a correctness invariant: the migrator binds by name).
- Reports unused header parameters (declared but not in YAML) as informational only — partial coverage of large APIs is by design.

### 8.4 Existing tests unchanged
`test_hip_invariants.sh`, `test_hip_api_tracepoints.sh`, and HSA equivalents test generic enter/exit invariants which are unmodified.

## 9. Build wiring

### 9.1 Build/CI dependency contract

| Stage | Python | PyYAML | libclang | When invoked |
|---|---|---|---|---|
| Default build | not required | not required | not required | Always — consumes checked-in headers. |
| Developer `make regenerate-lttng-curated` | required | required | not required | Opt-in target only. |
| CI drift gate (`lttng_curated_codegen` + `git diff`) | required | required | not required | Per CI run. |
| CI verify gate (`lttng_curated_verify.py`) | required | required | required | Per CI run. |

**Default build does not regenerate.** The custom command is wired to a *manual* target named `regenerate-lttng-curated`. It is **not** added as a dependency of `amdhip64` or any default-built target. This guarantees the build does not require Python or PyYAML at compile time.

### 9.2 CMake (HIP, mirrored for HSA)

```cmake
# Optional Python detection — informational only at configure time.
find_package(Python3 QUIET COMPONENTS Interpreter)
if(NOT Python3_FOUND)
    message(STATUS "Python3 not found; LTTng curated regeneration target unavailable.")
else()
    add_custom_command(
        OUTPUT ${CURATED_TP_H} ${CURATED_EMIT_H}
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/scripts/lttng_curated_codegen.py
                --yaml ${CURATED_YAML} --provider rocm_hip
                --out-tp ${CURATED_TP_H} --out-emit ${CURATED_EMIT_H}
        DEPENDS ${CURATED_YAML}
                ${CMAKE_CURRENT_SOURCE_DIR}/scripts/lttng_curated_codegen.py
    )
    # Manual target only — NOT a dependency of amdhip64. Default build
    # consumes the checked-in ${CURATED_TP_H} / ${CURATED_EMIT_H} directly.
    add_custom_target(regenerate-lttng-curated
                      DEPENDS ${CURATED_TP_H} ${CURATED_EMIT_H})
endif()
```

`pyyaml` is detected at codegen-script invocation time, not at CMake configure time. If a developer invokes `regenerate-lttng-curated` without PyYAML installed, the script aborts with a clear error.

### 9.3 CI gates

```yaml
- name: LTTng curated YAML drift check
  run: |
    python3 projects/clr/hipamd/scripts/lttng_curated_verify.py ...
    python3 projects/rocr-runtime/.../scripts/lttng_curated_verify.py ...

- name: LTTng curated codegen idempotency check
  run: |
    python3 projects/clr/hipamd/scripts/lttng_curated_codegen.py ...
    git diff --exit-code -- '**/rocm_*_curated_tp.h' '**/rocm_trace_emit_curated.h'
```

## 10. Delivery

- **Vehicle:** Folded into PR [#5475](https://github.com/ROCm/rocm-systems/pull/5475) as additional commits on `users/bewelton/lttng`.
- **PR growth:** 46 → ~58 files; +6.3k LOC → ~9.5k LOC.
- **Re-review:** Triggers a fresh debate-driven-development pass on the augmented PR.
- **Estimated effort:** ~11-12 working days end-to-end.

## 11. Out-of-scope (deferred)

- Struct-field exploration (`hipMemcpy3DParms.extent.*`).
- `void**` array deref (kernel argument blob capture).
- Symbolic enum-name string capture.
- RCCL coverage (separate library, separate future PR).
- Per-API success/failure aggregation (consumer-side concern).
- Args event on exception path (currently skipped).

---

## Appendix A — Full curated API list (78 APIs)

### A.1 HIP streams + sync (12)

| API | Args |
|---|---|
| `hipStreamCreate` | OUT handle stream |
| `hipStreamCreateWithFlags` | OUT handle stream, IN uint32 flags |
| `hipStreamCreateWithPriority` | OUT handle stream, IN uint32 flags, IN int32 priority |
| `hipStreamDestroy` | IN handle stream |
| `hipStreamGetFlags` | IN handle stream, OUT uint32 flags |
| `hipStreamGetPriority` | IN handle stream, OUT int32 priority |
| `hipStreamSynchronize` | IN handle stream |
| `hipStreamWaitEvent` | IN handle stream, IN handle event, IN uint32 flags |
| `hipStreamQuery` | IN handle stream |
| `hipStreamAddCallback` | IN handle stream, IN ptr callback, IN ptr userData, IN uint32 flags |
| `hipDeviceSynchronize` | (no args) |
| `hipStreamAttachMemAsync` | IN handle stream, IN ptr dev_ptr, IN size length, IN uint32 flags |

### A.2 HIP events (7)

| API | Args |
|---|---|
| `hipEventCreate` | OUT handle event |
| `hipEventCreateWithFlags` | OUT handle event, IN uint32 flags |
| `hipEventDestroy` | IN handle event |
| `hipEventRecord` | IN handle event, IN handle stream |
| `hipEventSynchronize` | IN handle event |
| `hipEventQuery` | IN handle event |
| `hipEventElapsedTime` | OUT float ms, IN handle start, IN handle end |

### A.3 HIP kernel launch + module (11)

| API | Args |
|---|---|
| `hipLaunchKernel` | IN ptr function_address, IN dim3 numBlocks, IN dim3 dimBlocks, IN ptr args, IN size sharedMemBytes, IN handle stream |
| `hipLaunchCooperativeKernel` | (same as hipLaunchKernel) |
| `hipLaunchCooperativeKernelMultiDevice` | IN ptr launchParamsList, IN int32 numDevices, IN uint32 flags |
| `hipModuleLaunchKernel` | IN handle f, IN dim3_packed gridDim, IN dim3_packed blockDim, IN uint32 sharedMemBytes, IN handle stream, IN ptr kernelParams, IN ptr extra (packed per §4.4 to fit 9-field budget) |
| `hipExtModuleLaunchKernel` | extends hipModuleLaunchKernel with IN dim3_packed globalGridDim (packed per §4.4) |
| `hipExtLaunchKernel` | (hipLaunchKernel + IN handle startEvent, IN handle stopEvent) |
| `hipExtLaunchMultiKernelMultiDevice` | IN ptr launchParamsList, IN int32 numDevices, IN uint32 flags |
| `hipModuleGetFunction` | OUT handle function, IN handle module, IN cstring kname |
| `hipModuleLoadData` | OUT handle module, IN ptr image |
| `hipModuleLoadDataEx` | OUT handle module, IN ptr image, IN uint32 numOptions, IN ptr options, IN ptr optionValues |
| `hipModuleUnload` | IN handle module |

### A.4 HIP memory (26)

Synchronous copies:
- `hipMemcpy` — IN ptr dst, IN ptr src, IN size sizeBytes, IN enum kind
- `hipMemcpyDtoH` — IN ptr dst, IN device_ptr src, IN size size
- `hipMemcpyHtoD` — IN device_ptr dst, IN ptr src, IN size size
- `hipMemcpyDtoD` — IN device_ptr dst, IN device_ptr src, IN size size
- `hipMemcpyPeer` — IN ptr dst, IN int32 dstDeviceId, IN ptr src, IN int32 srcDeviceId, IN size sizeBytes

Asynchronous copies:
- `hipMemcpyAsync`, `hipMemcpyDtoHAsync`, `hipMemcpyHtoDAsync`, `hipMemcpyDtoDAsync`, `hipMemcpyPeerAsync` (each: prior + IN handle stream)
- `hipMemcpy2DAsync` — IN ptr dst, IN size dpitch, IN ptr src, IN size spitch, IN size width, IN size height, IN enum kind, IN handle stream
- `hipMemcpy3DAsync` — IN ptr p, IN handle stream

Allocation / free:
- `hipMalloc` — OUT ptr ptr, IN size size
- `hipMallocAsync` — OUT ptr ptr, IN size size, IN handle stream
- `hipMallocHost` / `hipHostMalloc` — OUT ptr ptr, IN size size, IN uint32 flags
- `hipMallocManaged` — OUT ptr ptr, IN size size, IN uint32 flags
- `hipFree` — IN ptr ptr
- `hipFreeAsync` — IN ptr ptr, IN handle stream
- `hipFreeHost` / `hipHostFree` — IN ptr ptr

Memory ops:
- `hipMemset` — IN ptr dst, IN int32 value, IN size sizeBytes
- `hipMemsetAsync` — same + IN handle stream
- `hipMemsetD8` / `D16` / `D32` — IN device_ptr dst, IN uint32 value, IN size count
- `hipMemPrefetchAsync` — IN ptr dev_ptr, IN size count, IN int32 device, IN handle stream
- `hipMemAdvise` — IN ptr dev_ptr, IN size count, IN enum advice, IN int32 device

### A.5 HIP graphs (16)

- `hipGraphCreate` — OUT handle pGraph, IN uint32 flags
- `hipGraphDestroy` — IN handle graph
- `hipGraphInstantiate` — OUT handle pGraphExec, IN handle graph, IN ptr pErrorNode, IN ptr pLogBuffer, IN size bufferSize
- `hipGraphExecDestroy` — IN handle graphExec
- `hipStreamBeginCapture` — IN handle stream, IN enum mode
- `hipStreamEndCapture` — IN handle stream, OUT handle graph
- `hipStreamIsCapturing` — IN handle stream, OUT enum status
- `hipGraphAddKernelNode` — OUT handle pGraphNode, IN handle graph, IN ptr pDependencies, IN size numDependencies, IN ptr pNodeParams
- `hipGraphAddMemcpyNode` — (same shape, memcpy params)
- `hipGraphAddMemsetNode` — (same shape, memset params)
- `hipGraphAddEventRecordNode` — OUT handle pGraphNode, IN handle graph, IN ptr pDependencies, IN size numDependencies, IN handle event
- `hipGraphAddEventWaitNode` — (same as record)
- `hipGraphAddDependencies` — IN handle graph, IN ptr from, IN ptr to, IN size numDependencies
- `hipGraphLaunch` — IN handle graphExec, IN handle stream
- `hipGraphExecKernelNodeSetParams` — IN handle graphExec, IN handle node, IN ptr pNodeParams
- `hipGraphExecMemcpyNodeSetParams1D` — (similar shape)

### A.6 HSA queues + signals + memory (10)

- `hsa_queue_create` — IN handle agent, IN uint32 size, IN enum type, IN ptr callback, IN ptr data, IN uint32 private_segment_size, IN uint32 group_segment_size, OUT handle queue
- `hsa_queue_destroy` — IN handle queue
- `hsa_amd_queue_intercept_create` — (similar)
- `hsa_signal_create` — IN int64 initial_value, IN uint32 num_consumers, IN ptr consumers, OUT handle signal
- `hsa_signal_destroy` — IN handle signal
- `hsa_amd_signal_create` — (similar)
- `hsa_amd_memory_pool_allocate` — IN handle pool, IN size size, IN uint32 flags, OUT ptr ptr
- `hsa_amd_memory_pool_free` — IN ptr ptr
- `hsa_amd_memory_async_copy` — IN ptr dst, IN handle dst_agent, IN ptr src, IN handle src_agent, IN size size, IN uint32 num_dep_signals, IN ptr dep_signals, IN handle completion_signal
- `hsa_amd_memory_async_copy_on_engine` — (same + IN uint32 engine_id)
