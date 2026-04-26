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
Normal (non-throw) path:
1. wrapper enters
   -> tracepoint: rocm_hip:hip_api_enter        (corr_id, parent_corr_id)  [generic, unchanged]
2. function body runs (IN-params held in C locals across the body)
3. just before return:
   -> tracepoint: rocm_hip:<api>_args           (corr_id, IN-params..., OUT-params...) [NEW typed]
   -> tracepoint: rocm_hip:hip_api_exit_status  (corr_id, status)           [generic, unchanged]

Exception (throw) path:
1. wrapper enters
   -> tracepoint: rocm_hip:hip_api_enter        (corr_id, parent_corr_id)  [generic, unchanged]
2. function body throws; CATCH/CATCHRET runs (see §3.3)
   -> NO <api>_args event
   -> NO hip_api_exit_status event
   (consumer sees an unmatched hip_api_enter, interpreted as "exception thrown")
```

Three events per curated call on the normal path (vs. 2 today). Consumers that don't subscribe to `<api>_args` pay the standard `lttng_ust_tracepoint_enabled()` short-circuit cost (one atomic load + unlikely branch).

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

If the wrapper body throws, neither `<api>_args` nor `hip_api_exit_status` is emitted. This matches the existing behavior of the `CATCH` / `CATCHRET` macros in `hip_table_interface.cpp:55-76`, which only call `rocp_reg_auto_pop()` and return `hip::HandleException<...>()` — they intentionally do **not** emit an exit tracepoint. Consumers therefore observe `hip_api_enter` with no matching exit event on exception paths and must interpret an unmatched enter as "exception thrown" (this is documented today as an accepted limitation; see comment block at `hip_table_interface.cpp:73-76`).

Rationale for not emitting on exception:
- OUT params have not been written; capturing them would yield garbage.
- The existing wrapper macros do not emit exit on this path, and changing that behavior is **out of scope** for this spec (it would require touching all 500+ wrappers and is a separate concern from curated parameter capture).
- The dominant exception is allocation failure, where IN-params alone add little beyond `api_name` from `hip_api_enter`.

This applies symmetrically to the HSA `CATCHRET_HSA` equivalent.

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
| `bool` | `ctf_integer(uint32_t)` | C `bool` / `_Bool`. Captured value is 0 if the C bool is `false`, 1 if `true` (normative: codegen emits `(uint32_t)(!!(<arg>))` to guarantee canonical 0/1 regardless of the underlying `_Bool` storage representation). The verifier (§8.3) treats C `bool` as compatible with the `bool` DSL type only; using `uint32` for a C `bool` parameter is a hard error. |
| `dim3` | 3 × `ctf_integer(uint32_t)` | `dim3`. Auto-expands to `<name>_x`, `<name>_y`, `<name>_z`. |
| `dim3_packed` | 1 × `ctf_integer_hex(uint64_t)` | `dim3`. See dim3_packed encoding below. Used to stay under field-budget for high-arity APIs (see §4.4). |

**`dim3_packed` encoding (`ROCM_DIM3_PACK`).**

The encoding allocates lane widths to match real-world HIP `dim3` ranges:

| Bits | Field | Range | Rationale |
|---|---|---|---|
| 0–31 | `x` | `[0, 2^32-1]` | Grid X may need the full 32 bits in HIP; block X ≤ 1024 fits trivially. |
| 32–47 | `y` | `[0, 65535]` | HIP gridDim.y ≤ 65535 by hardware; blockDim.y ≤ 1024. |
| 48–62 | `z` | `[0, 32767]` | HIP gridDim.z ≤ 65535 in theory but practically ≤ 32767; blockDim.z ≤ 64. Saturates above 0x7FFF. |
| 63 | `overflow` | flag | Reserved overflow indicator. Set iff any dim exceeded its lane. |

The `z` lane is intentionally 15 bits (not 16) so that bit 63 can be unambiguously reserved as the overflow flag. A normal `z` value of `0x8000` therefore saturates to `0x7FFF` and sets bit 63 — the encoding never produces bit 63 = 1 for a non-overflowing dim. This trades the rare upper half of the z range for an unambiguous overflow signal.

Codegen emits a single inline helper macro/function:

```c
#define ROCM_DIM3_OVERFLOW_BIT (1ULL << 63)
#define ROCM_DIM3_Z_MAX        (0x7FFFu)   /* 15-bit lane max */

static inline uint64_t ROCM_DIM3_PACK(dim3 d) {
    /* Saturating pack with single-bit overflow indicator (bit 63).
       x: full 32 bits, never overflows the lane.
       y: 16-bit lane, saturates to 0xFFFF.
       z: 15-bit lane (bits 48-62), saturates to 0x7FFF; bit 63 reserved
          for the overflow flag, so z values >= 0x8000 are treated as
          overflow (saturated, with bit 63 set). */
    const uint64_t x = (uint64_t)d.x;
    const uint32_t y_raw = d.y;
    const uint32_t z_raw = d.z;
    const uint64_t y = (y_raw > 0xFFFFu) ? 0xFFFFu : y_raw;
    const uint64_t z = (z_raw > ROCM_DIM3_Z_MAX) ? ROCM_DIM3_Z_MAX : z_raw;
    const uint64_t overflow = ((y_raw > 0xFFFFu) || (z_raw > ROCM_DIM3_Z_MAX))
                                  ? ROCM_DIM3_OVERFLOW_BIT : 0ULL;
    return x | (y << 32) | (z << 48) | overflow;
}
```

**Overflow policy (normative).** When any dimension exceeds its lane width (only possible for `y` and `z`, since `x` has the full 32 bits):

1. The dim is saturated to its lane maximum (`0xFFFF` for y; `0x7FFF` for z).
2. Bit 63 (`ROCM_DIM3_OVERFLOW_BIT`) is set in the packed value.
3. The saturated value is a **lower bound**, not the true dim. Consumers MUST treat any packed value with bit 63 set as "true y and/or z is unknown but ≥ the lane-saturated value".
4. The codegen does not abort on overflow at runtime — capture must remain branch-light. Overflow is a degraded-data signal, not an error.
5. Because z is a 15-bit lane, **any input `z >= 0x8000` is treated as overflow** (saturated to `0x7FFF`, bit 63 set). This is intentional: it preserves the unambiguous meaning of bit 63 across all encoded values. In practice this only affects `gridDim.z` in the upper half of its theoretical 16-bit range, which is not used by typical HIP launches.

**Testable success criteria (added to §8.1):**

- `test_dim3_packed_normal_range`: assert `ROCM_DIM3_PACK({1, 2, 3}) == 0x0003'0002'00000001` (bit 63 clear).
- `test_dim3_packed_x_full_32bit`: assert `ROCM_DIM3_PACK({0xFFFFFFFF, 1, 1}) == 0x0001'0001'FFFFFFFF` and bit 63 clear.
- `test_dim3_packed_y_overflow`: `ROCM_DIM3_PACK({1, 0x10000, 1})` has bit 63 set, y lane = `0xFFFF`, z lane = 1.
- `test_dim3_packed_z_overflow`: `ROCM_DIM3_PACK({1, 1, 0x10000})` has bit 63 set, z lane (bits 48–62) = `0x7FFF`.
- `test_dim3_packed_z_high_bit_overflow`: `ROCM_DIM3_PACK({1, 1, 0x8000})` has bit 63 set and z lane (bits 48–62) = `0x7FFF` (saturated). This locks the disambiguation between a valid z=0x8000 and overflow — both are encoded as overflow, never as a clean payload.
- `test_dim3_packed_z_max_no_false_overflow`: `ROCM_DIM3_PACK({1, 1, 0x7FFF})` has bit 63 clear and z lane = `0x7FFF` (the maximum non-overflow z value).
- `test_dim3_packed_x_max_no_false_overflow`: x=`0xFFFFFFFF` with y=z=1 must NOT set bit 63 (x lane is wide enough).
| `cstring` | `ctf_string` | `const char*`. NULL-safe (NULL → empty string). Helper guards on `tracepoint_enabled()`. |

### 4.4 Field-budget rule (LTTng-UST 10-field / 20-arg limit)

Every generated `<api>_args` event MUST stay within LTTng-UST's documented limit of **10 LTTng fields** (which corresponds to 20 `LTTNG_UST_TP_ARGS` slots, since each field is 2 args). The mandatory `corr_id` field counts as 1 of the 10. This means each curated API may declare at most **9 fields** of payload after `corr_id`.

Note: `dim3` expands to **3** fields, not 1. Codegen MUST count expanded fields, not DSL entries, when validating against the limit.

**Field counting is post-type-expansion AND post-direction-expansion** (normative). The expanded field count for one DSL arg is `type_expansion × direction_expansion`, where:

- `type_expansion`: `dim3` → 3, `dim3_packed` → 1, every other type (including `bool`) → 1.
- `direction_expansion`: `IN` → 1, `OUT` → 1, `INOUT` → 2 (one input field `<name>` plus one output field `<name>_out`).

Examples: `{name: x, type: uint32, dir: INOUT}` contributes 2 fields. `{name: dims, type: dim3, dir: INOUT}` would contribute 6 fields (`x_x, x_y, x_z, x_out_x, x_out_y, x_out_z`).

When a curated API exceeds 9 payload fields under the natural mapping, the YAML author MUST apply one of the following mitigations, in this order of preference. Both mitigations are expressed entirely through existing per-arg fields (`type`, `name`) — there are no separate top-level `pack:` or `split:` directives in the DSL:

1. **Use `dim3_packed`** in place of `dim3` for one or more `dim3` args (saves 2 fields per dim3). Authored as `{name: <name>, type: dim3_packed, dir: IN}`.
2. **Drop low-value fields** (e.g. captured pointer values for opaque blob args like `kernelParams`, `extra`) by simply omitting them from the `args:` list, and document the omission in the API's YAML `notes:`.

Splitting one logical API into multiple events (`<api>_args` + `<api>_args_ext`) is **explicitly out of scope** for v1 — it would require defining cross-event ordering, naming, and consumer-side join semantics, none of which are needed for the curated API set defined in Appendix A. If a future API genuinely cannot fit in 9 payload fields after applying mitigations 1–2, the response is to file a follow-up that introduces a `split:` mechanism with full spec coverage; v1 codegen treats over-budget without an in-vocabulary mitigation as a hard error.

Codegen enforces this at generation time: it computes the **post-type-expansion AND post-direction-expansion** payload field count (i.e. for each DSL arg, `type_expansion × direction_expansion` per the rules above) and aborts with a clear error if any API exceeds 9 payload fields. Verify script (`lttng_curated_verify.py`) re-checks at CI time using the same expansion rule.

**INOUT scope (v1).** No API in the v1 curated set in Appendix A uses `INOUT` direction — every entry is `IN` or `OUT`. The DSL accepts `INOUT` as a value of `dir` and the field-counting rule above defines its expansion, but v1 codegen MUST reject any YAML entry with `dir: INOUT` as an out-of-scope direction (hard error in both codegen and `lttng_curated_verify.py`). Lifting this restriction is a follow-up: it requires (a) defining the temporal capture order (input-side latched at entry into `__rocm_in_<name>`, output-side re-read at exit only when status indicates success per §4.2), (b) confirming the doubled field budget against §4.4 for any candidate API, and (c) extending the migrator's per-arg binding to emit both `<name>` and `<name>_out` payload entries. Until then, an INOUT parameter must be modeled as a single `IN` capture (input value only) or a single `OUT` capture (final value only), and the YAML author MUST document the choice in the API's `notes:`.

**Known high-arity APIs that require mitigation:**

Counts in the "Natural fields" column include `corr_id` plus all expanded payload fields (each `dim3` counted as 3). The "Total after mitigation" column also includes `corr_id`. The 10-field LTTng limit corresponds to a 9-payload-field budget.

| API | Natural fields (incl. corr_id, dim3 expanded) | Mitigation in v1 | Total after mitigation |
|---|---|---|---|
| `hipLaunchKernel` | 11 (corr_id + function_address + 3 numBlocks + 3 dimBlocks + args + sharedMemBytes + stream) | Type `numBlocks` and `dimBlocks` as `dim3_packed` → saves 4 fields. | 7 |
| `hipLaunchCooperativeKernel` | 11 (same shape as `hipLaunchKernel`) | Type `numBlocks` and `dimBlocks` as `dim3_packed`. | 7 |
| `hipExtLaunchKernel` | 13 (adds 2 event handles) | Type `numBlocks` and `dimBlocks` as `dim3_packed`. | 9 |
| `hipModuleLaunchKernel` | 12 (corr_id + f + 3 grid + 3 block + sharedMem + stream + kernelParams + extra) | Type `gridDim` and `blockDim` as `dim3_packed` → 8 fields. | 8 |
| `hipExtModuleLaunchKernel` | 14 (adds 2 global grid dims) | Type `gridDim`, `blockDim`, and `globalGridDim` as `dim3_packed`. | 9 |
| `hsa_queue_create` | 9 (corr_id + agent + size + type + callback + data + private_seg + group_seg + queue) | None needed (8 payload fields, 1 spare). | 9 |
| `hsa_amd_memory_async_copy` | 9 (corr_id + dst + dst_agent + src + src_agent + size + num_dep + dep_signals + completion_signal) | None needed (8 payload fields, 1 spare). | 9 |
| `hsa_amd_memory_async_copy_on_engine` | 11 (`hsa_amd_memory_async_copy` + `engine_id` + `force_copy_on_sdma`) | Drop `dep_signals` from the YAML `args:` list → 9 payload + corr_id = 10 total (at limit). | 10 |

Any future API additions must pass the 9-payload-field budget at codegen time.

### 4.2 Direction semantics

- `IN` — captured at wrapper entry into a `__rocm_in_<name>` C local; emitted at exit.
- `OUT` — captured at exit by deref'ing the original out-pointer parameter. Only when `status == hipSuccess` (HIP) or `HSA_STATUS_SUCCESS` (HSA); on error path emitted as 0/empty.
- `INOUT` — both. IN-side captured into local; OUT-side re-read at exit and emitted as `<name>_out` field. **Out of scope for v1** — see §4.4 ("INOUT scope (v1)"); v1 codegen and the verifier reject `dir: INOUT`.

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
  # numBlocks + dimBlocks packed per §4.4 to fit the 9-payload-field budget
  # (without packing this would be 10 payload fields).
  args:
    - {name: function_address, type: ptr,         dir: IN}
    - {name: numBlocks,        type: dim3_packed, dir: IN}
    - {name: dimBlocks,        type: dim3_packed, dir: IN}
    - {name: args,             type: ptr,         dir: IN}
    - {name: sharedMemBytes,   type: size,        dir: IN}
    - {name: stream,           type: handle,      dir: IN}
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

All-IN example (`hipMemcpyAsync`). Per the §6.2 helper-signature rule, every curated helper accepts `status` as its last parameter even when unused:

```c
static inline void rocm_trace_emit_hipMemcpyAsync_args(
    uint64_t corr_id,
    void* dst, const void* src, size_t sizeBytes,
    hipMemcpyKind kind, hipStream_t stream,
    hipError_t /*status*/ /* unused: all-IN API */) {
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

`dim3_packed` expansion (`hipLaunchKernel`):

```c
static inline void rocm_trace_emit_hipLaunchKernel_args(
    uint64_t corr_id,
    const void* function_address, dim3 numBlocks, dim3 dimBlocks,
    void** args, size_t sharedMemBytes, hipStream_t stream,
    hipError_t /*status*/ /* unused: all-IN API */) {
    if (rocm_trace_disabled()) return;
    if (lttng_ust_tracepoint_enabled(rocm_hip, hipLaunchKernel_args)) {
        /* dim3_packed encoding per §4.1: see ROCM_DIM3_PACK in §4.1. */
        const uint64_t numBlocks_packed = ROCM_DIM3_PACK(numBlocks);
        const uint64_t dimBlocks_packed = ROCM_DIM3_PACK(dimBlocks);
        lttng_ust_do_tracepoint(rocm_hip, hipLaunchKernel_args, corr_id,
            (uint64_t)(uintptr_t)function_address,
            numBlocks_packed, dimBlocks_packed,
            (uint64_t)(uintptr_t)args, (uint64_t)sharedMemBytes,
            (uint64_t)(uintptr_t)stream);
    }
}
```

Resulting payload field count: `function_address` (1) + `numBlocks_packed` (1) + `dimBlocks_packed` (1) + `args` (1) + `sharedMemBytes` (1) + `stream` (1) = **6 payload fields** (well under the 9-field budget). Total event field count = 7 including `corr_id`.

When `HIP_ENABLE_LTTNG_UST=0`, the `#else` branch provides empty no-op definitions for every helper generated from `curated_apis.yaml` — preserves zero-cost OFF mode. The exact count is whatever YAML defines (the YAML is the single source of truth; see §1).

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

Three new variants of the existing `ROCM_TRACE_RET_*` macros. The macros follow a single helper-signature rule: **every `rocm_trace_emit_<api>_args` helper accepts `corr_id` first and the call's status as its last positional argument.** The migrator and codegen both honor this rule, so OUT-only and INOUT direction handling does not require any per-API contract negotiation between the macro and the helper:

- For status-returning APIs (`_STATUS_CURATED`, `_HSA` variant), status is the macro-evaluated `__rocm_status` (`hipError_t` or `hsa_status_t`).
- For pointer-returning APIs (`_PTR_CURATED`), status is synthesized **only for the `_args` helper's OUT-gating logic** from the return value: `__rocm_ptr != nullptr ? hipSuccess : hipErrorOutOfMemory` (this captures the only signal a pointer-returning HIP API conveys to its caller). The synthesized status is **not** emitted as a generic exit event — `_PTR_CURATED` continues to emit the existing `hip_api_exit_ptr` event so the generic schema is preserved verbatim (see the §1 Goal "augment without replacing" and the §2 non-goal "Generic events keep their current schema").
- For void-returning APIs (`_VOID_CURATED`), status is the literal `hipSuccess` — the call cannot fail at this layer, so OUT params (if any) are always treated as valid. The generic exit event remains `hip_api_exit_void`.

Codegen always emits the helper signature with `status` as the last parameter, even for all-IN APIs (where the helper simply ignores it). This keeps the macro/helper contract uniform and avoids per-API special cases.

**Generic exit-event preservation (normative).** Each `_CURATED` macro variant MUST emit the same generic exit tracepoint that its non-curated counterpart emits today:

| Macro | Generic exit tracepoint emitted | Matches non-curated counterpart |
|---|---|---|
| `_STATUS_CURATED` | `hip_api_exit_status` | `ROCM_TRACE_RET_STATUS` (hip_table_interface.cpp:77-83) |
| `_PTR_CURATED` | `hip_api_exit_ptr` | `ROCM_TRACE_RET_PTR` (hip_table_interface.cpp:85-90) |
| `_VOID_CURATED` | `hip_api_exit_void` | `ROCM_TRACE_RET_VOID` (hip_table_interface.cpp:92-97) |

This invariant is verified by `test_hip_api_tracepoints.sh` (existing) and re-asserted by `test_hip_curated_args_payload.sh` (new): for each curated pointer-returning API, the trace MUST contain both `<api>_args` and `hip_api_exit_ptr` (carrying `retval_ptr`), never `hip_api_exit_status`.

```c
/* Captured-args variants (one or more captured args). __VA_ARGS__ is
   non-empty by construction — see the migrator rule below and the
   no-arg variants further down. */
#define ROCM_TRACE_RET_STATUS_CURATED(api, expr, corr, ...)              \
    do {                                                                 \
        const hipError_t __rocm_status = (expr);                         \
        rocm_trace_emit_##api##_args((corr), __VA_ARGS__, __rocm_status);\
        rocm_trace_emit_hip_api_exit_status(__func__,                    \
            (corr), (int32_t)__rocm_status);                             \
        return __rocm_status;                                            \
    } while (0)

#define ROCM_TRACE_RET_PTR_CURATED(api, ptr_type, expr, corr, ...)       \
    do {                                                                 \
        ptr_type const __rocm_ptr = (expr);                              \
        /* Status is synthesized ONLY for the _args helper's OUT-gating  \
           logic; it is NOT emitted as a generic exit event. The generic \
           exit event remains hip_api_exit_ptr to preserve the existing  \
           pointer-return schema (see §6.2 generic-exit preservation). */\
        const hipError_t __rocm_status =                                 \
            (__rocm_ptr != nullptr) ? hipSuccess : hipErrorOutOfMemory;  \
        rocm_trace_emit_##api##_args((corr), __VA_ARGS__, __rocm_status);\
        rocm_trace_emit_hip_api_exit_ptr(__func__, (corr), __rocm_ptr);  \
        return __rocm_ptr;                                               \
    } while (0)

#define ROCM_TRACE_RET_VOID_CURATED(api, expr, corr, ...)                \
    do {                                                                 \
        (expr);                                                          \
        rocm_trace_emit_##api##_args((corr), __VA_ARGS__, hipSuccess);   \
        rocm_trace_emit_hip_api_exit_void(__func__, (corr));             \
        return;                                                          \
    } while (0)

/* Zero-captured-args variants. Required because the codebase is built
   under a mix of C++14 / C++17 / C++20 (rocclr.cmake, hiprtc CMakeLists,
   src/CMakeLists), so __VA_OPT__ is not portable across all
   translation units. The migrator selects the _NOARGS variant whenever
   the curated API has zero captured args (e.g. hipDeviceSynchronize) —
   this avoids any empty-__VA_ARGS__ expansion entirely. */
#define ROCM_TRACE_RET_STATUS_CURATED_NOARGS(api, expr, corr)            \
    do {                                                                 \
        const hipError_t __rocm_status = (expr);                         \
        rocm_trace_emit_##api##_args((corr), __rocm_status);             \
        rocm_trace_emit_hip_api_exit_status(__func__,                    \
            (corr), (int32_t)__rocm_status);                             \
        return __rocm_status;                                            \
    } while (0)

#define ROCM_TRACE_RET_PTR_CURATED_NOARGS(api, ptr_type, expr, corr)     \
    do {                                                                 \
        ptr_type const __rocm_ptr = (expr);                              \
        const hipError_t __rocm_status =                                 \
            (__rocm_ptr != nullptr) ? hipSuccess : hipErrorOutOfMemory;  \
        rocm_trace_emit_##api##_args((corr), __rocm_status);             \
        rocm_trace_emit_hip_api_exit_ptr(__func__, (corr), __rocm_ptr);  \
        return __rocm_ptr;                                               \
    } while (0)

#define ROCM_TRACE_RET_VOID_CURATED_NOARGS(api, expr, corr)              \
    do {                                                                 \
        (expr);                                                          \
        rocm_trace_emit_##api##_args((corr), hipSuccess);                \
        rocm_trace_emit_hip_api_exit_void(__func__, (corr));             \
        return;                                                          \
    } while (0)
```

The migrator rewrites `return <expr>;` to one of:

- `ROCM_TRACE_RET_STATUS_CURATED(<api>, <expr>, __rocm_corr, <captured-args>);` — when the API has at least one captured arg.
- `ROCM_TRACE_RET_STATUS_CURATED_NOARGS(<api>, <expr>, __rocm_corr);` — when the API has zero captured args (e.g. `hipDeviceSynchronize`).

The `_NOARGS` variant selection is mechanical: the migrator counts captured args (post `dim3_packed` packing) for the API and emits the `_NOARGS` form iff the count is zero. The macro itself appends `__rocm_status` to the helper call site in either form, so the helper signature invariant of §6.2 ("`corr_id` first, `status` last") holds for no-arg APIs too — the helper signature is just `void rocm_trace_emit_<api>_args(uint64_t corr_id, hipError_t status)`. The migrator never emits `__rocm_status` into the macro's argument list.

Helper signature rule (codegen invariant): the helper for any curated API has signature

```
void rocm_trace_emit_<api>_args(uint64_t corr_id,
                                <captured-args...>,
                                <status_type> status);
```

where `<status_type>` is `hipError_t` (HIP) or `hsa_status_t` (HSA). All-IN APIs ignore `status`; OUT/INOUT APIs use it to gate dereference per §4.2.

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
On the throw path, the existing `CATCH` / `CATCHRET` macros emit neither `<api>_args` nor `hip_api_exit_status` (see §3.3). Consumers see `hip_api_enter` without a matching exit, which is the documented existing behavior.

**Per-call IN-params on the exception path are unavailable.** Because args are emitted just before return (§3.1) and the throw bypasses that emit point, no `<api>_args` event exists for the failed call, and no other event in the stream carries that call's parameters. Consumers MUST treat an unmatched `hip_api_enter` as "exception thrown; per-call parameters unknown" — reconstruction from a nearby successful args event on the same TID is **not valid**, since correlation IDs do not match, "nearby" has no defined bound, and adjacent calls to the same API can carry different arguments. Documented in schema header.

Adding exception-path IN-param visibility is a separate design: it would require either (a) emitting the args event at wrapper entry instead of just before return (changes ordering semantics for all consumers and doubles the event count for normal-path calls), or (b) modifying the `CATCH` / `CATCHRET` macros to emit on the exception path (explicitly out-of-scope per §3.3). Either option is a follow-up spec, not a v1 change.

## 8. Tests

### 8.1 New test scripts
- `test_hip_curated_args_payload.sh` — picks ~8 representative APIs across categories; asserts payload field correctness via `babeltrace2` + grep. Includes at least one zero-captured-args API (`hipDeviceSynchronize`) to verify the `_NOARGS` macro variant compiles cleanly and emits its `_args` event with only `corr_id` payload.
- `test_hip_curated_args_coverage.sh` — runs a generated helper that calls every curated API; asserts every API's `_args` event appears with matching `corr_id` linkage to generic enter/exit.
- HSA mirrors of both.

### 8.2 Coverage gate update (`lttng_coverage_gate.sh`)
- Verifies every API in `curated_apis.yaml` is present in the migration inventory.
- Body-content scan: each curated wrapper contains the `/* __ROCM_CURATED__: <api> */` sentinel AND a curated-macro invocation. The normative matcher is the regex `ROCM_TRACE_RET_(STATUS|PTR|VOID)_CURATED(_NOARGS)?(_HSA(_NOARGS)?)?\(` so that all six curated-macro variants (`_STATUS_CURATED`, `_PTR_CURATED`, `_VOID_CURATED`, plus their `_NOARGS`, `_HSA`, and `_HSA_NOARGS` forms) match. (A naive substring scan for `_CURATED(` would miss `_CURATED_NOARGS(` and `_CURATED_HSA(`.)
- Additionally, when the API's YAML entry has at least one IN or INOUT arg, the wrapper body must contain at least one `__rocm_in_` local. APIs with only OUT args or no args are exempt from the `__rocm_in_` check.

### 8.3 `lttng_curated_verify.py` (separate CI gate)
- Loads YAML.
- For each API, looks up declaration in `/opt/rocm/include` via libclang.
- Verifies arg count + per-arg type match per the type-vocabulary mapping (§4.1). Mismatch → **hard error**. C `bool` / `_Bool` parameters in the header MUST be declared as DSL type `bool` in YAML; using `uint32` (or any other DSL type) for a C bool parameter is a hard error so that the canonical 0/1 conversion in §4.1 is always applied.
- Re-runs the §4.4 field-budget check using the post-type-expansion AND post-direction-expansion rule. Over-budget → **hard error**.
- Rejects any YAML entry with `dir: INOUT` as out-of-scope for v1 (see §4.4 "INOUT scope (v1)"). **Hard error**.
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

## Appendix A — Illustrative curated API list (~82 APIs)

> **Note:** This appendix is illustrative of the intended curation as of spec drafting. The authoritative source of curated APIs is `curated_apis.yaml` (HIP) and the HSA mirror; success criteria for coverage tests count against YAML, not this appendix. The category subtotals below sum to 82, matching the "~82" figure used in §1.

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
- `hsa_amd_memory_async_copy` — IN ptr dst, IN handle dst_agent, IN ptr src, IN handle src_agent, IN size size, IN uint32 num_dep_signals, IN ptr dep_signals, IN handle completion_signal *(8 payload fields; fits the 9-payload budget without mitigation)*
- `hsa_amd_memory_async_copy_on_engine` — same as above plus IN uint32 engine_id, IN bool force_copy_on_sdma *(`force_copy_on_sdma` is C `bool`, declared as the `bool` DSL type per §4.1 → `ctf_integer(uint32_t)` payload; total 10 payload fields; mitigation per §4.4 drops `dep_signals` to fit 9 payload)*
