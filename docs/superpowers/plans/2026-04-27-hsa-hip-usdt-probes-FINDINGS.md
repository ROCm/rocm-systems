# USDT probe implementation: findings and required redesign

**Date:** 2026-04-27
**Branch:** `users/bewelton/bpf-usdt-probes`
**Companion plan:** `2026-04-27-hsa-hip-usdt-probes.md` (now requires updates per below)

## What landed

| Task | Status | Commit | Notes |
|---|---|---|---|
| Plan doc | ✅ | `0d27b9e2` | Will need updates per Findings 2 + 3 below |
| Task 1: FindSystemtap.cmake (HIP + HSA) | ✅ | `6af88cef` | Includes aarch64 path suffix per code-review |
| Task 2 + 3: ROCM_ENABLE_USDT options | ✅ | `ba0a309a` | Verified configure on/off in container |
| Task 4: HIP shim header | ✅ | `2cf420ce` | Untouched, should work as-is |
| Task 6: HSA shim header (initial) | ✅ | `83d0eb20` | EntryGuard struct/method form — **broken** per Finding 1 |
| Task 6: HSA shim refactor to free functions | ✅ | `9d122f16` | Free-function form fixes Finding 1 |
| Tasks 5, 7–17 | ❌ | — | Blocked on redesign per Findings 2 + 3 |

Branch is pushed to `origin/users/bewelton/bpf-usdt-probes`.

## Findings during validation

### Finding 1: `DTRACE_PROBE` from C++ struct member methods produces malformed `.note.stapsdt` args

**Symptom:** `eu-readelf -n` shows `Args: '%rax %rdx'` (no `8@` size prefix) for probes emitted from inside an inlined member method that reads `this->name` / `this->corr`. BPF tools cannot extract such args and refuse to attach (`ERROR: couldn't get argument 0`).

**Root cause:** Interaction between sys/sdt.h's `_SDT_ARGSIZE` size-classification macro and gcc's operand attribute handling for member references in inlined methods. Operand cannot be size-classified, so the asm template emits the operand without the size prefix.

**Reproducer:**
```cpp
struct G {
    const char* name;
    uint64_t corr;
    G(const char* n, uint64_t c) : name(n), corr(c) {
        DTRACE_PROBE2(p, x, name, corr);  // BROKEN
    }
};
```

**Fix already landed (commit `9d122f16`):** Refactor to free functions where the operands are function parameters / locals, which gcc reliably size-classifies:
```cpp
inline uint64_t fire_entry(const char* name) noexcept {
    uint64_t corr = generate_correlation_id();
    DTRACE_PROBE2(hsa, api_entry, name, corr);  // WORKS
    return corr;
}
```

### Finding 2: Ubuntu 22.04's systemtap-sdt-dev 4.6 has duplicate-section codegen bug

**Symptom:** `.note.stapsdt` section contains orphan trailing bytes and `eu-readelf -n` reports "cannot get content of note: garbage data" between consecutive notes. Some BPF tools stop parsing at the first malformed note.

**Root cause:** Ubuntu 22.04's `sys/sdt.h` (4.6-2) emits `_SDT_ASM_3(.pushsection .note.stapsdt, "", "note")` and `_SDT_ASM_1(.popsection)` blocks redundantly in the macro expansion, producing duplicate section data.

**Fix:** Upstream `sys/sdt.h` (5.0+, shipping in Ubuntu 24.04) has cleaner macro structure. Either:
- Document minimum `systemtap-sdt-dev >= 5.0` (Ubuntu 24.04, RHEL 9, Debian Bookworm)
- Or vendor a known-good `sys/sdt.h` into the tree and prefer it over the system header

### Finding 3: BPF cannot fault in cold pages — string-arg pointers fail silently for unmapped .rodata pages

**Symptom:** `bpftrace ... { printf("%s", str(arg0)); }` returns empty string even though `arg0` is a valid .rodata pointer and `strings <binary>` confirms the literal is there. Numeric args (uint64, etc.) work correctly through the same probe.

**Root cause:** `bpf_probe_read_user_str()` reads currently-mapped pages only. It cannot trigger a page fault. If the application has not yet executed any code that touches the .rodata page containing the literal, the page is not mapped at probe time, and the helper silently returns zero bytes.

**Reproducer:**
```cpp
const char* a = "hello";
DTRACE_PROBE1(test, x, a);   // str(arg0) returns "" — page not faulted in
```
vs.
```cpp
const char* a = "hello";
printf("%s", a);             // page fault, page now mapped
DTRACE_PROBE1(test, x, a);   // str(arg0) returns "hello"
```

**Workaround attempts that DON'T work:**
- `asm volatile("" : : "r"(*a) : "memory");` — gcc may keep the byte in a register without dereferencing
- `volatile char ta = *a;` — same; the volatile read doesn't always reach the dereference

**Workaround that DOES work:** Any function call that reads the string content (e.g. `strlen`, `printf`). But this defeats the zero-cost premise of passing a static literal.

**Implications for the original plan:**
The HSA shim's `__func__` arg (a const char* to .rodata literal in the calling function) hits this issue regularly in production. Without fixing it, BPF tools see correlation IDs and return values correctly but cannot tell which HSA function fired the probe.

## Required redesign before continuing Tasks 5, 7–17

The plan must shift from string function names to **integer API IDs** for both HIP and HSA. HIP already has `enum hip_api_id_t` so its shim is mostly fine (it passes `(uint32_t)operation_id`, no string), but the plan's intent for HSA — passing `__func__` — must change.

### New design for HSA

**Generate `hsa_api_id.h`** (script-driven from `hsa_api_trace.h` which has 240 `_fn` entries):

```c
typedef enum {
    HSA_API_ID_NONE = 0,
    HSA_API_ID_hsa_init = 1,
    HSA_API_ID_hsa_shut_down = 2,
    /* ... 240 entries total ... */
    HSA_API_ID_LAST
} hsa_api_id_t;

extern const char* hsa_api_id_to_name(hsa_api_id_t id);
```

The name table lives in a single `.cpp` so BPF tools can parse it via:
```bash
nm libhsa-runtime64.so | grep hsa_api_id_to_name  # find table
# Or generate a parallel hsa_api_id_table.txt at build time and ship it for tools
```

### New shim signature

```cpp
// rocm_hsa_usdt.h
inline uint64_t fire_entry(uint32_t api_id) noexcept {
    uint64_t corr = generate_correlation_id();
    ROCM_HSA_USDT_ENTRY(api_id, corr);
    return corr;
}
```

Probe contract becomes:
- `hsa:api_entry(uint32_t api_id, uint64_t correlation_id)`
- `hsa:api_exit(uint32_t api_id, uint64_t correlation_id, uint64_t status_or_value)`

### New TRY/CATCH macros

The current plan has `TRY` capturing `__func__` automatically. The new design needs the API ID. Options:

**Option A (~200 source-site retrofits):** Make `TRY` take an explicit ID:
```cpp
#define TRY(api_id) return ::rocm::hsa::usdt::run_traced(api_id, [&]() -> hsa_status_t { try {

// Then at every site:
hsa_status_t hsa_init() {
    TRY(HSA_API_ID_hsa_init);
    return core::Runtime::runtime_singleton_->Acquire();
    CATCH;
}
```

**Option B (no source-site change):** Use a runtime hash-of-`__func__` lookup:
```cpp
#define TRY return ::rocm::hsa::usdt::run_traced_by_name(__func__, [&]() -> hsa_status_t { try {

// Inside the runtime:
inline uint32_t lookup_api_id(const char* name) {
    static const std::unordered_map<std::string, uint32_t> map = build_table();
    auto it = map.find(name);
    return it == map.end() ? HSA_API_ID_NONE : it->second;
}
```
**Drawback:** lookup cost on every probe site (even when no observer attached, since fire_entry is always called inside a TRY). Could be mitigated with a thread_local cache per-site, but at that point Option A is simpler.

**Option C (compile-time, tested):** Use a compile-time string hash and have the BPF side consume the same hash:
```cpp
constexpr uint32_t hash_func(const char* s) { /* fnv-1a */ }
#define TRY return ::rocm::hsa::usdt::run_traced(hash_func(__func__), [&]() ...)
```
**Caveat:** `__func__` is NOT `constexpr` in C++17. So this requires either `__PRETTY_FUNCTION__` (also not constexpr in standard C++17) or some other compile-time trick. Possibly `__FUNCTION__` via a macro-stringized form. Worth investigating.

**Recommended:** Option A. It's mechanical work (200 sites, scriptable) and has zero runtime overhead. Each site already has the function name visible in its `hsa_<name>` function declaration; a `sed` script can convert `TRY;` to `TRY(HSA_API_ID_<name>);` by reading the enclosing function definition.

## What you should do next

1. **Generate `hsa_api_id.h`** — script that parses `hsa_api_trace.h`'s `_fn` table and emits the enum. ~30 min.
2. **Update HSA shim** to take `uint32_t api_id`. Replace `name` everywhere. ~15 min.
3. **Sed-script the TRY retrofit** across hsa.cpp / hsa_ext_amd.cpp / hsa_ext_image.cpp / hsa_ven_amd_pc_sampling.cpp. Manual review for any unusual patterns. ~1 hour.
4. **Update plan doc** Tasks 5–17 to reflect the new design.
5. **Resume execution** — Tasks 7–17 are mostly unchanged in shape, just with the new arg signature.
6. **Document toolchain minimums** in build docs: systemtap-sdt-dev ≥ 5.0, bpftrace ≥ 0.20, kernel ≥ 5.10 with CAP_BPF + CAP_PERFMON.

## What's still useful from this session

- **The build-side work (Tasks 1–3) is reusable as-is.** No changes needed.
- **The HIP shim (Task 4) is reusable as-is.** HIP already uses integer API IDs, so it doesn't have Finding 3's problem.
- **The HSA shim (Task 6) needs the API-ID swap** but the free-function structure (Finding 1's fix) and the to_u64 helper stay.
- **The plan structure is sound.** Tasks 7–9 still want lambda-wrapping TRY/CATCH; the only delta is the arg signature.

## Validation evidence captured

End-to-end on container `bewelton_bpf` (`rocm/dev-ubuntu-24.04:latest` on banff):
- ✅ Compile + run with `ROCM_ENABLE_USDT=1`
- ✅ `bpftrace -l 'usdt:bin:hsa:*'` lists both `api_entry` and `api_exit`
- ✅ `bpftrace -e '... { printf("%d %d", arg1, arg2); }'` extracts numeric args correctly
- ❌ `str(arg0)` for `__func__` pointer returns empty (Finding 3) — fixed by API-ID redesign
