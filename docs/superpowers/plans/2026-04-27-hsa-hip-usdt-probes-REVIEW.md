# Cross-vendor adversarial review of USDT foundation

**Date:** 2026-04-27
**Branch reviewed:** `users/bewelton/bpf-usdt-probes` at commit `c3a57b68`
**Reviewer:** OpenAI GPT 5.5 (cross-vendor — different blind spots from the Anthropic implementer)
**Verdict:** REQUEST_REVISIONS

The foundation is not trash, but it is not safe to continue Tasks 7–17 on top of it without correcting several design assumptions. Keep the build plumbing and the HIP shim shape; revise the HSA shim and the plan before doing the large source-site retrofit.

## Critical issues

### 1. HSA "disabled-build" cost is NOT zero
`projects/rocr-runtime/runtime/hsa-runtime/core/inc/usdt/rocm_hsa_usdt.h:84-114`

`fire_entry()` always calls `generate_correlation_id()`, which performs a process-global atomic `fetch_add()` at lines 54-56, even when `ROCM_ENABLE_USDT` is off. Only `ROCM_HSA_USDT_ENTRY/EXIT` become no-ops. If Tasks 7–9 wire `run_traced()` into TRY/CATCH as planned, **every HSA API call pays a global atomic RMW in USDT-disabled builds.** This contradicts the documented "Disabled-build cost: zero" claim.

**Fix:** When `ROCM_ENABLE_USDT` is off, `run_traced` must skip the correlation_id generation entirely — expand to just running the lambda.

### 2. Planned HIP correlation pairing is broken
`docs/superpowers/plans/2026-04-27-hsa-hip-usdt-probes.md:346,369`

The plan emits entry with `hip::tls.usdt_correlation_id_++` (post-increment), then emits exit with `hip::tls.usdt_correlation_id_`. Entry gets the OLD value, exit gets the INCREMENTED value — they don't pair. Also breaks under nested HIP API calls.

**Fix:** Store the generated ID in the RAII spawner object's member; pre-increment or generate-then-store.

### 3. HSA shim still exposes the obsolete string-pointer probe contract
`projects/rocr-runtime/runtime/hsa-runtime/core/inc/usdt/rocm_hsa_usdt.h:10-14,84-99`

Findings doc says `const char* func_name` is unusable (BPF can't fault cold pages), but the committed shim still documents and implements `hsa:api_entry(const char* func_name, ...)`. Probe ABIs fossilize quickly once scripts use them. Do not preserve a known-bad contract on the branch.

**Fix:** Switch the HSA shim to `(uint32_t api_id, uint64_t corr)` BEFORE any HSA call sites use it.

## Important issues

### 4. Process-global atomic correlation counter is wrong for HSA hot paths
Even in USDT-enabled/unattached builds, every traced HSA API call contends on one cache line across all application threads. Much more expensive than "one nop." HIP's planned TLS counter is cheaper (once #2 is fixed). HSA should use per-thread counters or packed `(tid, local_counter)`.

### 5. FindSystemtap doesn't validate the toolchain works
`projects/clr/hipamd/src/cmake/FindSystemtap.cmake:9-17` (and HSA copy)

Findings doc says Ubuntu 22.04's systemtap-sdt-dev 4.6 produces malformed `.note.stapsdt`. Find module only checks the header exists. Configure will happily succeed on a known-broken environment.

**Fix:** Add a `try_compile` step that compiles a small probe object and validates the args spec with `readelf -n` or eu-readelf. Or vendor a known-good `sys/sdt.h`.

### 6. CMake wiring is globally scoped
Already flagged in earlier review. `include_directories()` / `add_compile_definitions()` should be `Systemtap::sdt` linked into specific targets with target-scoped definitions.

### 7. cmake_minimum_required vs add_compile_definitions mismatch
`projects/rocr-runtime/runtime/hsa-runtime/CMakeLists.txt:43,101`

CMakeLists declares minimum 3.7 but uses `add_compile_definitions()` which needs newer. Use `target_compile_definitions()` after target creation, or update the minimum deliberately.

### 8. HSA API ID stability is undefined
`docs/superpowers/plans/2026-04-27-hsa-hip-usdt-probes-FINDINGS.md:92-110`

Generating IDs from `hsa_api_trace.h` order is fine for v1, but if BPF tools consume integer IDs, the IDs ARE the ABI. Need:
- Append-only enum policy
- CI check detecting renumbering
- Documented relationship to `hsa_api_trace_version.h` (already has table major/step versions)

### 9. `hsa_api_id_to_name()` can't be called from BPF
BPF programs can't call arbitrary userspace functions. Need a shipped name table (text/JSON/header) that BPF tools compile against, not just a symbol in the DSO.

### 10. Lambda-wrapping TRY/CATCH is high-risk
`docs/superpowers/plans/2026-04-27-hsa-hip-usdt-probes.md:585-827`

Wrapping public API bodies in lambdas changes control-flow, return typing, debug stack frames, inlining, and possibly sanitizer/profiler behavior. HSA is exception-sensitive and hot-path-sensitive. Needs compiler-matrix validation, optimization-level validation, and tests for void/pointer/enum/signed-int/exception paths BEFORE mass conversion.

### 11. `to_u64()` doc imprecise for signed narrow integers
`projects/rocr-runtime/runtime/hsa-runtime/core/inc/usdt/rocm_hsa_usdt.h:59-75`

Doc says "bit-preserved, not sign-extended" but `static_cast<uint64_t>(int32_t{-1})` yields `0xffffffffffffffff` (sign-extended), not `0x00000000ffffffff`. Define whether exit values are numeric casts or raw bit patterns.

### 12. Planned e2e test correlation join is insufficient
`docs/superpowers/plans/2026-04-27-hsa-hip-usdt-probes.md:1480-1534`

`@corr[tid, arg0]` overwrites under nested/recursive API calls. Test should track a stack or use correlation_id as the primary key.

## Validation of findings

Reviewer agreed all three findings (struct-method probe bug, sdt 4.6 bug, BPF page-fault) are credible. But:

- **Finding 1's lesson** isn't "structs are bad" — it's "USDT macro args must be size-classifiable simple locals/parameters." Add a CI check that validates `.note.stapsdt` arg strings.
- **Finding 2 response** of "require sdt 5.0+" is too blunt — Ubuntu 22.04 is still major ROCm dev environment. Better: configure-time validation test that detects malformed output, not version-gating.
- **Finding 3's API-ID swap** is the right direction. Reviewer suggested also considering hybrid (numeric ID authoritative + optional string for human convenience).

## Strongest counter-argument from reviewer

> The most damaging objection is that this design is trying to turn USDT into a low-overhead universal API tracing ABI without first proving the ABI and overhead model. The HSA implementation plan changes every public API control-flow path, introduces correlation bookkeeping on hot calls, depends on fragile sys/sdt.h codegen, and creates a new externally consumed probe contract whose versioning is not defined. If this lands prematurely, the team may be stuck supporting an unstable tracing ABI that is slower than expected, hard to decode reliably, and not actually integrated with ROCprofiler's existing correlation model.

## Recommended sequence before continuing

1. Fix #1 (zero disabled-build cost) and #3 (drop string contract from HSA shim) on this branch.
2. Add the CI inventory test that validates `.note.stapsdt` arg specs (covers Finding 1's CI check + Finding 2's validation step).
3. Define and document the USDT probe ABI versioning policy.
4. Generate the HSA API ID enum as a separate landed change with append-only CI.
5. Replan Task 5 (HIP) to fix the post-increment correlation pairing bug.
6. Then start Tasks 7–9 on the corrected foundation.

Item #2 is the highest-leverage addition — it would have caught all three findings before they bit during execution.

## Strengths reviewer noted

- HIP shim is appropriately small and uses integer IDs already (avoids HSA's string-arg failure)
- Free-function HSA refactor is directionally correct
- Findings doc captures real validation failures honestly; stopping before Tasks 7–17 was the right call
