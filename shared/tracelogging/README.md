# shared/tracelogging

Vendored Microsoft TraceLogging (LTTng flavor) — the alternate backend for the
curated ROCm HIP/HSA tracepoints. This mirrors `shared/lttng/`: a small vendored
dependency plus a CMake module and codegen scripts, shared by both consumers
(rocr-runtime, clr/hipamd).

## What this is

The curated tracepoint pipeline (`shared/lttng/scripts/`) turns
`curated_apis.yaml` into per-API emit helpers. This directory provides a
**backend swap**: instead of emitting `LTTNG_UST_TRACEPOINT_EVENT` macros +
`lttng_ust_do_tracepoint()`, the codegen here
(`scripts/tracelogging_curated_codegen.py`) emits `TraceLoggingWrite()` calls
against a `TRACELOGGING_DEFINE_PROVIDER` provider.

It is gated by the SAME CMake flag as classic LTTng-UST
(`HSA_ENABLE_LTTNG_UST` / `HIP_ENABLE_LTTNG_UST`) — it is not a third
coexisting option. The runtime is built with EITHER the classic macro backend
OR the TraceLogging backend, chosen at codegen time (which generated
`.cpp`/`.h` is checked in / configured).

## Contents

- `tracelogging-src/` — vendored subtree of
  [microsoft/tracelogging](https://github.com/microsoft/tracelogging)
  `LTTng/`. See `tracelogging-src/VENDORED_COMMIT.txt` for the pinned commit
  and file set. MIT licensed (`tracelogging-src/LICENSE`).
- `cmake/VendorTracelogging.cmake` — builds `lttngh` (the TraceLogging LTTng
  helper) as a static lib and exposes the `rocm_tracelogging` target.
- `scripts/tracelogging_curated_codegen.py` — codegen fork that reuses
  `shared/lttng/scripts/lttng_curated_lib.py` + `lttng_curated_verify.py`
  (the libclang real-header type resolver) and emits TraceLoggingWrite bodies.

## Provider names

The TraceLogging providers use DISTINCT names from the classic LTTng providers
(`rocm_hsa_tlg` / `rocm_hip_tlg` vs `rocm_hsa` / `rocm_hip`). LTTng-UST allows
only one registered provider per process with a given name; the classic
`rocm_hsa` / `rocm_hip` providers (hand-written non-curated events like
`hsa_doorbell_ring`) keep their names, and the TraceLogging curated provider
coexists under the `_tlg` name.

## Registration lifecycle

`TRACELOGGING_DEFINE_PROVIDER` creates the provider in the *unregistered*
state; `TraceLoggingWrite` on an unregistered provider is a no-op. The provider
must be explicitly registered/unregistered — this is not automatic, unlike the
classic provider's constructor-based static-init.

The provider symbol is a linker-section token, so the `TRACELOGGING_DEFINE_PROVIDER`,
every `TraceLoggingWrite`, and the register/unregister wrappers all live in the
generated `rocm_trace_emit_curated.cpp` (a single DSO-local TU set). The
register/unregister wrappers are exported as `extern "C"` and called from each
runtime's real init/shutdown path:

- **HSA** supports repeated `hsa_init()`/`hsa_shut_down()`. Register from
  `Runtime::LoadTools()` (first Acquire) and unregister from
  `Runtime::UnloadTools()` (last Release). A refcounted, mutex-guarded state
  machine in the emit `.cpp` makes repeated cycles safe: registration happens
  once per 0→1 transition, unregistration once per 1→0, never overlapping,
  never double-registering.
- **HIP** init is one-shot (`std::call_once`). The provider is registered from
  the existing library-constructor path once and unregistered from a
  destructor; the same refcounted wrapper is used, so HIP simply never exceeds
  a refcount of 1.
