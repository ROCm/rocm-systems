// ARCH: gfx1250
//
// Lit fixture for SOP1 s_set_pc_i64 — Unresolvable refusal path.
// See SemOp::S_SET_PC_I64 in semop.hpp and the analysis in
// setpc_analysis.cpp. The S_SET_PC_I64 handler in handle_sop1.cpp
// is contractually obligated to refuse — via
// RaiseFailure::unsupportedShape — any site whose source SGPR pair
// the static analysis cannot trace back to either:
//
//   (A) a complete s_get_pc_i64 + s_add_co_u32 + s_add_co_ci_u32
//       chain in the same basic block (Pattern A), or
//   (B) a chain-terminator at some prior call site whose ret-pair
//       low index matches this read (Pattern B).
//
// This fixture pins the negative path: we deliberately strip any
// usable PC chain by writing the SGPR pair via s_mov_b32 (a generic
// SGPR write) before issuing s_set_pc_i64. The analysis pass'
// generic-write fallthrough invalidates any tracked chain at that
// pair, leaving the s_set_pc_i64 with no resolvable provenance.
// Pass 2 then has no chainTerminator entry for this pair (no other
// site in the kernel sets one up), so the site is recorded as
// Unresolvable and the handler refuses.
//
// This is a NEGATIVE test: %not inverts the exit code, so the test
// passes only when raise_cli actually fails. We FileCheck the
// raise_cli stderr line for the mnemonic + format pair so a future
// regression that silently emits a stub `br` (or worse, an
// `unreachable`) would flip both the exit code and the diagnostic.
//
// MAINTENANCE: this fixture intentionally produces a `s_set_pc_i64`
// to garbage. The kernel must NEVER be dispatched — only the raiser
// pre-translation refusal is exercised. If this file ever becomes
// part of a runtime test, change the construction to use kernarg
// loads (s_load_b64) instead, so a stray dispatch at least faults
// at a deterministic address rather than executing wild code.

#include <hip/hip_runtime.h>

extern "C" __global__ void
setpc_unresolvable_kernel(int *out) {
  int tid = (int)threadIdx.x;
  asm volatile(
      // Strip any PC chain provenance by hand-loading the SGPR pair
      // with a generic s_mov_b32. The analysis pass classifies this
      // as a generic SGPR write that invalidates any tracked chain
      // (and never establishes one for s[10:11] in the first place).
      "s_mov_b32 s10, 0\n\t"
      "s_mov_b32 s11, 0\n\t"
      // Indirect set-PC reading the unresolvable pair. The handler
      // must refuse with RaiseFailure::unsupportedShape; the analysis
      // refusalReason should mention either "chain" or the specific
      // pair "s[10:11]".
      "s_set_pc_i64 s[10:11]\n\t"
      :
      :
      : "s10", "s11");
  out[tid] = tid;
}
