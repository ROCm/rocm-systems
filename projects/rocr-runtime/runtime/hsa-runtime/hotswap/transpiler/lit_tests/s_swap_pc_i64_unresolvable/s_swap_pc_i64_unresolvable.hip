// ARCH: gfx1250
//
// Lit fixture for SOP1 s_swap_pc_i64 — Unresolvable refusal path.
// See SemOp::S_SWAP_PC_I64 in semop.hpp and the swap branch in
// transpiler/setpc_analysis.cpp.  The S_SWAP_PC_I64 handler in
// transpiler/handle_sop1.cpp is contractually obligated to refuse
// — via RaiseFailure::unsupportedShape — any swap site whose source
// SGPR pair (the call target) the static analysis cannot trace
// back to a complete intra-block PC chain. The user-visible failure
// pin is required because the tensilelite corpus pattern (the
// dominant real-world s_swap_pc_i64 caller) constructs the call
// target via `s_get_pc_i64 + s_add_co_u32 sX, sX, sZ` where sZ is a
// runtime scalar derived from a kernarg — i.e. a true
// dynamic-dispatch call we cannot lower without enumerating
// candidate callees. A silent stub `br` (or, worse, an
// `unreachable`) at such a site would miscompile the kernel; the
// refusal contract fails loudly instead.
//
// This fixture pins the negative path: we strip any usable PC
// chain by writing the source SGPR pair via s_mov_b32 (a generic
// SGPR write that the analysis pass invalidates) before issuing
// s_swap_pc_i64. With no resolvable provenance for s[10:11], the
// site is recorded as Unresolvable and the handler refuses.
//
// This is a NEGATIVE test: %not inverts the exit code, so the test
// passes only when raise_cli actually fails. We FileCheck the
// raise_cli stderr for the mnemonic + format pair so a regression
// that silently emits a stub `br` would flip both the exit code
// and the diagnostic.
//
// MAINTENANCE: this fixture intentionally produces an
// `s_swap_pc_i64` to garbage. The kernel must NEVER be dispatched
// — only the raiser pre-translation refusal is exercised.

#include <hip/hip_runtime.h>

extern "C" __global__ void
setpc_swap_unresolvable_kernel(int *out) {
  int tid = (int)threadIdx.x;
  asm volatile(
      // Strip any PC chain provenance by hand-loading the source
      // SGPR pair with a generic s_mov_b32. The analysis pass
      // classifies these as generic SGPR writes and never establishes
      // a chain for s[10:11].
      "s_mov_b32 s10, 0\n\t"
      "s_mov_b32 s11, 0\n\t"
      // Branch-and-link reading the unresolvable pair as the call
      // target. The handler must refuse with
      // RaiseFailure::unsupportedShape; the analysis refusalReason
      // names the source pair s[10:11] explicitly.
      "s_swap_pc_i64 s[20:21], s[10:11]\n\t"
      :
      :
      : "s10", "s11", "s20", "s21");
  out[tid] = tid;
}
