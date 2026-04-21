// ARCH: gfx950
//
// Lift fixture for the gfx8+ scalar-cache atomic wrap-decrement
// (`s_atomic_dec sDST, sBASE, sOFFSET`).  See SemOp::S_ATOMIC_DEC in
// transpiler/semop.hpp, the three
// `E(S_ATOMIC_DEC_{IMM,SGPR,SGPR_IMM}, S_ATOMIC_DEC)` entries in
// transpiler/opcode_map.cpp, and the shared
// {SemOp::S_ATOMIC_SWAP, SemOp::S_ATOMIC_DEC} block at the bottom of
// transpiler/handle_smem.cpp (AtomicRMWInst::UDecWrap dispatch).
//
// We compile for gfx950 — the source ISA of the AITER corpus this
// handler unblocks — and the companion .ll raises under
// `--isa=gfx950 --target-isa=gfx942`, matching the canonical race
// pipeline (gfx950 workload -> gfx942 box) that BatchRaise.AiterGfx950
// now exercises end-to-end.
//
// Producer shape: the AITER split-k reduction barrier (every
// `bf16gemm_*_splitk_clean.co`) emits exactly this form —
//   s_atomic_dec s52, s[40:41], s62
// — on the "last workgroup runs the epilogue" gate, keyed on the
// returned pre-decrement value being 1.  HIP has no language-level
// spelling that lowers to the scalar-cache atomic (the C-level
// `atomicDec` goes through VMEM's `global_atomic_dec_u32`), so the
// fixture issues the instruction directly via inline asm bound to
// `"+s"` / `"s"` operand constraints:
//   * `"+s"(rmw)` — tied read-modify-write on the sdst slot: the
//     input value is the wrap threshold (what the HW compares `old`
//     against), the output value is the returned pre-decrement `old`.
//     This is the disassembler's (sdst, sbase, soffset) shape where
//     TableGen declares `$sdst_in` tied (see decode.cpp's
//     `kKnownTiedIn` comment).
//   * `"s"(counter)` — 64-bit pointer -> SGPR pair base.
//   * `"s"(offset)`  — 32-bit SGPR offset (the `_SGPR` addressing
//     form; matches the AITER producer above verbatim).
//
// The `s_waitcnt lgkmcnt(0)` after the atomic is the canonical
// scalar-cache completion fence; we include it for semantic
// completeness but do not pin it in the .ll check (the lift drops
// waitcnt tokens on the floor — they're codegen-side, not lift-side).

#include <hip/hip_runtime.h>

extern "C" __global__ void
s_atomic_dec_kernel(unsigned int *counter, unsigned int wrap_threshold,
                    unsigned int offset, unsigned int *is_last_out) {
  // One workgroup, one lane performs the atomic.  Keeps the inline-asm
  // layout simple — no cross-block synchronisation is implied — and
  // gives the "am I the last workgroup?" idiom a single deterministic
  // observer for the split-k barrier semantics we're pinning.
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;

  unsigned int rmw = wrap_threshold;
  asm volatile(
      "s_atomic_dec %[rmw], %[ptr], %[off]\n\t"
      "s_waitcnt lgkmcnt(0)\n\t"
      : [rmw] "+s"(rmw)
      : [ptr] "s"(counter), [off] "s"(offset)
      : "memory");

  // Split-k barrier idiom: the epilogue runs iff the pre-decrement
  // value was exactly 1 (we were the last workgroup).  Any rewrite of
  // the atomic that loses the returned-old-value write-back would
  // silently miscompile this branch.
  *is_last_out = (rmw == 1u) ? 1u : 0u;
}
