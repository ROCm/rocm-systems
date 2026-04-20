// ARCH: gfx1250
//
// Lift fixture for the gfx12+ `s_load_b32` SMEM dword load with an
// SGPR offset carrying the `scale_offset` (CPol::SCAL) modifier.
// Hardware semantic: the SGPR offset is an ELEMENT index, not a
// byte offset — hardware multiplies it by the load's data-type size
// (4 bytes for b32) before adding to the SGPR-pair base.  See the
// S_LOAD_B32 SemOp in transpiler/semop.hpp and the SGPR-offset
// branch of `handleSMEM` in transpiler/handle_smem.cpp; this
// fixture pins the IR shape both reference.
//
// Why this fixture matters: the SGPR-offset path of `handle_smem.cpp`
// was silently ignoring `di.hasScaleOffset`, treating every SGPR
// offset as raw bytes.  On gfx12 kernels that use the
// `scale_offset` modifier for a per-block index (the idiomatic
// `mask[blockIdx.x]` pattern produced by Triton and by HIPCC for
// `s_and_saveexec_b32` variants), this produced an off-by-N* byte
// offset everywhere `blockIdx.x != 0`.  The bug surfaced as a
// correctness miscompile that only triggered on multi-block
// launches.  This fixture regression-pins the hardware-matching
// scale-by-4 multiplication so the bug cannot return.
//
// The FLAT/GLOBAL sibling paths (`flat_addr.cpp`) were already
// honouring `hasScaleOffset` — see the `di.hasScaleOffset` checks
// in `decodeGlobalLoadAddr` / `decodeGlobalStoreAddr`.  The SMEM
// bug was isolated to the scalar-offset arm of the scalar memory
// handler.
//
// === Producer pattern (inline asm) ===
//
// Inline asm is used rather than a C-level index expression because
// HIP/AMDGPU frontend consistently folds a per-block index into the
// immediate-offset slot of the SMEM load rather than keeping it in
// an SGPR with `scale_offset`.  Only inline asm reliably pins the
// SGPR-offset + SCAL encoding.
//
// The `"s"` constraint on the offset forces the per-block index to
// be materialised as an SGPR (typically via
// `__builtin_amdgcn_readfirstlane` upstream), matching the shape
// every real corpus producer issues.

#include <hip/hip_runtime.h>

extern "C" __global__ void s_load_b32_scale_offset_kernel(
    int *__restrict__ out,
    const unsigned int *__restrict__ mask_in,
    unsigned int block_index) {
  unsigned int sBlockIndex = __builtin_amdgcn_readfirstlane(block_index);
  int v;
  asm volatile(
      "s_load_b32 %[dst], %[base], %[off] offset:0x0 scale_offset\n\t"
      "s_wait_kmcnt 0\n\t"
      : [dst] "=s"(v)
      : [base] "s"(mask_in), [off] "s"(sBlockIndex));
  out[threadIdx.x] = v;
}
