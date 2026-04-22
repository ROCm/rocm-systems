// ARCH: gfx1250
//
// Lift fixture for the `global_store_short_d16_hi` /
// `global_store_d16_hi_b16` half-register store.  Companion to the
// existing `ds_store_b16_d16_hi` fixture — same "upper 16 bits of
// the source VGPR" semantics, different address space
// (addrspace(1) here, addrspace(3) for the DS sibling).  See
// `SemOp::GLOBAL_STORE_SHORT_D16_HI` / `SemOp::FLAT_STORE_SHORT_D16_HI`
// in `transpiler/semop.hpp`, the handler blocks in
// `transpiler/handle_flat.cpp` under
// `if (sop == SemOp::GLOBAL_STORE_BYTE || ... GLOBAL_STORE_SHORT_D16_HI
// || ...)` and
// `if (sop == SemOp::FLAT_STORE_DWORD || ... FLAT_STORE_SHORT_D16_HI)`,
// and the opcode mappings in `transpiler/opcode_map.cpp`.
//
// DOMINANT CORPUS PATTERN — bf16 RNE-cast epilogue.  The AMDGPU
// compiler emits `v_add3_u32 v, fp32_bits, odd_bit, 0x7fff` (the
// round-to-nearest-even biased sum in the upper 16 bits of `v`)
// followed by `global_store_d16_hi_b16 v_addr, v, off` to write the
// bf16 result without an explicit `v_lshrrev_b32 v, 16, v` shift.
// Pre-fix, the raiser lifted `GLOBAL_STORE_SHORT_D16_HI` identically
// to plain `GLOBAL_STORE_SHORT` via a `trunc i32 to i16` on the
// source VGPR — silently writing bits [15:0] of the biased sum
// instead of bits [31:16].  The typical-input low-16 pattern is
// 0x7FFF (from `<any-non-overflowing-value> + 0x7FFF`), so every
// bf16-cast-store kernel wrote a field of NaNs — exactly the
// pattern `topk_forward_bisect_m_laneprobe` pins at the
// compare_correctness level.  A regression that re-introduces the
// no-shift form would surface here as a missing `lshr 16` before
// the `trunc i32 to i16`.
//
// We force emission of `global_store_d16_hi_b16` via inline asm.

#include <hip/hip_runtime.h>

extern "C" __global__ void
global_store_short_d16_hi_kernel(unsigned short *out, unsigned int packed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned short *slot = out + tid;
  asm volatile("global_store_d16_hi_b16 %[a], %[v], off\n\t"
               :
               : [a] "v"(slot), [v] "v"(packed)
               : "memory");
}
