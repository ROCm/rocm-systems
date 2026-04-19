// ARCH: gfx1250
//
// Lit fixture for the gfx9+/gfx1250 D16 byte buffer loads
// (`buffer_load_d16_u8` / `buffer_load_d16_hi_u8`).  See
// SemOp::BUFFER_LOAD_UBYTE_D16{,_HI} in transpiler/semop.hpp; the
// MUBUF / VBUF entries in transpiler/opcode_map.cpp; and the
// d16Half merge path in transpiler/handle_mubuf.cpp.
//
// Both `_d16_*` variants are partial-write loads: the loaded byte is
// zero-extended (for u8) or sign-extended (for i8) to i16 and merged
// into the lo half (`_D16`) or hi half (`_D16_HI`) of the destination
// VGPR, preserving the OTHER 16 bits.  This is the
// `D16PreservesUnusedBits` predicate from BUFInstructions.td:1155-1177.
//
// We pin both halves in one fixture because the merge logic is
// symmetric (mask + or vs. shl + or); covering one variant per half
// exercises the entire d16Half branch of mubufClassify.  We use u8
// (zero-extend) rather than i8 (sign-extend) because that's the
// variant the tensilelite gfx1250 corpus actually issues
// (24 kernels in kerneldex's worklist refused on `buffer_load_d16_u8`
// before this handler landed).
//
// Inline asm is required: HIP's natural lowering for `*(uint8_t*)p`
// goes through `global_load_u8` (FLAT addr space), not through the
// MUBUF/VBUFFER buffer-descriptor path.  The D16 variants don't
// surface in HIP source at all; they're produced by Triton's MLIR
// AMDGPU lowering when packing two adjacent i8 lanes into one VGPR.

#include <hip/hip_runtime.h>

extern "C" __global__ void
buffer_load_d16_u8_kernel(unsigned int *out, unsigned int *src) {
  // Build a 4-dword raw-buffer SRD pointing at `src` so the buffer
  // load targets observable global memory (matches the corpus
  // pattern; otherwise the verification below would be vacuous).
  unsigned int srd[4];
  srd[0] = (unsigned int)((uintptr_t)src & 0xFFFFFFFFu);
  srd[1] = (unsigned int)(((uintptr_t)src >> 32) & 0xFFFFFFFFu);
  srd[2] = 0xFFFFFFFFu;
  srd[3] = 0x00027000u;
  typedef unsigned int s4u32 __attribute__((ext_vector_type(4)));
  s4u32 srdReg = {srd[0], srd[1], srd[2], srd[3]};

  unsigned int voff = (unsigned int)threadIdx.x * 4u;

  // Pre-load the destination VGPR with a known sentinel so the
  // partial-write merge is observable.  `+v` constraint binds the
  // SAME VGPR for input and output; the d16 load below names this
  // VGPR as both the destination and (implicitly, via tied-def) the
  // prior value to preserve the unused half of.
  //
  // Sentinel chosen so both halves are individually distinguishable:
  //   lo half = 0xFEED  (wins after `_d16_u8`     merges newly-
  //                      loaded byte into lo, preserving hi 0xCAFE)
  //   hi half = 0xCAFE  (wins after `_d16_hi_u8` merges newly-
  //                      loaded byte into hi, preserving lo 0xFEED)
  unsigned int lo = 0xCAFEFEEDu;  // [hi=CAFE, lo=FEED]
  asm volatile(
      // _D16: load byte, zero-extend to i16, merge into LO half
      // of %[d], preserving HI (= 0xCAFE).
      "buffer_load_d16_u8 %[d], %[v], %[s], null offen scope:SCOPE_DEV\n\t"
      "s_wait_loadcnt 0\n\t"
      : [d] "+v"(lo)
      : [v] "v"(voff), [s] "s"(srdReg)
      : "memory");

  unsigned int hi = 0xCAFEFEEDu;
  asm volatile(
      // _D16_HI: load byte, zero-extend to i16, merge into HI half
      // of %[d], preserving LO (= 0xFEED).
      "buffer_load_d16_hi_u8 %[d], %[v], %[s], null offen scope:SCOPE_DEV\n\t"
      "s_wait_loadcnt 0\n\t"
      : [d] "+v"(hi)
      : [v] "v"(voff), [s] "s"(srdReg)
      : "memory");

  out[threadIdx.x * 2 + 0] = lo;
  out[threadIdx.x * 2 + 1] = hi;
}
