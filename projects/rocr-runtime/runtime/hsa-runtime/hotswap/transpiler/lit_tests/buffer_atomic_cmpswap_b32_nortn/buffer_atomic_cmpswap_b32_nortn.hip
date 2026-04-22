// ARCH: gfx1250
//
// Non-RTN sibling of `lit_tests/buffer_atomic_cmpswap_b32/`.
// See that fixture's block comment for the full rationale on why
// this coverage matters — the short version is that the handler's
// `op.dst(0)`-keyed vdata-pair read handles both RTN and non-RTN
// forms because operand-0 of a MUBUF buffer-atomic is always the
// vdata register (whether tied-def or pure-source), and the
// write-back is gated by `di.numDefs > 0` which correctly skips
// for non-RTN.
//
// This fixture pins the invariant for CMPSWAP specifically — the
// cmpxchg atomic must still emit (the compare-and-exchange side-
// effect is preserved) but the `extractvalue ..., 0` write-back
// must be skipped.

#include <hip/hip_runtime.h>

extern "C" __global__ void
buffer_atomic_cmpswap_b32_nortn_kernel(unsigned int *out,
                                       unsigned int cmp,
                                       unsigned int newVal) {
  unsigned int srd[4];
  srd[0] = (unsigned int)((uintptr_t)out & 0xFFFFFFFFu);
  srd[1] = (unsigned int)(((uintptr_t)out >> 32) & 0xFFFFFFFFu);
  srd[2] = 0xFFFFFFFFu;
  srd[3] = 0x00027000u;

  unsigned int voff = (unsigned int)threadIdx.x * 4u;

  // {cmp, new} pair still allocated as consecutive VGPRs via the
  // `v2u32` vector constraint — the RTN vs non-RTN distinction is
  // only about the GLC bit, the vdata layout is identical.  Omit
  // `th:TH_ATOMIC_RETURN` to force the glc=0 form; use a pure-
  // input `v` constraint (no `+v`) since we don't read the result.
  typedef unsigned int s4u32 __attribute__((ext_vector_type(4)));
  s4u32 srdReg = {srd[0], srd[1], srd[2], srd[3]};
  typedef unsigned int v2u32 __attribute__((ext_vector_type(2)));
  v2u32 cmpNew = {cmp, newVal};
  asm volatile("buffer_atomic_cmpswap_b32 %[d], %[v], %[s], null offen "
               "scope:SCOPE_DEV\n\t"
               "s_wait_loadcnt 0\n\t"
               :
               : [d] "v"(cmpNew), [v] "v"(voff), [s] "s"(srdReg)
               : "memory");
}
