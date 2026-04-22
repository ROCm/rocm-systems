// ARCH: gfx1250
//
// Non-RTN sibling of `lit_tests/buffer_atomic_swap_b32/`.  This
// fixture proves that the MUBUF-atomic handler in
// `handle_mubuf.cpp` correctly handles the `glc=0` / non-RTN form
// of `buffer_atomic_swap_b32` — the atomic happens, the vdata is a
// pure-source operand (not a tied-def), and the handler's
// `di.numDefs > 0` guard correctly SKIPS the write-back path.
//
// Why a dedicated fixture: the `/principled` audit of the RTN
// sibling fixture flagged a concern that the handler's
// `op.dst(0)`-keyed data read might crash on non-RTN forms (no
// tied destination).  Turns out the concern was misplaced:
// `op.dst(i)` in `raise_context.hpp:489` is
// `ctx.parseReg(di.getReg(i), i)` — it reads operand at absolute
// position `i`, which for MUBUF buffer atomics is ALWAYS the vdata
// register (operand 0), whether the instruction is tied-def RTN
// (glc=1) or source-only non-RTN (glc=0).  So the same code path
// handles both; the only difference is the write-back, which is
// guarded by `di.numDefs > 0` and correctly skips for non-RTN.
//
// This fixture PINS the no-write-back invariant: a handler change
// that accidentally writes the atomicrmw result back to a
// non-existent destination (e.g. by dropping the `numDefs` guard)
// would fail lit here before surprising a user.

#include <hip/hip_runtime.h>

extern "C" __global__ void
buffer_atomic_swap_b32_nortn_kernel(unsigned int *out, unsigned int newVal) {
  // Raw-buffer SRD construction — same shape as the RTN sibling.
  unsigned int srd[4];
  srd[0] = (unsigned int)((uintptr_t)out & 0xFFFFFFFFu);
  srd[1] = (unsigned int)(((uintptr_t)out >> 32) & 0xFFFFFFFFu);
  srd[2] = 0xFFFFFFFFu;
  srd[3] = 0x00027000u;

  unsigned int voff = (unsigned int)threadIdx.x * 4u;

  // NON-RTN form: omit `th:TH_ATOMIC_RETURN` AND use a `v` input-
  // only constraint on the data register.  The absence of
  // `TH_ATOMIC_RETURN` tells the gfx12 encoder to emit the glc=0
  // form (no returned old value); the pure-input constraint
  // matches the hardware's "pure-source vdata" shape for non-RTN.
  typedef unsigned int s4u32 __attribute__((ext_vector_type(4)));
  s4u32 srdReg = {srd[0], srd[1], srd[2], srd[3]};
  asm volatile("buffer_atomic_swap_b32 %[d], %[v], %[s], null offen "
               "scope:SCOPE_DEV\n\t"
               "s_wait_loadcnt 0\n\t"
               :
               : [d] "v"(newVal), [v] "v"(voff), [s] "s"(srdReg)
               : "memory");
}
