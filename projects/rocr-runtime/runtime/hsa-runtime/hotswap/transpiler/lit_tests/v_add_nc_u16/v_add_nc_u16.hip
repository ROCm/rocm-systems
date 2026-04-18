// ARCH: gfx1250
//
// Lift fixture for VOP3 v_add_nc_u16 (gfx10+ 16-bit no-carry
// add). See the V_ADD_NC_U16 SemOp in transpiler/semop.hpp and
// the matching handler block in transpiler/handle_valu.cpp; this
// fixture pins the IR shape both reference (handler ↔ test
// back-reference is non-negotiable per the transpiler test
// policy).
//
// We exercise the default op_sel:[0,0,0] path which dominates the
// observed corpus — the handler still emits the explicit
// read-mask-OR merge so the dst-half preservation semantics are
// observable in the IR shape (and therefore pinnable). A
// dedicated op_sel:[1,0,0] / [0,0,1] / etc. fixture would
// require hand-built inline asm; until the corpus shows such
// shapes we leave that to a future test (the handler already
// handles all 8 op_sel combinations symmetrically).
//
// `unsigned short` source operands ensure the compiler picks the
// 16-bit add SemOp; without the explicit narrow type, integer
// promotion would route through V_ADD_NC_U32. Loading from a
// pointer keeps the compiler from constant-folding past the asm
// barrier the kernel-arg edge would otherwise allow.

#include <hip/hip_runtime.h>

extern "C" __global__ void v_add_nc_u16_kernel(unsigned short *out,
                                               const unsigned short *in) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned short a = in[2 * tid + 0];
  unsigned short b = in[2 * tid + 1];
  unsigned short r;
  asm volatile("v_add_nc_u16 %[r], %[a], %[b]\n\t"
               : [r] "=v"(r)
               : [a] "v"(a), [b] "v"(b));
  out[tid] = r;
}
