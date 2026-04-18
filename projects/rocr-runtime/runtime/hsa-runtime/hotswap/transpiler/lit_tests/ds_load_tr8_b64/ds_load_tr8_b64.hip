// ARCH: gfx1250
//
// Lift fixture for the gfx1250 64-bit transposed LDS load with 8-bit
// elements (ds_load_tr8_b64).  See the DS_LOAD_TR8_B64 SemOp in
// transpiler/semop.hpp and the matching shared-emulation block in
// transpiler/handle_ds.cpp; this fixture pins the lowering shape both
// reference (handler ↔ test back-reference is non-negotiable per the
// transpiler test policy).
//
// We use inline asm to issue the load directly because the compiler
// will not naturally emit DS_LOAD_TR8_B64 for an ordinary LDS access:
// the gfx1250 isel pattern for `int_amdgcn_ds_load_tr8_b64` only
// fires inside a v_wmma consumer chain, and the
// `__builtin_amdgcn_ds_load_tr8_b64_v2i32` clang builtin is OpenCL-
// flavoured and does not accept HIP's generic-cast `__shared__`
// pointers without a hand-rolled addrspacecast.
//
// The load is bound to a 64-bit VGPR pair via `=v` on an
// `unsigned long long` output (matching the v_mul_u64 fixture's
// VOP2 64-bit pattern); the result is written to global memory so
// the load is not DCE'd.  `s_wait_dscnt 0` after the load is
// required by the gfx12 memory model before the result is consumed.

#include <hip/hip_runtime.h>

extern "C" __global__ void
ds_load_tr8_b64_kernel(unsigned long long *out) {
  __shared__ unsigned int lds_buf[64];

  int tid = (int)threadIdx.x;
  lds_buf[2 * tid + 0] = 0x10203040u + (unsigned int)tid;
  lds_buf[2 * tid + 1] = 0x50607080u + (unsigned int)tid;
  __syncthreads();

  // Compute the LDS byte offset within lds_buf for this lane.  The
  // shared array starts at LDS offset 0 of the kernel's group-static
  // allocation; each lane's slot is `2 * tid * sizeof(unsigned int)`
  // bytes in.  The low 32 bits of an LDS (addrspace 3) pointer are
  // just that byte offset on AMDGPU, so casting through uintptr_t and
  // truncating to unsigned int gives a valid v-class addr operand
  // for ds_load_tr_b64.
  unsigned int addr = (unsigned int)(uintptr_t)&lds_buf[2 * tid];
  unsigned long long result;
  asm volatile("ds_load_tr_b64 %[r], %[a]\n\t"
               "s_wait_dscnt 0\n\t"
               : [r] "=v"(result)
               : [a] "v"(addr));
  out[tid] = result;
}
