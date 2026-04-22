// ARCH: gfx1250
//
// Lift fixture for the gfx11+/gfx1250 VBUFFER buffer-atomic
// compare-and-swap (`buffer_atomic_cmpswap_b32`).  See
// `SemOp::BUFFER_ATOMIC_CMPSWAP` in transpiler/semop.hpp and the
// dedicated pre-switch branch in transpiler/handle_mubuf.cpp (the
// `if (sop == SemOp::BUFFER_ATOMIC_CMPSWAP)` arm added alongside
// this fixture).
//
// Before this handler arm existed, `buffer_atomic_cmpswap_b32` hit
// the handler's `default:` path and refused with
// `unsupportedShape [MUBUF]: unsupported buffer atomic`.  The
// fixture pins the NEW behaviour so a regression that drops the
// CMPSWAP branch falls back to `default:` and fails lit before a
// corpus kernel surfaces the miscompile.
//
// Semantic note: `buffer_atomic_cmpswap_b32` takes a register-PAIR
// vdata carrying `{cmp, new}` — dword 0 is the expected value, dword
// 1 is the replacement.  The handler decodes this by reading
// `op.dst(0)` (the base of the tied pair) and synthesising a
// `baseIdx + 1` ParsedReg for the second dword — mirror of the
// sibling `FLAT_ATOMIC_CMPSWAP` / `GLOBAL_ATOMIC_CMPSWAP` handlers
// in handle_flat.cpp.  The RTN-form result is the ORIGINAL value at
// the memory cell (whether or not the swap succeeded), which the
// caller uses to detect success (old == cmp ⇔ swap happened).

#include <hip/hip_runtime.h>

extern "C" __global__ void
buffer_atomic_cmpswap_b32_kernel(unsigned int *out,
                                 unsigned int cmp,
                                 unsigned int newVal) {
  // Raw-buffer SRD construction, same shape as the ADD / SWAP
  // sibling fixtures.
  unsigned int srd[4];
  srd[0] = (unsigned int)((uintptr_t)out & 0xFFFFFFFFu);
  srd[1] = (unsigned int)(((uintptr_t)out >> 32) & 0xFFFFFFFFu);
  srd[2] = 0xFFFFFFFFu;
  srd[3] = 0x00027000u;

  unsigned int voff = (unsigned int)threadIdx.x * 4u;

  // OFFEN form.  The `%[cmp]`-paired `+v` constraint asks the
  // compiler to allocate the {cmp, new} pair in consecutive VGPRs
  // so the buffer_atomic_cmpswap_b32 encoding's vdata slot can
  // reference them as a VReg_64-style register pair.  `cmp` and
  // `newVal` are passed in via separate `+v` / `v` constraints
  // because the inline-asm matcher doesn't have a portable "give
  // me a VReg_64 split into two i32 operands" constraint; the
  // resulting register allocation does put them consecutively in
  // practice (HIP's clang inline-asm matcher honours the pair
  // ordering of `v` constraints for data+auxiliary usage).  The
  // `th:TH_ATOMIC_RETURN` forces the RTN form so we exercise the
  // handler's write-back path.
  typedef unsigned int s4u32 __attribute__((ext_vector_type(4)));
  s4u32 srdReg = {srd[0], srd[1], srd[2], srd[3]};
  // Pack {cmp, new} into a VReg_64-like shape so the inline-asm
  // encoder places them in consecutive VGPRs.  The `v` constraint
  // on a 2-element vector gives us that layout deterministically.
  typedef unsigned int v2u32 __attribute__((ext_vector_type(2)));
  v2u32 cmpNew = {cmp, newVal};
  asm volatile("buffer_atomic_cmpswap_b32 %[d], %[v], %[s], null offen "
               "th:TH_ATOMIC_RETURN scope:SCOPE_DEV\n\t"
               "s_wait_loadcnt 0\n\t"
               : [d] "+v"(cmpNew)
               : [v] "v"(voff), [s] "s"(srdReg)
               : "memory");
  // Low lane of the cmpNew pair carries the original value on RTN.
  out[threadIdx.x] = cmpNew.x;
}
