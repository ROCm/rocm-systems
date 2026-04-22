// ARCH: gfx1250
//
// Lift fixture for the gfx11+/gfx1250 VBUFFER buffer-atomic swap
// (`buffer_atomic_swap_b32`).  Sibling of `buffer_atomic_add_u32/`;
// the fixture shape mirrors it exactly except for the opcode and
// the IR-side `atomicrmw add` → `atomicrmw xchg` change.  See
// `SemOp::BUFFER_ATOMIC_SWAP` in transpiler/semop.hpp and the
// matching handler arm in transpiler/handle_mubuf.cpp (the
// `case SemOp::BUFFER_ATOMIC_SWAP: atomicOp = AtomicRMWInst::Xchg;`
// row added alongside this fixture).
//
// Before this handler arm existed, `buffer_atomic_swap_b32` hit the
// handler's `default:` path and refused with
// `unsupportedShape [MUBUF]: unsupported buffer atomic`.  The fixture
// pins the NEW behaviour so a regression that accidentally drops the
// SWAP case falls back to the default arm and fails lit here before
// a corpus kernel surfaces the miscompile.
//
// No current Triton corpus kernel issues `buffer_atomic_swap_b32`
// (the handler gap was flagged in the opcode-map-audit exercise that
// also removed the `S_ADD_U64` duplicate); this fixture is the
// defensive regression target for when one does.  The companion
// `buffer_atomic_cmpswap_b32/` fixture pins the CMPSWAP half.

#include <hip/hip_runtime.h>

extern "C" __global__ void
buffer_atomic_swap_b32_kernel(unsigned int *out, unsigned int newVal) {
  // Raw-buffer SRD construction, same shape as the ADD sibling
  // fixture — layout matches the gfx12 raw-buffer descriptor ABI.
  unsigned int srd[4];
  srd[0] = (unsigned int)((uintptr_t)out & 0xFFFFFFFFu);
  srd[1] = (unsigned int)(((uintptr_t)out >> 32) & 0xFFFFFFFFu);
  srd[2] = 0xFFFFFFFFu;
  srd[3] = 0x00027000u;

  unsigned int voff = (unsigned int)threadIdx.x * 4u;

  // OFFEN form.  `v0`-style data register is bound via the `v`
  // constraint; the SRSRC 4-dword SGPR tuple is bound via the `s`
  // constraint on a 16-byte composite.  The `scope:SCOPE_DEV`
  // modifier matches what the existing ADD lift models.  The
  // `th:TH_ATOMIC_RETURN` (glc=1 on earlier ISAs) forces the RTN
  // form so we exercise the write-back path in the handler — the
  // old value of the memory cell ends up back in `data` via the
  // tied-def shape.
  typedef unsigned int s4u32 __attribute__((ext_vector_type(4)));
  s4u32 srdReg = {srd[0], srd[1], srd[2], srd[3]};
  unsigned int oldVal = newVal;
  asm volatile("buffer_atomic_swap_b32 %[d], %[v], %[s], null offen "
               "th:TH_ATOMIC_RETURN scope:SCOPE_DEV\n\t"
               "s_wait_loadcnt 0\n\t"
               : [d] "+v"(oldVal)
               : [v] "v"(voff), [s] "s"(srdReg)
               : "memory");
  out[threadIdx.x] = oldVal;
}
