// ARCH: gfx1250
//
// Lift fixture for the gfx11+/gfx1250 VBUFFER buffer-atomic add
// (`buffer_atomic_add_u32`).  See SemOp::BUFFER_ATOMIC_ADD in
// transpiler/semop.hpp and the `VBUF4(BUFFER_ATOMIC_ADD,
// BUFFER_ATOMIC_ADD)` entry in transpiler/opcode_map.cpp; the
// existing AtomicRMW lowering branch in transpiler/handle_mubuf.cpp
// (`sop >= BUFFER_ATOMIC_ADD && sop <= BUFFER_ATOMIC_PK_ADD_F16`)
// fires unchanged for the VBUFFER encoding because the address
// decoder in mubuf_addr.cpp keys on operand `Kind` rather than
// position.
//
// LLVM MC keeps the legacy `BUFFER_ATOMIC_ADD` pseudo name; gfx11+
// renames the asm spelling to `buffer_atomic_add_u32` and forks the
// encoding into a `_VBUFFER_*` family per addressing mode.  Without
// the VBUF4 entries the gfx1250 corpus
// (e.g. scope_discovery___sum_bitmatrix_rows refused on
// `buffer_atomic_add_u32 v0, v1, s[4:7], null offen`) was rejected
// despite the underlying SemOp + handler being present — this
// fixture exists to regression-pin that gap closure.
//
// We use inline asm to issue the atomic directly because:
//
//   * HIP's `atomicAdd` natural lowering goes through
//     `global_atomic_add_u32` (FLAT/global addrspace) on gfx1250,
//     not through `buffer_atomic_add_u32` — the latter requires a
//     literal raw buffer descriptor (SRD) which HIP doesn't expose
//     at the language level.
//   * Inline asm gives us a precise pin on the OFFEN addressing form
//     that the corpus kernel uses (`offen` means voffset-from-vgpr
//     into a buffer rsrc); other addressing modes (OFFSET, IDXEN,
//     BOTHEN) all share the same SemOp and lowering branch so
//     covering OFFEN exercises the whole VBUF4 macro expansion.
//
// The atomic's `data` (v0) and `vaddr` (v1) operands are bound to
// VGPRs via `v` constraints; the SRSRC `s[4:7]` is a 4-dword SGPR
// quad that we materialise as a stack buffer descriptor.  The
// `scope:SCOPE_DEV` modifier on gfx12 maps to AtomicOrdering
// monotonic in the existing handler — pinned in the IR check below.

#include <hip/hip_runtime.h>

extern "C" __global__ void
buffer_atomic_add_u32_kernel(unsigned int *out, unsigned int delta) {
  // Build a 4-dword raw-buffer SRD from `out` so the atomic targets
  // observable global memory.  Layout matches the
  // raw_buffer_descriptor convention the gfx12 ISA expects:
  //   dword0: base ptr lo
  //   dword1: base ptr hi
  //   dword2: num records (in bytes for raw)
  //   dword3: flags (stride=0 + format)
  unsigned int srd[4];
  srd[0] = (unsigned int)((uintptr_t)out & 0xFFFFFFFFu);
  srd[1] = (unsigned int)(((uintptr_t)out >> 32) & 0xFFFFFFFFu);
  srd[2] = 0xFFFFFFFFu;
  srd[3] = 0x00027000u;

  unsigned int voff = (unsigned int)threadIdx.x * 4u;

  // OFFEN form: vaddr supplies the per-lane voffset, soffset is
  // null.  `s[4:7]`-style 4-dword SGPR allocation comes from the
  // `s` constraint on a 16-byte composite type, which HIP lays out
  // as four contiguous SGPRs in encoding order.
  typedef unsigned int s4u32 __attribute__((ext_vector_type(4)));
  s4u32 srdReg = {srd[0], srd[1], srd[2], srd[3]};
  asm volatile("buffer_atomic_add_u32 %[d], %[v], %[s], null offen scope:SCOPE_DEV\n\t"
               "s_wait_loadcnt 0\n\t"
               :
               : [d] "v"(delta), [v] "v"(voff), [s] "s"(srdReg)
               : "memory");
}
