// ARCH: gfx1250
//
// Regression guard for the principled BUFFER_STORE_* lowering in
// transpiler/handle_mubuf.cpp.  See SemOp::BUFFER_STORE_DWORDX4 in
// transpiler/semop.hpp; the `if (isStore)` block in
// transpiler/handle_mubuf.cpp; and the design discussion in
// hotswap/docs/buffer-store-lowering.md.
//
// What this fixture pins (the .ll RUN-line FileCheck does the actual
// asserts; this is just the source of the lifted IR):
//
//   1. The BUFFER_STORE_DWORDX4 (gfx12 spelling: `buffer_store_b128`)
//      lifts to a `call void @llvm.amdgcn.raw.buffer.store.v4i32(...)`
//      intrinsic — symmetric with the existing load path which uses
//      `@llvm.amdgcn.raw.buffer.load.<ty>`.
//
//   2. The lifted IR contains ZERO `addrspace(5)` allocas.  This is
//      the actual ABI invariant that fixed Triton vector-add (R1):
//      any `addrspace(5)` alloca surviving PromoteMemToReg makes the
//      AMDGPU backend emit `.amdhsa_enable_private_segment 1` plus
//      `.amdhsa_private_segment_fixed_size > 0`.  Salmon's KD does
//      not request `flat_scratch_init` (we model only the source
//      ABI's user-SGPR set), so on entry FLAT_SCRATCH is undefined;
//      any flat instruction touching the scratch aperture (including
//      the legacy OOB sink store path) is a fault waiting to happen.
//
//   3. NO `addrspacecast ... to ptr addrspace(5)` and NO
//      `select i1 ... ptr addrspace(5)`-style sink shape — the legacy
//      lowering routed every BUFFER_STORE through a software
//      OOB-select against an `i32` sink alloca, which was wrong on
//      three independent axes (size mismatch for >4B stores, forced
//      scratch enablement, and asymmetry with the load path).
//
//   4. NO `@llvm.amdgcn.flat.store` and NO plain `store ... addrspace(5)`
//      — we want hardware OOB drop via the BUFFER unit, not
//      flat_store-with-software-sink.
//
// We use inline asm to issue `buffer_store_b128` directly because:
//
//   * HIP's natural store lowering goes through global / FLAT (not
//     through MUBUF/VBUFFER), so the BUFFER_STORE_* SemOp would never
//     surface from idiomatic HIP.
//
//   * Inline asm gives us a precise pin on the OFFEN addressing form
//     (voffset-from-vgpr into a buffer rsrc); other addressing modes
//     (OFFSET, IDXEN, BOTHEN) all share the same SemOp + handler
//     branch, so covering OFFEN exercises the BUFFER_STORE_DWORDX4
//     handler entirely.
//
// The kernel body deliberately holds NO local allocations — no
// arrays, no `__shared__`, nothing that would itself introduce an
// `addrspace(5)` alloca for legitimate reasons.  The SRD construction
// is done with a vector-typed scalar (which lives in registers under
// `s` constraint, no stack slot needed); the only memory operands
// are the inline-asm buffer descriptor and the data VGPR quad.

#include <hip/hip_runtime.h>

extern "C" __global__ void
buffer_store_dwordx4_kernel(unsigned int *out) {
  // Build a 4-dword raw-buffer SRD pointing at `out` so the buffer
  // store targets observable global memory.  Layout matches the gfx12
  // raw_buffer_descriptor convention:
  //   dword0: base ptr lo
  //   dword1: base ptr hi
  //   dword2: num records (in bytes for raw)
  //   dword3: flags (stride=0 + format)
  unsigned int srd[4];
  srd[0] = (unsigned int)((uintptr_t)out & 0xFFFFFFFFu);
  srd[1] = (unsigned int)(((uintptr_t)out >> 32) & 0xFFFFFFFFu);
  srd[2] = 0xFFFFFFFFu;
  srd[3] = 0x00027000u;
  typedef unsigned int s4u32 __attribute__((ext_vector_type(4)));
  s4u32 srdReg = {srd[0], srd[1], srd[2], srd[3]};

  // 4-dword payload, distinct dword constants so the lifted IR's
  // `<4 x i32>` value is observably the right shape (not implicit
  // zeros that could be folded into `zeroinitializer`).
  typedef unsigned int v4u32 __attribute__((ext_vector_type(4)));
  v4u32 data = {0xDEADBEEFu, 0xCAFEBABEu, 0xFEEDFACEu, 0xBADC0FFEu};

  unsigned int voff = (unsigned int)threadIdx.x * 16u;

  // gfx12 BUFFER_STORE_DWORDX4 spelling is `buffer_store_b128`.  See
  // the `kStoreOpcodeAliases` table in transpiler/transpiler.cpp for
  // the gfx12 -> gfx9 mnemonic mapping.  `null` for the SOFFSET slot,
  // OFFEN for per-lane voffset, scope:SCOPE_DEV maps to monotonic
  // (no acquire/release) at lift time.
  asm volatile("buffer_store_b128 %[d], %[v], %[s], null offen scope:SCOPE_DEV\n\t"
               "s_wait_storecnt 0\n\t"
               :
               : [d] "v"(data), [v] "v"(voff), [s] "s"(srdReg)
               : "memory");
}
