// ARCH: gfx1250
//
// Lift fixture for the gfx1250 96-bit LDS load/store pair
// (`ds_load_b96` / `ds_store_b96`).  See SemOp::DS_READ_B96 and
// SemOp::DS_WRITE_B96 in transpiler/semop.hpp and the
// `{dwords=3, loadBits=96}` entry in `dsClassify` plus the generic
// `if (isDsRead || isDsWrite)` path in transpiler/handle_ds.cpp;
// this fixture pins the lowering shape both reference (handler ↔
// test back-reference is non-negotiable per the transpiler test
// policy).
//
// LLVM MC keeps the legacy `DS_READ_B96` / `DS_WRITE_B96` pseudo
// names for what gfx11+ disassembles as `ds_load_b96` /
// `ds_store_b96` (DSInstructions.td:1576-1578); we canonicalise on
// the gfx11+ asm spelling, mirroring the s_set_pc_i64 /
// DS_LOAD_TR8_B64 precedent.
//
// We use inline asm to issue the load and store directly because:
//
//   * The HIP compiler's natural lowering of `int3` / `__attribute__
//     ((vector_size(12)))` into the 96-bit LDS slot lands on
//     ds_write_b96/ds_load_b96 only when the alignment + size both
//     work out, and the `__shared__` allocator's exact layout
//     across hipcc versions is fragile to depend on for an MC-level
//     opcode pin.
//   * Inline asm gives the operator a one-line pivot to recompile
//     against future ISA spellings without chasing alignment
//     promotion / vectoriser quirks.
//
// The store is bound to a 96-bit VGPR triple via `=v` on a 12-byte
// composite; the load result is then written to global memory so
// neither the load nor the store is DCE'd.  `s_wait_dscnt 0` after
// the load is required by the gfx12 memory model before the
// loaded VGPR triple is consumed.
//
// MAINTENANCE: any change to the `dsClassify` 3-dword branch in
// handle_ds.cpp must keep the `<3 x i32>` shape this fixture
// pins — alternative lowerings (e.g. 3 separate i32 loads, an
// `i96` integer load, or a `<6 x i16>` slicing) would silently
// change the IR shape downstream tooling already greps for.

#include <hip/hip_runtime.h>

extern "C" __global__ void
ds_load_b96_kernel(unsigned int *out) {
  __shared__ unsigned int lds_buf[64 * 3];

  int tid = (int)threadIdx.x;
  // Per-lane LDS slot: 3 dwords starting at offset 3*tid*4 bytes.
  unsigned int addr = (unsigned int)(uintptr_t)&lds_buf[3 * tid];

  // Materialise the 3 dwords we want to round-trip through
  // ds_store_b96 / ds_load_b96.  Using distinct per-dword payloads
  // so a future regression that collapses one slot into another
  // (e.g. a vectoriser bug that turned `<3 x i32>` into `<2 x i32>`
  // + scalar) is observable at runtime, not just in IR.
  unsigned int s0 = 0xAA000000u + (unsigned int)tid;
  unsigned int s1 = 0xBB000000u + (unsigned int)tid;
  unsigned int s2 = 0xCC000000u + (unsigned int)tid;

  // Pack-then-store via ds_store_b96.  Triple-VGPR `v` constraint
  // works because `=v` on a `__attribute__((vector_size(12)))` lays
  // out three contiguous v-class registers in encoding order, which
  // is what ds_store_b96 expects for its `data` operand.
  typedef unsigned int v3u32 __attribute__((vector_size(12)));
  v3u32 stData = {s0, s1, s2};
  asm volatile("ds_store_b96 %[a], %[d]\n\t"
               "s_wait_dscnt 0\n\t"
               :
               : [a] "v"(addr), [d] "v"(stData));

  // __syncthreads to make sure every lane's store is visible before
  // any lane's load picks it up, even though within a wave the
  // s_wait_dscnt above already serialises this lane's accesses.
  __syncthreads();

  // Load-back via ds_load_b96 into a fresh triple.
  v3u32 ldData;
  asm volatile("ds_load_b96 %[d], %[a]\n\t"
               "s_wait_dscnt 0\n\t"
               : [d] "=v"(ldData)
               : [a] "v"(addr));

  // Splat each loaded dword to a distinct global slot so DCE can't
  // collapse the load and so a stuck-dword regression is observable.
  out[3 * tid + 0] = ldData[0];
  out[3 * tid + 1] = ldData[1];
  out[3 * tid + 2] = ldData[2];
}
