// ARCH: gfx1250
//
// Lift fixture for the gfx11+ two-address LDS load family
// (`ds_load_2addr_b32` / `ds_load_2addr_stride64_b32`, aka
// SemOp::DS_READ2_B32 / SemOp::DS_READ2ST64_B32 in
// transpiler/semop.hpp).  The companion `.ll` pins the lowering
// shape; the handler back-reference is the dedicated DS_READ2 /
// DS_WRITE2 block in transpiler/handle_ds.cpp (search for the
// `ds2Classify` lambda).
//
// Why a dedicated fixture is non-negotiable here:
//
// Before the `ds2Classify` block landed, the generic single-offset
// DS_READ handler lifted DS_READ2_B32 into a single contiguous
// `load <2 x i32>` at `vaddr + raw_offset0` — correct only when
// `offset0:0 offset1:1` and silently wrong for every other offset
// pair.  The corpus routinely uses non-(0,1) pairs (e.g. tensilelite
// GEMM tiles issue `offset0:4 offset1:6`, `offset0:64 offset1:66`,
// …), so that path produced a silent miscompile on 4 kernels /
// 144 DS_READ2_B32 sites.  This fixture exists to keep the new
// handler semantics pinned and make any regression back to the
// contiguous `<2 x i32>` shape visible in the diff.
//
// Semantic pins:
//
//   1. DS_READ2_B32 with `offset0:4 offset1:6` must emit TWO
//      independent `load i32, ptr addrspace(3)` at byte offsets 16
//      and 24 (= rawField * 4), NOT a single `load <2 x i32>` at
//      any byte offset.
//   2. DS_READ2ST64_B32 with `offset0:2 offset1:3` must emit TWO
//      independent `load i32, ptr addrspace(3)` at byte offsets 512
//      and 768 (= rawField * 256) — this pins the ST64 unit scaling
//      that distinguishes the stride-64 variant from the dword-
//      granular non-ST64 form.
//   3. Each emitted load carries `align 4` (the access-width
//      alignment the handler stamps via `CreateAlignedLoad`); the
//      AMDGPU LDS ABI guarantees dword-alignment for these accesses
//      and pinning it prevents the backend from falling back to
//      byte-granular byte-address expansions.
//
// We use inline asm for both instructions because:
//
//   * `ds_load_2addr_stride64_b32` has no C-level selection path —
//     the compiler emits DS_READ2ST64_B32 only when it can prove
//     both elements of a load pair are exactly `N * 64` dwords apart
//     in an `__shared__` array.  Reliably hitting that idiom without
//     a chain of alignment hints is fragile; inline asm makes the
//     mnemonic the one degree of freedom under test.
//   * The non-(0,1) offset pair we want (`offset0:4 offset1:6`) is
//     similarly hard to coerce out of the HIP vectoriser without an
//     array layout the fixture would need to hard-code.  The inline-
//     asm form mirrors what the ds_load_b96 fixture already does for
//     the 96-bit load/store pair.
//
// Live-out: each loaded dword is splatted to a distinct global slot so
// DCE cannot collapse the pair into a single read.  A runtime
// regression that mixed up the two halves (wrong pointer scaling or
// stride) would therefore show up as observable data corruption, not
// just silent IR drift.
//
// MAINTENANCE: the SemOp pair this fixture covers lives in
// `transpiler/semop.hpp` under `// -- DS --` — the docstrings there
// explain the adjacency of DS_READ2_B32 and DS_READ2ST64_B32 in the
// enum.  The opcode map uses `E(DS_READ2_B32, DS_READ2_B32)` /
// `E(DS_READ2ST64_B32, DS_READ2ST64_B32)` so the canonicalization
// chain collapses the `_gfx9` / `_gfx11` / `_gfx12` reals onto these
// pseudos before dispatch.

#include <hip/hip_runtime.h>

extern "C" __global__ void
ds_load_2addr_b32_kernel(unsigned int *out) {
  // Enough LDS to reach the 256-byte-per-rawField stride of the
  // ST64 variant with the offset pair (2, 3) — that lands at bytes
  // 512 and 768, so we need at least 1 KiB to stay in-bounds.
  __shared__ unsigned int lds_buf[1024];

  int tid = (int)threadIdx.x;
  unsigned int addr = (unsigned int)(uintptr_t)&lds_buf[0];

  typedef unsigned int v2u32 __attribute__((vector_size(8)));

  // DS_READ2_B32 with offset0:4 offset1:6.  Hardware reads at byte
  // offsets 16 (= 4 * 4) and 24 (= 6 * 4); the destination is a
  // two-dword VGPR pair, written into the `ld_dword` pair below.
  v2u32 ld_dword;
  asm volatile(
      "ds_load_2addr_b32 %[d], %[a] offset0:4 offset1:6\n\t"
      "s_wait_dscnt 0\n\t"
      : [d] "=v"(ld_dword)
      : [a] "v"(addr));

  // DS_READ2ST64_B32 with offset0:2 offset1:3.  Hardware reads at
  // byte offsets 512 (= 2 * 256) and 768 (= 3 * 256); same two-dword
  // destination shape as the non-ST64 variant.
  v2u32 ld_stride64;
  asm volatile(
      "ds_load_2addr_stride64_b32 %[d], %[a] offset0:2 offset1:3\n\t"
      "s_wait_dscnt 0\n\t"
      : [d] "=v"(ld_stride64)
      : [a] "v"(addr));

  out[4 * tid + 0] = ld_dword[0];
  out[4 * tid + 1] = ld_dword[1];
  out[4 * tid + 2] = ld_stride64[0];
  out[4 * tid + 3] = ld_stride64[1];
}
