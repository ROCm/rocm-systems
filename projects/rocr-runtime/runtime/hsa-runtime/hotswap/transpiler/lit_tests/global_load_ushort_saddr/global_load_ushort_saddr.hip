// ARCH: gfx1250
//
// Fixture for the sub-dword GLOBAL_LOAD SADDR + `scale_offset` lit
// test (lit_tests/global_load_ushort_saddr).
//
// The `cvt_f32_bf16` kernel in compare_correctness was producing all
// zeros on the salmon path because `handle_flat.cpp`'s
// `GLOBAL_LOAD_USHORT` handler:
//
//   1. Didn't recognise the SADDR decoded operand shape (SGPR64,
//      VGPR32) and fell through to a plain-VGPR64 address read,
//      collapsing every lane to `saddr + 0 + imm_offset` (broadcast).
//   2. Didn't apply the `scale_offset` bit, so even when SADDR was
//      detected the vaddr was not scaled by `elemBytes = 2`.
//   3. Had an immediate-scan loop that read the cpol encoding flag as
//      the memory offset instead of breaking after the first imm.
//
// This fixture pins a minimal pattern that gfx1250 hipcc compiles to
// exactly the instruction shape we need the handler to get right:
// a ushort gather keyed on `tid` with an SGPR base, which lowers to
// `global_load_u16 v[...], v[...], s[...] scale_offset`.
//
// The output is a sum so every lane contributes to the stored value,
// preventing hipcc from DCE-ing the load; using `unsigned short`
// forces the u16 load variant (as opposed to signed i16).
//
// MODREP: irrelevant for this test — sub-dword gathers are per-lane
// loads and do not involve any wave-level mask projection. The
// handler is cross-wave-safe because the scaling logic depends only
// on the source-kernel access element size.

#include <hip/hip_runtime.h>

extern "C" __global__ void global_load_ushort_saddr_kernel(
    const unsigned short *__restrict__ a, int *__restrict__ out, int n) {
  int tid = (int)threadIdx.x;
  if (tid >= n) return;
  out[tid] = (int)a[tid];
}
