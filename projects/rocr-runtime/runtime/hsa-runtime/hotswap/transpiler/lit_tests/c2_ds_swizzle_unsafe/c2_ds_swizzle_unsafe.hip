// ARCH: gfx1250
//
// Negative-case fixture for the Class-2 ds_swizzle_b32 sub-category
// (lit_tests/c2_ds_swizzle_unsafe).
//
// CROSS_LANE_SURVEY.md item P6 lifted ds_swizzle_b32 through
// `llvm.amdgcn.ds.swizzle`, but the lift is only modulo-replication-
// safe for the QUAD_PERM and BITMASK_PERM swizzle sub-modes (those
// are 4-lane- or 32-lane-bounded by the imm encoding itself, so each
// 32-lane half of a wave64 target reproduces the source's wave32
// shuffle independently).
//
// FFT_MODE / ROTATE_MODE / unknown high-bit imms span the full
// source-wave width and are NOT modulo-replication-safe. The
// classifier's `dsSwizzleSafeForModRep` helper refuses them with a
// detail line that names the offending imm so operators can identify
// the exact sub-mode that needs P6.b coverage.
//
// The immediate `offset:0xe03f` here selects FFT_MODE_ENC (top
// nibble = 0xE) with FFT swizzle pattern 0x1F. This is a hypothetical
// pattern (no GPT-OSS corpus kernel uses FFT_MODE today) chosen
// purely to exercise the unsafe-imm refusal path.

#include <hip/hip_runtime.h>

extern "C" __global__ void c2_ds_swizzle_unsafe_kernel(int *a) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int v = a[tid];
  asm volatile(
      "ds_swizzle_b32 %[v], %[v] offset:0xe03f\n\t"
      "s_wait_dscnt 0\n\t"
      : [v] "+v"(v));
  a[tid] = v;
}
