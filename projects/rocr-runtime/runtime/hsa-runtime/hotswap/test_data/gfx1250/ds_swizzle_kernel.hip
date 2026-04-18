// Source for ds_swizzle_gfx1250.hsaco — pinned end-to-end test for
// CROSS_LANE_SURVEY P6 (`ds_swizzle_b32` lift via
// llvm.amdgcn.ds.swizzle).
//
// Per the same principle as permlane16_swap_kernel.hip (P4) and
// dpp_quad_perm_kernel.hip (P5), this fixture gives the P6 lift a
// CI-resident regression gate:
//
//   1. Read the pre-built gfx1250 (wave32) `.hsaco`.
//   2. Lift through the transpiler pipeline to gfx942 (wave64).
//   3. Run the lifted kernel on gfx942 hardware.
//   4. Verify per-lane output matches the expected swizzle.
//
// We use offset:0x081F = BITMASK_PERM with and=0x1F, or=0, xor=2
// — a SWAP-2 pattern (lane L reads from lane L XOR 2). This is
// distinct from the GPT-OSS `sum_bitmatrix_rows` corpus pattern
// (offset:0x041F = SWAP-1, XOR 1) and exercises a different bit
// of the BITMASK_PERM XOR mask, which catches imm-extraction
// bugs that happen to round-trip the SWAP-1 case.
//
// Build (regenerate when the kernel is changed):
//   hipcc --offload-arch=gfx1250 --genco \
//     test_data/gfx1250/ds_swizzle_kernel.hip \
//     -o test_data/gfx1250/ds_swizzle_gfx1250.hsaco
//   /opt/rocm-7.2.1/lib/llvm/bin/clang-offload-bundler \
//     --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1250 \
//     --input=test_data/gfx1250/ds_swizzle_gfx1250.hsaco \
//     --output=test_data/gfx1250/ds_swizzle_gfx1250.hsaco \
//     --unbundle
//
// Per-lane setup: input[L] = L.
// Expected after the swizzle (per-32-lane-half independent on
// wave64, per the bit-5-preservation hardware contract documented
// in handle_ds.cpp::DS_SWIZZLE_B32 MODREP block):
//   output[L] = input[L XOR 2] within each 32-lane half.
//
// For wave64 lanes 32..63, the upper half independently performs
// SWAP-2: lane 32↔34, 33↔35, …, 60↔62, 61↔63. The expected
// pattern is therefore output[L] = (L XOR 2) for every L in
// [0, 64), which the test verifies for all 64 lanes.

#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(64) void ds_swizzle_kernel(
    int *out) {
  int L = threadIdx.x;
  int v = L;
  asm volatile(
      "ds_swizzle_b32 %[v], %[v] offset:0x081f\n\t"
      "s_wait_dscnt 0\n\t"
      : [v] "+v"(v));
  out[L] = v;
}
