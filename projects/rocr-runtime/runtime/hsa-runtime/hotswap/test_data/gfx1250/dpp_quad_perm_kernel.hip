// Source for dpp_quad_perm_gfx1250.hsaco — pinned end-to-end test
// for CROSS_LANE_SURVEY P5 (DPP modifier lift via
// llvm.amdgcn.update.dpp).
//
// Per the same principle as permlane16_swap_kernel.hip (P4), this
// fixture gives the P5 emulation a CI-resident regression gate:
//
//   1. Read the pre-built gfx1250 (wave32) `.hsaco`.
//   2. Lift through the transpiler pipeline to gfx942 (wave64).
//   3. Run the lifted kernel on gfx942 hardware.
//   4. Verify per-lane output matches the expected DPP quad_perm
//      shuffle.
//
// We use `quad_perm:[1,0,3,2]` — within each 4-lane quad, swap
// adjacent pairs (lane 0↔1, 2↔3, 4↔5, 6↔7, …, 62↔63). This is
// equivalently a per-lane XOR-1 read: lane L gets input[L XOR 1].
// The pattern is structurally per-quad-independent, so wave64
// modulo-replication is trivially correct (every 4-lane group
// performs the same swap, regardless of wave width).
//
// Build (regenerate when the kernel is changed):
//   hipcc --offload-arch=gfx1250 --genco \
//     test_data/gfx1250/dpp_quad_perm_kernel.hip \
//     -o test_data/gfx1250/dpp_quad_perm_gfx1250.hsaco
//   /opt/rocm-7.2.1/lib/llvm/bin/clang-offload-bundler \
//     --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1250 \
//     --input=test_data/gfx1250/dpp_quad_perm_gfx1250.hsaco \
//     --output=test_data/gfx1250/dpp_quad_perm_gfx1250.hsaco \
//     --unbundle
//
// Per-lane setup: input[L] = L.
// Expected after the DPP shuffle:
//   output[L] = input[L XOR 1] = L XOR 1.

#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(64) void dpp_quad_perm_kernel(
    int *out) {
  int L = threadIdx.x;
  int v = L;
  asm volatile(
      "v_mov_b32_dpp %[v], %[v] quad_perm:[1,0,3,2] row_mask:0xf "
      "bank_mask:0xf\n\t"
      : [v] "+v"(v));
  out[L] = v;
}
