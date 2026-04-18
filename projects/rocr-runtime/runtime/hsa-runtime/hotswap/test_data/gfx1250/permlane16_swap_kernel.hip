// Source for permlane16_swap_gfx1250.hsaco — pinned end-to-end test
// for CROSS_LANE_SURVEY P4 (`v_permlane16_swap_b32` lift).
//
// This is the source kernel for a Gfx1250Gpu GTest that:
//   1. Reads the pre-built gfx1250 (wave32) `.hsaco`.
//   2. Lifts it through the transpiler pipeline to gfx942 (wave64).
//   3. Runs the lifted kernel on gfx942 hardware.
//   4. Verifies per-lane output matches the expected XOR-16 swap.
//
// The test exists to give the P4 emulation a CI-resident regression
// gate. Without it, the only check that the lifted ds_bpermute
// chain actually swaps correctly is the embedded probe results in
// the handler comment block — documentation, not a test.
//
// Build (regenerate when the kernel is changed):
//   hipcc --offload-arch=gfx1250 --genco \
//     test_data/gfx1250/permlane16_swap_kernel.hip \
//     -o test_data/gfx1250/permlane16_swap_gfx1250.hsaco
//   /opt/rocm-7.2.1/lib/llvm/bin/clang-offload-bundler \
//     --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1250 \
//     --input=test_data/gfx1250/permlane16_swap_gfx1250.hsaco \
//     --output=test_data/gfx1250/permlane16_swap_gfx1250.hsaco \
//     --unbundle
//
// Per-lane setup:
//   vdst_in[L]  = L
//   src0_in[L]  = 1000 + L
//
// Expected after the swap (per the VOP_PERMLANE_SWAP profile):
//   new_vdst[L]      = src0_in[L XOR 16] = 1000 + (L XOR 16)
//   new_src0_out[L]  = vdst_in[L XOR 16] = (L XOR 16)
//
// On the lifted gfx942 wave64 kernel, modulo-replication runs each
// 32-lane half independently, so the lower (lanes 0..31) and upper
// (lanes 32..63) halves each perform the swap in isolation. The
// expected output is identical for all 64 lanes (using the per-half
// XOR-16 partner), and the test verifies all 64 entries.

#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(64) void permlane16_swap_kernel(
    int *out_vdst, int *out_src0_out) {
  int L = threadIdx.x;
  int v = L;
  int u = 1000 + L;
  asm volatile(
      "v_permlane16_swap_b32 %[v], %[u]\n\t"
      : [v] "+v"(v), [u] "+v"(u));
  out_vdst[L] = v;
  out_src0_out[L] = u;
}
