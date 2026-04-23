// Verifies v3 post-xor3 under the EXACT Triton init state v3=v4=v2.
//
// The prior `bitonic_cross16_probe_kernel.hip` probe used distinct
// v3, v4 inputs and confirmed the cross-wired-swap + standard-xor3
// textbook model.  That left one hypothesis open: maybe my model
// predicts `v3 = self` after xor3 WHEN v3_in = v4_in = v2, but
// gfx1250 silicon does something different (e.g. pipeline forwarding
// reads pre-swap v3).
//
// This probe tests exactly that: initialize v3 = v4 = v2, run the
// permlane16_swap + xor3 pair, dump v3.  If v3 = L (self), my model
// is fully validated and the compare CAN'T be the issue — some
// other invariant I've been missing must drive the sort.
//
// Build:
//   hipcc --offload-arch=gfx1250 --genco \
//     test_data/gfx1250/bitonic_xor3_triton_state_kernel.hip \
//     -o test_data/gfx1250/bitonic_xor3_triton_state_gfx1250.hsaco
//   /opt/rocm-7.2.1/lib/llvm/bin/clang-offload-bundler \
//     --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1250 \
//     --input=test_data/gfx1250/bitonic_xor3_triton_state_gfx1250.hsaco \
//     --output=test_data/gfx1250/bitonic_xor3_triton_state_gfx1250.hsaco \
//     --unbundle
#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(32)
    void bitonic_xor3_triton_state_kernel(int *out_v3_after_xor3,
                                             int *out_v4_after_swap) {
  int L = threadIdx.x;
  int v2 = L;
  int v3 = L;   // EXACT Triton state: v3 = v2
  int v4 = L;   // EXACT Triton state: v4 = v2
  asm volatile(
      "v_permlane16_swap_b32 %[v3], %[v4]\n\t"
      "v_xor3_b32 %[v3_out], %[v3], %[v4], %[v2]\n\t"
      : [v3] "+v"(v3), [v4] "+v"(v4), [v3_out] "=v"(v3)
      : [v2] "v"(v2));
  out_v3_after_xor3[L] = v3;
  out_v4_after_swap[L] = v4;
}
