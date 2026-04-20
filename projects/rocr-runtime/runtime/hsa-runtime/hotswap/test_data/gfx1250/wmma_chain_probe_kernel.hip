// Source for wmma_chain_probe_gfx1250.hsaco — targeted probes for
// isolating the gfx1250 → gfx942 `v_wmma_f32_16x16x32_f16` lowering
// failure mode observed in `Gfx1250Gpu.Matmul128x128` (large 128×128
// Triton matmul: ~16,000 / 16,384 elements mismatched).
//
// The existing small `matmul_f16_gfx1250.hsaco` (64×64 tile) passes
// end-to-end, while the large one (128×128 tile) fails, so the bug
// must live in the combination of:
//
//   - chain depth  (how many WMMAs accumulate into a single C tile)
//   - parallelism  (how many independent C tiles a wave owns)
//   - VGPR pressure (enough to spill into the gfx1250 `s_set_vgpr_msb`
//                    "high half" encoding, not exercised by the small
//                    matmul which stays below 256 VGPRs)
//
// This file provides a ladder of probes that vary one dimension at a
// time so the first failing step binary-searches the root cause:
//
//   wmma_chain1_kernel     1 tile × 1-chain (sanity)
//   wmma_chain2_kernel     1 tile × 2-chain (matches small matmul)
//   wmma_chain4_kernel     1 tile × 4-chain (matches large matmul K/32=4)
//   wmma_parallel4_kernel  4 tiles × 1-chain
//   wmma_parallel16_kernel 16 tiles × 1-chain (matches large matmul
//                                              per-K-iter tile count)
//   wmma_parallel16_chain4_kernel 16 tiles × 4-chain (full topology of
//                                              the failing large matmul
//                                              per wave)
//
// Inputs are chosen so the CPU reference is trivially computable:
//   A fragment per lane = all-ones   (<16 x half> of 1.0)
//   B fragment per lane = all-ones   (<16 x half> of 1.0)
// Per-WMMA contribution per output element = 32 (K dimension).
// Chain depth N ⇒ per-element final value = 32 * N.
// The per-tile accumulator initial value is 0.
//
// Build (regenerate when the kernel is changed):
//   DD=projects/rocr-runtime/runtime/hsa-runtime/hotswap/test_data/gfx1250
//   hipcc --offload-arch=gfx1250 --genco \
//     "$DD/wmma_chain_probe_kernel.hip" \
//     -o "$DD/wmma_chain_probe_gfx1250.hsaco.bundle"
//   /opt/rocm-7.2.1/lib/llvm/bin/clang-offload-bundler --type=o \
//     --targets=hipv4-amdgcn-amd-amdhsa--gfx1250 \
//     --input="$DD/wmma_chain_probe_gfx1250.hsaco.bundle" \
//     --output="$DD/wmma_chain_probe_gfx1250.hsaco" --unbundle
//   rm -f "$DD/wmma_chain_probe_gfx1250.hsaco.bundle"
//
// Verify (expected: one `v_wmma_f32_16x16x32_f16` per logical WMMA):
//   llvm-objdump -d --mcpu=gfx1250 \
//     "$DD/wmma_chain_probe_gfx1250.hsaco" | \
//     grep -c 'v_wmma_f32_16x16x32_f16'

#include <hip/hip_runtime.h>

typedef _Float16 v16half __attribute__((ext_vector_type(16)));
typedef float    v8f     __attribute__((ext_vector_type(8)));

// WMMA is Wave32 on gfx1250; one WMMA occupies the whole wavefront.
// Running with __launch_bounds__(64) (two Wave32 wavefronts per block)
// keeps the transpiled Wave64 projection fully populated (lanes 0-31
// from wavefront-0, lanes 32-63 from wavefront-1, both running the
// same uniform data pattern), matching the "full Wave64" topology the
// Triton matmul kernels launch (wg=128 → 4 Wave32 → 2 Wave64).
// Partial-Wave64 (wg=32) projection — with lanes 32-63 unpopulated —
// is a separate (and currently failing) path tracked by xfail.cmake.
#define PROBE_LAUNCH_BOUNDS __launch_bounds__(64)

// ----- 1 tile × 1 chain ----------------------------------------------------
extern "C" __global__ PROBE_LAUNCH_BOUNDS void
wmma_chain1_kernel(const v16half *a_ptr, const v16half *b_ptr, v8f *c_ptr) {
  v16half a = a_ptr[0];
  v16half b = b_ptr[0];
  v8f c = {0, 0, 0, 0, 0, 0, 0, 0};
  c = __builtin_amdgcn_wmma_f32_16x16x32_f16(false, a, false, b,
                                              0, c, false, false);
  c_ptr[0] = c;
}

// ----- 1 tile × 2 chain ----------------------------------------------------
extern "C" __global__ PROBE_LAUNCH_BOUNDS void
wmma_chain2_kernel(const v16half *a_ptr, const v16half *b_ptr, v8f *c_ptr) {
  v16half a = a_ptr[0];
  v16half b = b_ptr[0];
  v8f c = {0, 0, 0, 0, 0, 0, 0, 0};
  c = __builtin_amdgcn_wmma_f32_16x16x32_f16(false, a, false, b,
                                              0, c, false, false);
  c = __builtin_amdgcn_wmma_f32_16x16x32_f16(false, a, false, b,
                                              0, c, false, false);
  c_ptr[0] = c;
}

// ----- 1 tile × 4 chain ----------------------------------------------------
extern "C" __global__ PROBE_LAUNCH_BOUNDS void
wmma_chain4_kernel(const v16half *a_ptr, const v16half *b_ptr, v8f *c_ptr) {
  v16half a = a_ptr[0];
  v16half b = b_ptr[0];
  v8f c = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < 4; ++i) {
    c = __builtin_amdgcn_wmma_f32_16x16x32_f16(false, a, false, b,
                                                0, c, false, false);
  }
  c_ptr[0] = c;
}

// ----- 2 tiles × 1 chain ---------------------------------------------------
// Smallest parallel configuration that triggers the Parallel4 failure
// pattern; used to localize the regression between 1-tile (passing) and
// N-tile (failing) cases.
extern "C" __global__ PROBE_LAUNCH_BOUNDS void
wmma_parallel2_kernel(const v16half *a_ptr, const v16half *b_ptr,
                       v8f *c_ptr) {
  v16half a = a_ptr[0];
  v16half b0 = b_ptr[0];
  v16half b1 = b_ptr[1];
  v8f c0 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8f c1 = {0, 0, 0, 0, 0, 0, 0, 0};
  c0 = __builtin_amdgcn_wmma_f32_16x16x32_f16(false, a, false, b0, 0, c0, false, false);
  c1 = __builtin_amdgcn_wmma_f32_16x16x32_f16(false, a, false, b1, 0, c1, false, false);
  c_ptr[0] = c0;
  c_ptr[1] = c1;
}

// ----- 4 tiles × 1 chain ---------------------------------------------------
// 4 independent accumulators, matches small matmul per-K-iter tile count.
extern "C" __global__ PROBE_LAUNCH_BOUNDS void
wmma_parallel4_kernel(const v16half *a_ptr, const v16half *b_ptr,
                       v8f *c_ptr) {
  v16half a = a_ptr[0];
  v16half b0 = b_ptr[0];
  v16half b1 = b_ptr[1];
  v16half b2 = b_ptr[2];
  v16half b3 = b_ptr[3];
  v8f c0 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8f c1 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8f c2 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8f c3 = {0, 0, 0, 0, 0, 0, 0, 0};
  c0 = __builtin_amdgcn_wmma_f32_16x16x32_f16(false, a, false, b0, 0, c0, false, false);
  c1 = __builtin_amdgcn_wmma_f32_16x16x32_f16(false, a, false, b1, 0, c1, false, false);
  c2 = __builtin_amdgcn_wmma_f32_16x16x32_f16(false, a, false, b2, 0, c2, false, false);
  c3 = __builtin_amdgcn_wmma_f32_16x16x32_f16(false, a, false, b3, 0, c3, false, false);
  c_ptr[0] = c0;
  c_ptr[1] = c1;
  c_ptr[2] = c2;
  c_ptr[3] = c3;
}

// ----- 16 tiles × 1 chain --------------------------------------------------
// Matches the large matmul's per-K-iter tile count (16 WMMAs per group).
// The 16 independent accumulators typically push the kernel past the
// low-256-VGPR envelope, exercising the gfx1250 `s_set_vgpr_msb`
// encoding on source operands of the WMMA.
extern "C" __global__ PROBE_LAUNCH_BOUNDS void
wmma_parallel16_kernel(const v16half *a_ptr, const v16half *b_ptr,
                        v8f *c_ptr) {
  v16half a0 = a_ptr[0];
  v16half a1 = a_ptr[1];
  v16half a2 = a_ptr[2];
  v16half a3 = a_ptr[3];
  v16half b0 = b_ptr[0];
  v16half b1 = b_ptr[1];
  v16half b2 = b_ptr[2];
  v16half b3 = b_ptr[3];
  v8f c[16];
  for (int i = 0; i < 16; ++i)
    c[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
  v16half a_arr[4] = {a0, a1, a2, a3};
  v16half b_arr[4] = {b0, b1, b2, b3};
  for (int m = 0; m < 4; ++m) {
    for (int n = 0; n < 4; ++n) {
      int t = m * 4 + n;
      c[t] = __builtin_amdgcn_wmma_f32_16x16x32_f16(
          false, a_arr[m], false, b_arr[n], 0, c[t], false, false);
    }
  }
  for (int i = 0; i < 16; ++i)
    c_ptr[i] = c[i];
}

// ----- 16 tiles × 4 chain --------------------------------------------------
// Full topology of one wave64 slice of the failing 128×128 matmul per
// K-iter (16 tiles) × (K/32 = 4 K-iters).
extern "C" __global__ PROBE_LAUNCH_BOUNDS void
wmma_parallel16_chain4_kernel(const v16half *a_ptr, const v16half *b_ptr,
                                v8f *c_ptr) {
  v16half a0 = a_ptr[0];
  v16half a1 = a_ptr[1];
  v16half a2 = a_ptr[2];
  v16half a3 = a_ptr[3];
  v16half b0 = b_ptr[0];
  v16half b1 = b_ptr[1];
  v16half b2 = b_ptr[2];
  v16half b3 = b_ptr[3];
  v8f c[16];
  for (int i = 0; i < 16; ++i)
    c[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
  v16half a_arr[4] = {a0, a1, a2, a3};
  v16half b_arr[4] = {b0, b1, b2, b3};
  for (int k = 0; k < 4; ++k) {
    for (int m = 0; m < 4; ++m) {
      for (int n = 0; n < 4; ++n) {
        int t = m * 4 + n;
        c[t] = __builtin_amdgcn_wmma_f32_16x16x32_f16(
            false, a_arr[m], false, b_arr[n], 0, c[t], false, false);
      }
    }
  }
  for (int i = 0; i < 16; ++i)
    c_ptr[i] = c[i];
}
