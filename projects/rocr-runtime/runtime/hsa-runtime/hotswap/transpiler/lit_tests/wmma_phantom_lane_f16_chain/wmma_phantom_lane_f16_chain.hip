// ARCH: gfx1250
//
// Minimal 2-WMMA-chained phantom-lane repro for Phase 3 debugging.
// This exercises the K-loop accumulator chain path of `matmul_fp16`
// in the smallest possible shape: one 16×16 K=32 WMMA whose output
// feeds another 16×16 K=32 WMMA via C-operand chaining.  Under
// `__launch_bounds__(32)` the kernel is in the phantom-lane regime
// so `raiser.cpp` selects `ModuloReplicationProjection`, and the
// MFMA lowering must correctly thread the accumulator between the
// two WMMA calls so iter-1's WMMA consumes iter-0's true output
// (not a stale / zero value on the source-active lanes).

#include <hip/hip_runtime.h>

typedef __fp16 v16h __attribute__((ext_vector_type(16)));
typedef float  v8f  __attribute__((ext_vector_type(8)));

// Writes each thread's 8 D-fragment floats into `c_ptr[threadIdx.x]`
// so the host can inspect every WMMA-layout cell without racing.
extern "C" __global__ __launch_bounds__(32) void
wmma_f16_chain_kernel(const v16h *a0_ptr,
                      const v16h *b0_ptr,
                      const v16h *a1_ptr,
                      const v16h *b1_ptr,
                      v8f *c_ptr) {
  v16h a0 = a0_ptr[0];
  v16h b0 = b0_ptr[0];
  v16h a1 = a1_ptr[0];
  v16h b1 = b1_ptr[0];
  v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
  acc = __builtin_amdgcn_wmma_f32_16x16x32_f16(
      /*a_neg=*/false, a0,
      /*b_neg=*/false, b0,
      /*c_mod=*/(short)0, acc,
      /*a_reuse=*/false,
      /*b_reuse=*/false);
  acc = __builtin_amdgcn_wmma_f32_16x16x32_f16(
      /*a_neg=*/false, a1,
      /*b_neg=*/false, b1,
      /*c_mod=*/(short)0, acc,
      /*a_reuse=*/false,
      /*b_reuse=*/false);
  c_ptr[threadIdx.x] = acc;
}
