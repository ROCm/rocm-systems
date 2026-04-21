// End-to-end gpu-test fixture for the `1.0f / sqrtf(x)` fdiv expansion
// on the salmon gfx1250 → gfx942 path.  Pins the runtime correctness of
// the v_div_scale_f32-literal-numer decoder fix (commit 52ccf893aa +
// follow-up 46f6acdb88) — without this test, the only end-to-end
// signal for the fix was the compare_correctness layer-norm recipe,
// which had separate orthogonal miscompiles at each commit point that
// could have masked a regression on the rcp-sqrt math.
//
// The kernel's single expression `out[i] = 1.0f / __builtin_sqrtf(x)`
// forces hipcc's AMDGPU backend to emit the full IEEE-conformant fdiv
// expansion with a literal `1.0` as the numerator in both scale calls
// (verified by `llvm-objdump --mcpu=gfx1250` on the produced .hsaco;
// same instruction shape as `lit_tests/v_div_scale_f32_literal_numer`).
// After salmon lift, those scale calls become
// `@llvm.amdgcn.div.scale.f32(1.0, <sqrt-result>, flag)` — the
// canonical lifted shape the lit fixture pins at IR level; this GPU
// test pins the end-to-end numerical result after re-lowering.
//
// Pure per-thread elementwise: no reductions, no cross-lane ops, no
// wave-size-sensitive idioms.  Cross-widening correctness for the
// div-scale pair is the only observable behaviour this fixture
// exercises; any regression that silently collapses the IEEE divide
// to a constant (the pre-audit bug shape) surfaces as a non-zero
// error count in every element, which the harness reports as an
// ordinary failure rather than an exotic per-lane pattern.
//
// Regeneration (matches the per-handler fixture recipe documented in
// test_data/gfx1250/README.md for permlane16_swap / dpp_quad_perm /
// ds_swizzle):
//
//   DD=projects/rocr-runtime/runtime/hsa-runtime/hotswap/test_data/gfx1250
//   hipcc --offload-arch=gfx1250 --genco \
//     "$DD/rcp_sqrt_kernel.hip" \
//     -o "$DD/rcp_sqrt_gfx1250.hsaco.bundle"
//   /opt/rocm-7.2.1/lib/llvm/bin/clang-offload-bundler --type=o \
//     --targets=hipv4-amdgcn-amd-amdhsa--gfx1250 \
//     --input="$DD/rcp_sqrt_gfx1250.hsaco.bundle" \
//     --output="$DD/rcp_sqrt_gfx1250.hsaco" --unbundle
//   rm -f "$DD/rcp_sqrt_gfx1250.hsaco.bundle"
//
// Verify with:
//   llvm-objdump -d --mcpu=gfx1250 "$DD/rcp_sqrt_gfx1250.hsaco" | grep -E 'v_sqrt_f32|v_div_scale_f32'
// The expected output contains at least one `v_sqrt_f32` and two
// `v_div_scale_f32` with a literal `1.0` operand in src2 (the shape
// that trips the literal-numer decoder).

#include <hip/hip_runtime.h>

extern "C" __global__ void rcp_sqrt_kernel(float *out, const float *in) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  // `volatile` load keeps the compute expression live across any
  // whole-program analysis the compiler might apply — without it
  // hipcc's AMDGPU backend can (and does, on some versions) prove
  // the computation dead and DCE the entire kernel body.
  float x = *(const volatile float *)(in + tid);
  out[tid] = 1.0f / __builtin_sqrtf(x);
}
