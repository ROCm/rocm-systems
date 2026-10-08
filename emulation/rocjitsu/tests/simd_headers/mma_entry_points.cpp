// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#if defined(RJ_TEST_MMA_FALLBACKS)
#include "rocjitsu/isa/arch/amdgpu/shared/mma/no_avx512_dense.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma/no_stdx_dense.h"
#else
#include "rocjitsu/isa/arch/amdgpu/shared/mma/select.h"
#endif

#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cstdint>
#include <vector>

namespace rocjitsu::amdgpu {

// Actual calls instantiate the template bodies; a requires-expression alone
// would miss backend/callback mismatches inside them. This function is never run.
void compile_mma_entry_points(Wavefront &wf) {
  using namespace mma_backend;
  struct Result32 {
    uint32_t reg, lane, val;
  };
  struct Result16 {
    uint32_t reg, lane, sub;
    uint16_t val;
  };
  struct Result64 {
    uint32_t reg, lane, lo, hi;
  };
  std::vector<Result32> results32;
  std::vector<Result16> results16;
  std::vector<Result64> results64;
  auto stage = [](uint32_t, uint32_t, float *, float *) {};
  auto scale_word = [](uint32_t) { return uint64_t{0}; };
  auto scale_for = [](uint64_t, uint32_t, uint32_t) { return 1.0f; };
  auto scale_block = [](float value, uint32_t, uint32_t, uint32_t, uint32_t) { return value; };
  auto dense_k = [](uint32_t, uint32_t k) { return 2 * k; };
  auto initial_acc = [](uint32_t, uint32_t) { return 0.0f; };
  auto read_acc = [](auto &, uint32_t, uint32_t, uint32_t) { return 0.0f; };
  auto pack = [](float value) { return util::f32_to_f16(value); };
  auto apply_neg = [](auto value, uint32_t) { return value; };

  // Pass scalar counts as values, including temporaries, to keep the backend
  // contract independent of an adapter's local variable types and lifetimes.
  try_exec_f32_mixed_simd(wf, {16, 16, 16}, 1, 0, ACC_FROM_VGPR, 64, stage, results32);
  try_exec_wmma_f32_mixed_simd(wf, {16, 16, 16}, 16, 16, 0, 4, 8, extract_f16, extract_f16,
                               ACC_FROM_VGPR, 0, 32, results32);
  try_exec_wmma_f32_scaled_mixed_simd(wf, {16, 16, 64}, 8, 8, 0, 4, 8, extract_fp8, extract_fp8,
                                      ACC_FROM_VGPR, scale_word, scale_word, 0, 0, 0, 0, false, 0,
                                      4, scale_for, results32);
  try_exec_swmmac_f32_mixed_simd(wf, {16, 16, 64}, 16, 16, 0, 4, extract_f16, extract_f16, 32, 32,
                                 dense_k, initial_acc, results32);
  try_exec_wmma_packed16_simd(wf, {16, 16, 16}, 16, 0, 4, 8, extract_f16, extract_f16, read_acc,
                              pack, ACC_FROM_VGPR, 32, results16);
  try_exec_swmmac_packed16_simd(wf, {16, 16, 64}, 16, 0, 4, extract_f16, extract_f16, pack, 32, 32,
                                dense_k, initial_acc, results16);
  try_exec_f32_scaled_impl_simd(wf, {16, 16, 64}, 1, 8, 8, 0, 4, 8, extract_fp8, extract_fp8,
                                scale_block, ACC_FROM_VGPR, 0, 64, 2, results32);
  try_exec_i32_i8_simd(wf, {16, 16, 16}, 1, 0, 4, 8, ACC_FROM_VGPR, 0, 0, 0, 64, results32);
  try_exec_wmma_i32_simd(wf, {16, 16, 16}, 8, 0, 4, 8, extract_i8, extract_i8, false, ACC_FROM_VGPR,
                         32, results32);
  try_exec_swmmac_i32_simd(wf, {16, 16, 64}, 8, 0, 4, 8, extract_i8, extract_i8, false,
                           ACC_FROM_VGPR, 32, 32, dense_k, results32);
  try_exec_f64_simd(wf, {16, 16, 4}, 1, 0, 4, 8, ACC_FROM_VGPR, apply_neg, results64);

  // Fixed-shape entry points select portable, AVX-512, or fallback bodies.
  try_exec_wmma_f32_f32_spec_simd<16, 16, 16>(wf, 12, 0, 4, 8, ACC_FROM_VGPR, 0);
  try_exec_f32_mfma_f32_spec_simd<16, 16, 4, 1>(wf, 12, 0, 4, 8, ACC_FROM_VGPR, 0, 0, 0);
  try_exec_f32_mfma_f16_spec_simd<16, 16, 16>(wf, 12, 0, 4, 8, ACC_FROM_VGPR, 0, 0, 0);
  try_exec_f32_mfma_bf16_spec_simd<16, 16, 16>(wf, 12, 0, 4, 8, ACC_FROM_VGPR, 0, 0, 0);
  try_exec_wmma_f32_16x16x32_f16_simd(wf, 12, 0, 4, 8, ACC_FROM_VGPR, 0);
  try_exec_wmma_f32_16x16x32_bf16_simd(wf, 12, 0, 4, 8, ACC_FROM_VGPR, 0);
  try_exec_wmma_f32_f8_spec_simd<16, 16, 32, true, true>(wf, 12, 0, 4, 8, ACC_FROM_VGPR, 0);
  try_exec_wmma_bf16f32_16x16x32_bf16_simd(wf, 12, 0, 4, 8, ACC_FROM_VGPR, 0);
  try_exec_wmma_f16_spec_simd<16, 16, 32>(wf, 12, 0, 4, 8, ACC_FROM_VGPR);
  try_exec_wmma_bf16_spec_simd<16, 16, 32>(wf, 12, 0, 4, 8, ACC_FROM_VGPR);
  try_exec_wmma_f16_f8_spec_simd<16, 16, 32, true, true>(wf, 12, 0, 4, 8, ACC_FROM_VGPR, false);
  try_exec_wmma_i32_16x16x64_iu8_simd(wf, 12, 0, 4, 8, true, true, false, ACC_FROM_VGPR);
  try_exec_f32_mfma_f8_spec_simd<16, 16, 32, true, true>(wf, 12, 0, 4, 8, ACC_FROM_VGPR, 0, 0, 0);
  try_exec_i32_mfma_i8_spec_simd<16, 16, 16>(wf, 12, 0, 4, 8, ACC_FROM_VGPR);
}

} // namespace rocjitsu::amdgpu
