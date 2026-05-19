// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hsa_translate_test.cpp
/// @brief End-to-end hardware tests for translated DBT code objects.

#include "hsa_test_utils.h"
#include "tools/dbt_translate.h"
#include "tools/hsa_run_kernel.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#ifdef HAS_HOST_AMDGPU

namespace {

using rocjitsu::dbt_test::bytes_of;
using rocjitsu::dbt_test::detect_hsa_host_target;
using rocjitsu::dbt_test::find_output;
using rocjitsu::dbt_test::kernel_path;
using rocjitsu::dbt_test::ptr_arg;
using rocjitsu::dbt_test::u32_arg;

uint16_t f32_to_f16(float val) {
  uint32_t fbits;
  std::memcpy(&fbits, &val, sizeof(fbits));
  uint32_t sign = (fbits >> 16) & 0x8000;
  int32_t exp = static_cast<int32_t>((fbits >> 23) & 0xFF) - 127;
  uint32_t mant = fbits & 0x7FFFFF;
  if ((fbits & 0x7FFFFFFF) == 0)
    return static_cast<uint16_t>(sign);
  if (exp > 15)
    return static_cast<uint16_t>(sign | 0x7BFF);
  if (exp < -14)
    return static_cast<uint16_t>(sign);
  return static_cast<uint16_t>(sign | ((exp + 15) << 10) | (mant >> 13));
}

float f16_to_f32(uint16_t h) {
  uint32_t sign = (h >> 15) & 1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  if (exp == 0)
    return sign ? -0.0f : 0.0f;
  uint32_t f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
  float result;
  std::memcpy(&result, &f, sizeof(result));
  return result;
}

} // namespace

TEST(HsaTranslateTest, TranslateAndDispatchVectorAdd) {
  auto target = detect_hsa_host_target();
  ASSERT_NE(target.mach, 0u) << "Test requires RDNA3 (gfx1100) or RDNA4 (gfx1200/1201) GPU";

  rocjitsu::tools::TranslateOptions translate;
  translate.input_path = kernel_path("vector_add");
  translate.input_target = ROCJITSU_CODE_TARGET_GFX950;
  translate.guest_arch = ROCJITSU_CODE_ARCH_CDNA4;
  translate.host_arch = target.arch;
  translate.target_mach = target.mach;
  translate.validate_host_decode = true;

  auto translated = rocjitsu::tools::translate_code_object(translate);
  ASSERT_TRUE(translated.ok()) << translated.errors.front().message;
  ASSERT_FALSE(translated.value.elf_bytes.empty());
  EXPECT_TRUE(translated.warnings.empty()) << translated.warnings.front();

  constexpr uint32_t N = 1024;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  std::vector<float> A(N), B(N), C_golden(N);
  for (uint32_t i = 0; i < N; ++i) {
    A[i] = dist(rng);
    B[i] = dist(rng);
    C_golden[i] = A[i] + B[i];
  }

  rocjitsu::tools::HsaRunOptions run;
  run.code_object_bytes = translated.value.elf_bytes;
  run.kernel_name = "vector_add";
  run.require_agent_isa = target.isa_name.find("gfx1201") != std::string::npos ? "gfx1201"
                                                                                : "gfx1200";
  run.grid_x = N;
  run.workgroup_x = 64;
  run.kernarg_size = 32;
  run.buffers = {
      {.name = "A",
       .size = N * sizeof(float),
       .input = bytes_of(A),
       .zero_fill = false,
       .copy_output = false},
      {.name = "B",
       .size = N * sizeof(float),
       .input = bytes_of(B),
       .zero_fill = false,
       .copy_output = false},
      {.name = "C",
       .size = N * sizeof(float),
       .input = {},
       .zero_fill = true,
       .copy_output = true},
  };
  run.arg_patches = {
      ptr_arg(0, "A"),
      ptr_arg(8, "B"),
      ptr_arg(16, "C"),
      u32_arg(24, N),
  };

  auto executed = rocjitsu::tools::run_hsa_kernel(run);
  ASSERT_TRUE(executed.ok()) << executed.errors.front().message;
  const auto *C_output = find_output(executed.value, "C");
  ASSERT_NE(C_output, nullptr);
  ASSERT_EQ(C_output->bytes.size(), N * sizeof(float));

  const auto *C_result = reinterpret_cast<const float *>(C_output->bytes.data());
  int mismatches = 0;
  for (uint32_t i = 0; i < N; ++i) {
    if (std::abs(C_result[i] - C_golden[i]) > 1e-5f)
      ++mismatches;
  }
  EXPECT_EQ(mismatches, 0) << mismatches << " element mismatches out of " << N;
}

TEST(HsaTranslateTest, TranslateAndDispatchMfma16x16) {
  auto target = detect_hsa_host_target();
  ASSERT_NE(target.mach, 0u) << "Test requires RDNA4 (gfx1200/1201) GPU";
  if (target.arch != ROCJITSU_CODE_ARCH_RDNA4) {
    GTEST_SKIP() << "MFMA->WMMA semantic expansion is currently implemented only for RDNA4";
  }

  rocjitsu::tools::TranslateOptions translate;
  translate.input_path = kernel_path("matmul_mfma_16x16");
  translate.input_target = ROCJITSU_CODE_TARGET_GFX950;
  translate.guest_arch = ROCJITSU_CODE_ARCH_CDNA4;
  translate.host_arch = target.arch;
  translate.target_mach = target.mach;

  auto translated = rocjitsu::tools::translate_code_object(translate);
  ASSERT_TRUE(translated.ok()) << translated.errors.front().message;
  ASSERT_FALSE(translated.value.elf_bytes.empty());

  constexpr uint32_t M = 16;
  constexpr uint32_t N = 16;
  constexpr uint32_t K = 16;
  constexpr int kNumIterations = 10;

  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  int total_mismatches = 0;
  for (int iter = 0; iter < kNumIterations; ++iter) {
    std::vector<uint16_t> A(M * K), B(K * N);
    for (auto &v : A)
      v = f32_to_f16(dist(rng));
    for (auto &v : B)
      v = f32_to_f16(dist(rng));

    std::vector<float> C_golden(M * N, 0.0f);
    for (uint32_t i = 0; i < M; ++i)
      for (uint32_t j = 0; j < N; ++j)
        for (uint32_t k = 0; k < K; ++k)
          C_golden[i * N + j] += f16_to_f32(A[i * K + k]) * f16_to_f32(B[k * N + j]);

    rocjitsu::tools::HsaRunOptions run;
    run.code_object_bytes = translated.value.elf_bytes;
    run.kernel_name = "matmul_mfma_16x16";
    run.require_agent_isa = target.isa_name.find("gfx1201") != std::string::npos ? "gfx1201"
                                                                                  : "gfx1200";
    run.grid_x = 64;
    run.workgroup_x = 64;
    run.kernarg_size = 24;
    run.buffers = {
        {.name = "A",
         .size = M * K * sizeof(uint16_t),
         .input = bytes_of(A),
         .zero_fill = false,
         .copy_output = false},
        {.name = "B",
         .size = K * N * sizeof(uint16_t),
         .input = bytes_of(B),
         .zero_fill = false,
         .copy_output = false},
        {.name = "C",
         .size = M * N * sizeof(float),
         .input = {},
         .zero_fill = true,
         .copy_output = true},
    };
    run.arg_patches = {
        ptr_arg(0, "A"),
        ptr_arg(8, "B"),
        ptr_arg(16, "C"),
    };

    auto executed = rocjitsu::tools::run_hsa_kernel(run);
    ASSERT_TRUE(executed.ok()) << executed.errors.front().message;
    const auto *C_output = find_output(executed.value, "C");
    ASSERT_NE(C_output, nullptr);
    ASSERT_EQ(C_output->bytes.size(), M * N * sizeof(float));

    const auto *C_result = reinterpret_cast<const float *>(C_output->bytes.data());
    int mismatches = 0;
    for (uint32_t i = 0; i < M * N; ++i) {
      const float expected = C_golden[i];
      const float got = C_result[i];
      const float tol = std::max(0.01f * std::abs(expected), 0.001f);
      if (std::abs(got - expected) > tol)
        ++mismatches;
    }
    total_mismatches += mismatches;
  }

  EXPECT_EQ(total_mismatches, 0) << total_mismatches << " total mismatches across "
                                 << kNumIterations << " iterations";
}

#endif // HAS_HOST_AMDGPU
