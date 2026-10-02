// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/transcendental_simd.h"
#include "util/amdgpu_rcp.h"
#include "util/amdgpu_rsq.h"

#include <array>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace {
using rocjitsu::amdgpu::transcendental::evaluate_f32_simd;
using rocjitsu::amdgpu::transcendental::F32Operation;

uint32_t reference(F32Operation operation, uint32_t bits, bool quiet = true) {
  const float input = std::bit_cast<float>(bits);
  return std::bit_cast<uint32_t>(operation == F32Operation::Rcp
                                     ? util::amdgpu_rcp_f32(input, quiet)
                                     : util::amdgpu_rsq_f32(input, quiet));
}

template <size_t O, size_t C>
void check_corrections(F32Operation operation, const uint16_t (&offsets)[O],
                       const uint16_t (&corrections)[C]) {
  const int64_t last_index = (O - 1) * 32768 - 1;
  auto check = [&](int64_t center) {
    std::array<uint32_t, 8> inputs{}, expected{}, outputs{};
    for (unsigned lane = 0; lane < inputs.size(); ++lane) {
      int64_t index = center + lane - 3;
      if (index < 0 || index > last_index)
        index = 0;
      inputs[lane] = 0x3f800000u + static_cast<uint32_t>(index);
      expected[lane] = reference(operation, inputs[lane]);
    }
    evaluate_f32_simd(operation, inputs.data(), outputs.data(), inputs.size(), 0, false);
    EXPECT_EQ(outputs, expected) << "op " << unsigned(operation) << " index " << center;
  };
  for (size_t bucket = 0; bucket + 1 < O; ++bucket) {
    check(bucket << 15);
    check((bucket << 15) + 32767);
    for (size_t i = offsets[bucket]; i < offsets[bucket + 1]; ++i)
      check((bucket << 15) + (corrections[i] & 0x7fff));
  }
}

TEST(ReciprocalSimd, CorrectionHitsMissesAndSegmentBoundaries) {
  check_corrections(F32Operation::Rcp, util::detail::kAmdgpuRcpCorrectionOffsets,
                    util::detail::kAmdgpuRcpCorrections);
  check_corrections(F32Operation::Rsq, util::detail::kAmdgpuRsqCorrectionOffsets,
                    util::detail::kAmdgpuRsqCorrections);
}

TEST(ReciprocalSimd, PowersAndExtremeExponents) {
  for (auto operation : {F32Operation::Rcp, F32Operation::Rsq})
    for (uint32_t exponent = 0; exponent < 256; ++exponent)
      for (uint32_t sign : {0u, 0x80000000u}) {
        std::array<uint32_t, 8> inputs{}, expected{}, outputs{};
        constexpr uint32_t mantissas[]{0, 1, 2, 0x3ffff, 0x40000, 0x7ffffd, 0x7ffffe, 0x7fffff};
        for (unsigned lane = 0; lane < inputs.size(); ++lane) {
          inputs[lane] = sign | (exponent << 23) | mantissas[lane];
          expected[lane] = reference(operation, inputs[lane]);
        }
        evaluate_f32_simd(operation, inputs.data(), outputs.data(), inputs.size(), 3, true);
        ASSERT_EQ(outputs, expected) << "op " << unsigned(operation) << " exponent " << exponent;
      }
}

TEST(ReciprocalSimd, FixedSpecialAndPowerResults) {
  constexpr std::array<uint32_t, 16> inputs{0,          0x80000000, 1,          0x807fffff,
                                            0x3f800000, 0x40000000, 0x40800000, 0xbf800000,
                                            0x7f800000, 0xff800000, 0x7f800001, 0xff800001,
                                            0x7fc12345, 0xffc12345, 0x7f7fffff, 0xff7fffff};
  constexpr std::array<uint32_t, 16> rcp{0x7f800000, 0xff800000, 0x7f800000, 0xff800000,
                                         0x3f800000, 0x3f000000, 0x3e800000, 0xbf800000,
                                         0,          0x80000000, 0x7f800001, 0xff800001,
                                         0x7fc12345, 0xffc12345, 0,          0x80000000};
  // Positive max-finite RSQ remains covered by the scalar differential above.
  constexpr std::array<uint32_t, 8> rsq{0x7f800000, 0xff800000, 0x7f800000, 0xff800000,
                                        0x3f800000, 0x3f3504f3, 0x3f000000, 0xffc00000};
  std::array<uint32_t, 16> output{};
  evaluate_f32_simd(F32Operation::Rcp, inputs.data(), output.data(), inputs.size(), 0, false);
  EXPECT_EQ(output, rcp);
  std::array<uint32_t, 8> rsq_output{};
  evaluate_f32_simd(F32Operation::Rsq, inputs.data(), rsq_output.data(), rsq_output.size(), 3,
                    true);
  EXPECT_EQ(rsq_output, rsq);
}

TEST(ReciprocalSimd, IndependentHostControlAndEnabledTraps) {
#if defined(__x86_64__)
  std::fenv_t original{};
  ASSERT_EQ(std::fegetenv(&original), 0);
  const unsigned original_mxcsr = _mm_getcsr();
  const int original_errno = errno;
  constexpr std::array<uint32_t, 8> input{0x7f800001, 0xff800001, 0,          0x80000001,
                                          0xbf800000, 0x00800000, 0x7f7fffff, 0x3fa12345};
  for (auto operation : {F32Operation::Rcp, F32Operation::Rsq})
    for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO})
      for (unsigned controls : {0u, 0x8040u})
        for (bool traps : {false, true}) {
          // Assertions and test-framework formatting execute only after restore.
          std::fenv_t before{}, after{}, held{};
          std::feholdexcept(&held);
          std::fesetround(rounding);
          unsigned csr = _mm_getcsr();
          csr = (csr & ~0xe07fu) | controls | 0x4000u;
          if (traps) {
            csr &= ~0x1f80u;
            unsigned short control;
            asm volatile("fnstcw %0" : "=m"(control));
            control &= ~0x3fu;
            asm volatile("fldcw %0" : : "m"(control));
          } else {
            csr |= 0x24u;
          }
          _mm_setcsr(csr);
          std::fegetenv(&before);
          errno = EILSEQ;
          std::array<uint32_t, 8> output{};
          evaluate_f32_simd(operation, input.data(), output.data(), output.size(), 0, false);
          const int observed_errno = errno;
          const unsigned observed_csr = _mm_getcsr();
          std::fegetenv(&after);
          std::fesetenv(&original);
          _mm_setcsr(original_mxcsr);
          errno = original_errno;
          EXPECT_EQ(observed_errno, EILSEQ);
          EXPECT_EQ(observed_csr, csr);
          EXPECT_EQ(std::memcmp(&before, &after, sizeof(before)), 0);
          for (unsigned lane = 0; lane < output.size(); ++lane)
            EXPECT_EQ(output[lane], reference(operation, input[lane], false));
        }
#else
  GTEST_SKIP() << "Independent x87 and SSE controls are x86-specific";
#endif
}
} // namespace
