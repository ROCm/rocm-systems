// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for floating-point logarithm and exponential entry points.

#include "common/cpu.h"
#include "common/integer.h"
#include "math/older_half_unary_test.h"
#include "math/unary_test.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_math.h"
#include "rj_arith/rj_arith.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

// Check unary math results and semantics/status support across CPU levels.
TEST(MathExpF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_exp_f32_w32, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathLogF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_log_f32_w32, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathExpF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_exp_f32_w32_gfx12, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathLogF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_log_f32_w32_gfx12, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathExpF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_exp_f32_w64, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathLogF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_log_f32_w64, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathExpF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_exp_f32_w64_gfx12, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathLogF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_log_f32_w64_gfx12, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathExpF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_exp_f16_w32, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathLogF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_log_f16_w32, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathExpF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_exp_f16_w32_gfx12, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathLogF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_log_f16_w32_gfx12, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathExpF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_exp_f16_w64, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathLogF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_log_f16_w64, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathExpF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_exp_f16_w64_gfx12, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathLogF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_log_f16_w64_gfx12, 8);
}

// gfx1201 TRANS OMOD overflow ignores ROUND=RTZ (MODE=0xff).
TEST(MathTrans, OutputOverflowIgnoresGuestRounding) {
  using namespace rocjitsu::arith::test;
  struct Case {
    decltype(&rj_arith_v_exp_f32_w32_gfx12) call;
    uint32_t input, expected;
  };
  const Case cases[] = {
      {rj_arith_v_exp_f32_w32_gfx12, 0x42fe0000U, 0x7f800000U}, // exp2(127)
      {rj_arith_v_exp_f16_w32_gfx12, 0x4b80U, 0x7c00U},         // exp2(15)
      {rj_arith_v_rcp_f32_w32_gfx12, 0x00800000U, 0x7f800000U},
      {rj_arith_v_rcp_f16_w32_gfx12, 0x0400U, 0x7c00U},
  };
  for (auto cpu : supported_cpu_levels()) {
    for (auto semantics :
         {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      for (const auto &test : cases) {
        Registers<2, 32> r;
        r.v[0][0] = test.input;
        r.v[1][0] = 0; // FP16 writes preserve the other half.
        uint32_t status = 0;
        ASSERT_EQ(test.call(cpu | semantics | RJ_ARITH_FP_ROUND_ZERO, 1, RJ_ARITH_ALU_OMOD_4,
                            r.v + 1, r.v, &status),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[1][0], test.expected);
      }
    }
  }
}

TEST(MathExpF32, VectorModelParity) {
  using namespace rocjitsu::arith::test;
  const uint32_t boundaries[] = {0,           1,           0x33800000U, 0x337fffffU, 0x42fc0000U,
                                 0x42fc0001U, 0x43000000U, 0x7f800000U, 0x7f800123U};
  const uint64_t modes[] = {0, RJ_ARITH_ALU_NEG_A, RJ_ARITH_ALU_OMOD_2,
                            RJ_ARITH_ALU_OMOD_HALF | RJ_ARITH_ALU_CLAMP};
  for (bool gfx12 : {false, true}) {
    auto call = gfx12 ? rj_arith_v_exp_f32_w32_gfx12 : rj_arith_v_exp_f32_w32;
    for (uint64_t cpu : supported_cpu_levels()) {
      if (cpu == RJ_ARITH_CPU_BASELINE) {
        continue;
      }
      std::minstd_rand rng;
      for (int sample = 0; sample < 1024; ++sample) {
        uint32_t expected[32], actual[32];
        for (int lane = 0; lane < 32; ++lane) {
          uint32_t bits = rocjitsu::arith::test::random_word(rng);
          if (lane & 1) {
            bits = (bits & 0x807fffffU) | (uint32_t(103 + sample % 32) << 23);
          }
          if (lane < 9) {
            bits = boundaries[lane] | (sample & 1 ? 0x80000000U : 0);
          }
          expected[lane] = actual[lane] = bits;
        }
        uint64_t flags = RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
        if (sample & 2) {
          flags |= RJ_ARITH_FP_IEEE_MODE;
        }
        if (!gfx12 && (sample & 4)) {
          flags |= RJ_ARITH_FP_DX10_CLAMP_OFF;
        }
        const uint32_t exec_mask = sample & 8 ? 0xaaaaaaaaU : 0xffffffffU;
        uint32_t *expected_d = expected, *actual_d = actual;
        const uint32_t *expected_a = expected, *actual_a = actual;
        ASSERT_EQ(call(flags, exec_mask, modes[sample % 4], &expected_d, &expected_a, nullptr),
                  RJ_ARITH_SUCCESS);
        ASSERT_EQ(call(flags | cpu, exec_mask, modes[sample % 4], &actual_d, &actual_a, nullptr),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 32; ++lane) {
          ASSERT_EQ(actual[lane], expected[lane]) << cpu << ':' << sample << ':' << lane;
        }
      }
    }
  }
}

TEST(MathOlderHalfLogExp, Arithmetic) {
  rocjitsu::arith::test::check_older_half_unary<32>(rj_arith_v_exp_f16_w32_gfx908, 8);
  rocjitsu::arith::test::check_older_half_unary<64>(rj_arith_v_exp_f16_w64_gfx908, 8);
  rocjitsu::arith::test::check_older_half_unary<32>(rj_arith_v_log_f16_w32_gfx908, 9);
  rocjitsu::arith::test::check_older_half_unary<64>(rj_arith_v_log_f16_w64_gfx908, 9);
}

TEST(MathHalfLogExp, CapturedDigests) {
  struct Captured {
    uint32_t denorm_mode;
    bool overflow;
    uint64_t digest[4];
  };
  // FNV hashes of raw F16 hardware captures over all 65536 input encodings.
  // Order: LOG signaling/quiet policy, then EXP signaling/quiet policy.
  const Captured captured[] = {
      {0u,
       false,
       {0x7ec126a2efb7cc29ull, 0x7b139780c1becc29ull, 0xc64da58f553026d8ull,
        0xa9b68cfe7b8372d8ull}},
      {0u,
       true,
       {0xf5160d9ac46aec29ull, 0x3999dfce791d6c29ull, 0x689733bef391b6d8ull,
        0xbdc5546b38d182d8ull}},
      {1u,
       false,
       {0x1a7dc0df37544143ull, 0xab561022e31ba143ull, 0xc64da58f553026d8ull,
        0xa9b68cfe7b8372d8ull}},
      {1u,
       true,
       {0x07029e1e69dff32bull, 0xab21f15c2191eb2bull, 0x689733bef391b6d8ull,
        0xbdc5546b38d182d8ull}},
      {2u,
       false,
       {0x7ec126a2efb7cc29ull, 0x7b139780c1becc29ull, 0x590d5c6485394589ull,
        0x7a49d8175530c589ull}},
      {2u,
       true,
       {0xf5160d9ac46aec29ull, 0x3999dfce791d6c29ull, 0x610235de125ed589ull,
        0x36d64dcb7d64d589ull}},
      {3u,
       false,
       {0x1a7dc0df37544143ull, 0xab561022e31ba143ull, 0x590d5c6485394589ull,
        0x7a49d8175530c589ull}},
      {3u,
       true,
       {0x07029e1e69dff32bull, 0xab21f15c2191eb2bull, 0x610235de125ed589ull,
        0x36d64dcb7d64d589ull}},
  };
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (const auto &sample : captured) {
      for (bool logarithm : {true, false}) {
        for (bool quiet : {false, true}) {
          using Call = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *,
                               const uint32_t *const *, uint32_t *);
          const Call call = logarithm
                                ? (quiet ? rj_arith_v_log_f16_w32_gfx12 : rj_arith_v_log_f16_w32)
                                : (quiet ? rj_arith_v_exp_f16_w32_gfx12 : rj_arith_v_exp_f16_w32);
          const uint64_t flags =
              cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS |
              (quiet ? RJ_ARITH_FP_IEEE_MODE : 0) | (sample.overflow ? RJ_ARITH_FP16_OVFL : 0) |
              ((sample.denorm_mode & 1) ? 0 : RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) |
              ((sample.denorm_mode & 2) ? 0 : RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
          uint64_t digest = 14695981039346656037ULL;
          for (int base = 0; base < 65536; base += 32) {
            uint32_t input[32], output[32];
            for (int lane = 0; lane < 32; ++lane) {
              input[lane] = uint32_t(base + lane);
            }
            const uint32_t *a = input;
            uint32_t *d = output;
            uint32_t status = 0;
            ASSERT_EQ(call(flags, UINT32_MAX, 0, &d, &a, &status), RJ_ARITH_SUCCESS);
            for (int lane = 0; lane < 32; ++lane) {
              digest = (digest ^ (output[lane] & 0xffffU)) * 1099511628211ULL;
            }
          }
          EXPECT_EQ(digest, sample.digest[(logarithm ? 0 : 2) + quiet])
              << "log=" << logarithm << " quiet=" << quiet << " denorm=" << sample.denorm_mode
              << " overflow=" << sample.overflow << " cpu=" << cpu;
        }
      }
    }
  }
}
