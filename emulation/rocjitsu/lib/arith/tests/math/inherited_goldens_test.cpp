// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Selected literal math boundaries and compact complete-domain capture hashes.
// These inherit the former rocjitsu tests; no new numerical model is constructed here.

#include "common/capture_hash.h"
#include "common/cpu.h"
#include "common/registers.h"
#include "rj_arith/detail/rj_arith_math.h"

#include <bit>
#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;
using Call = decltype(&rj_arith_v_rcp_f32_w32);

// Preserve selected reduction/rounding boundaries and NaN policies on every CPU path.
TEST(InheritedMathGoldens, Encodings) {
  const Call calls[] = {rj_arith_v_rcp_f32_w32, rj_arith_v_rsq_f32_w32, rj_arith_v_sqrt_f32_w32,
                        rj_arith_v_log_f32_w32, rj_arith_v_exp_f32_w32, rj_arith_v_sin_f32_w32,
                        rj_arith_v_cos_f32_w32};
  const struct {
    int operation;
    uint32_t input, expected;
    bool quiet = true;
  } cases[] = {{0, 0x3f802922, 0x3f7fadd6},
               {0, 0x7fa12345, 0x7fe12345},
               {0, 0x7fa12345, 0x7fa12345, false},
               {1, 0x40000001, 0x3f3504f2},
               {1, 0x7f7fffff, 0x1f800000},
               {2, 0x3f7a2707, 0x3f7d0f30},
               {2, 0x00800001, 0x20000000},
               {2, 0x7fa12345, 0x7fa12345, false},
               {3, 0x3f7c0000, 0xbcba1f74},
               {3, 0x3f800001, 0x3438aa3b},
               {4, 0xb3800000, 0x3f7fffff},
               {4, 0x3f0567ec, 0x3fb7b03d},
               {4, 0xc2fc0001, 0},
               {5, 1, 6},
               {5, 0x3a000000, 0x3b490fcb},
               {5, 0x3e7fff08, 0x3f800000},
               {6, 0x3e7fff08, 0x37c2c761},
               {6, 0x7f7fffff, 0x3f800000}};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (const auto &sample : cases) {
      Registers<2, 32> r;
      r.v[0][0] = sample.input;
      const uint64_t flags = cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS |
                             (sample.quiet ? RJ_ARITH_FP_IEEE_MODE : 0);
      ASSERT_EQ(calls[sample.operation](flags, 1, 0, r.v + 1, r.v, nullptr), RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[1][0], sample.expected)
          << "operation=" << sample.operation << " input=" << sample.input << " cpu=" << cpu;
    }
  }
}

// Hash a contiguous raw-bit interval, or an exactly spaced signed unit interval.
// Full waves keep the inherited exhaustive hashes inexpensive; one scalar CPU path
// checks the model, while the selected encodings above cover every CPU path.
void check_digest(Call call, uint32_t first, int count, uint64_t expected, bool fraction = false) {
  Registers<2, 32> r;
  uint64_t digest = capture_hash_seed;
  for (int base = 0; base < count; base += 32) {
    for (int lane = 0; lane < 32; ++lane) {
      r.v[0][lane] = fraction ? first | std::bit_cast<uint32_t>(float(base + lane) * 0x1p-24f)
                              : first + uint32_t(base + lane);
    }
    ASSERT_EQ(call(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 0xffffffffU, 0, r.v + 1,
                   r.v, nullptr),
              RJ_ARITH_SUCCESS);
    for (int lane = 0; lane < 32; ++lane) {
      digest = capture_hash_word(digest, r.v[1][lane]);
    }
  }
  EXPECT_EQ(digest, expected) << "first=" << first;
}

// Reuse the former gfx1100/gfx1201 capture digests without keeping a second implementation.
TEST(InheritedMathGoldens, CompleteDomains) {
  check_digest(rj_arith_v_rcp_f32_w32_gfx12, 0x3f800000, 1 << 23, 0xd54ec24992572df9ULL);
  check_digest(rj_arith_v_rsq_f32_w32_gfx12, 0x3f800000, 1 << 24, 0x010bc79eb6e48cafULL);
  check_digest(rj_arith_v_sqrt_f32_w32_gfx12, 0x3f800000, 1 << 24, 0x514d49d31f12176bULL);
  check_digest(rj_arith_v_log_f32_w32_gfx12, 0x3f000000, 1 << 24, 0xc33a54efbae098beULL);
  check_digest(rj_arith_v_exp_f32_w32_gfx12, 0, 1 << 24, 0x68959bb684f42e5fULL,
               /*fraction=*/true);
  check_digest(rj_arith_v_exp_f32_w32_gfx12, 0x80000000, 1 << 24, 0x88b75684494db8d3ULL,
               /*fraction=*/true);
}

} // namespace
