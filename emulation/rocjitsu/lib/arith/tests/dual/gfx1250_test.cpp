// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 dual slots retain ordinary arithmetic, with encoding-specific modifiers.

#include "common/cpu.h"
#include "common/integer.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_dual.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include "rj_arith/detail/rj_arith_integer.h"
#include "rj_arith/detail/rj_arith_register.h"

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

// The ordinary instruction is the inherited model. Check forwarding, masked
// two-register writes, partial overlap and status without duplicating its oracle.
template <typename Dual, typename Ordinary>
void check(Dual dual, Ordinary ordinary,
           uint64_t supported_modifiers = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B) {
  using namespace rocjitsu::arith::test;
  Registers<8, 32> input;
  std::minstd_rand random;
  for (int reg = 0; reg < 8; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      input.v[reg][lane] = random_word(random);
    }
  }
  input.v[0][2] = 0x7fc12345U;
  input.v[2][3] = 0x7f800000U;
  input.v[0][0] = 1;
  input.v[1][0] = 0x7ff80000U;
  input.v[2][1] = 0;
  input.v[3][1] = 0x7ff00000U;
  for (auto cpu : supported_cpu_levels()) {
    for (auto semantics : semantics_modes()) {
      for (int aliasing_offset : {0, 1, 6}) {
        for (uint64_t modifiers : {uint64_t{0}, supported_modifiers}) {
          SCOPED_TRACE(::testing::Message()
                       << cpu << '/' << semantics << '/' << aliasing_offset << '/' << modifiers);
          auto actual = input, expected = input;
          uint32_t actual_status = 0x80000000U, expected_status = actual_status;
          const uint64_t flags = cpu | semantics;
          const uint32_t exec_mask = 0x55555555U;
          const int wanted = ordinary(flags, exec_mask, modifiers, expected.v + aliasing_offset,
                                      expected.v, expected.v + 2, expected.v + 4, &expected_status);
          EXPECT_EQ(dual(flags, exec_mask, modifiers, actual.v + aliasing_offset, actual.v,
                         actual.v + 2, actual.v + 4, &actual_status),
                    wanted);
          EXPECT_EQ(actual_status, expected_status);
          for (int reg = 0; reg < 8; ++reg) {
            for (int lane = 0; lane < 32; ++lane) {
              EXPECT_EQ(actual.v[reg][lane], expected.v[reg][lane]);
            }
          }
        }
      }
    }
  }
}

// Adapt the binary shape to the shared ternary forwarding check.
template <typename Function> auto binary(Function function) {
  return [function](auto flags, auto exec_mask, auto modifiers, auto d, auto a, auto b, auto,
                    auto state) { return function(flags, exec_mask, modifiers, d, a, b, state); };
}

} // namespace

TEST(DualWideGfx1250, Add) {
  check(binary(rj_arith_v_dual_add_f64_w32_gfx1250), binary(rj_arith_v_add_f64_w32_gfx12));
}
TEST(DualWideGfx1250, Mul) {
  check(binary(rj_arith_v_dual_mul_f64_w32_gfx1250), binary(rj_arith_v_mul_f64_w32_gfx1250));
}
TEST(DualWideGfx1250, MinNum) {
  check(binary(rj_arith_v_dual_min_num_f64_w32_gfx1250), binary(rj_arith_v_min_num_f64_w32));
}
TEST(DualWideGfx1250, MaxNum) {
  check(binary(rj_arith_v_dual_max_num_f64_w32_gfx1250), binary(rj_arith_v_max_num_f64_w32));
}
TEST(DualWideGfx1250, Fma) {
  check(rj_arith_v_dual_fma_f64_w32_gfx1250, rj_arith_v_fma_f64_w32_gfx1250);
}

TEST(DualGfx1250, AddF32) {
  check(binary(rj_arith_v_dual_add_f32_w32_gfx1250), binary(rj_arith_v_add_f32_w32_gfx12));
}

TEST(DualGfx1250, SubF32) {
  check(binary(rj_arith_v_dual_sub_f32_w32_gfx1250), binary(rj_arith_v_sub_f32_w32_gfx12));
}

TEST(DualGfx1250, SubrevF32) {
  check(binary(rj_arith_v_dual_subrev_f32_w32_gfx1250), binary(rj_arith_v_subrev_f32_w32_gfx12));
}

TEST(DualGfx1250, MulF32) {
  auto ordinary = [](auto flags, auto exec_mask, auto modifiers, auto d, auto a, auto b,
                     auto status) {
    if ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
      return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
    }
    return rj_arith_v_mul_f32_w32_gfx12(flags, exec_mask, modifiers, d, a, b, status);
  };
  check(binary(rj_arith_v_dual_mul_f32_w32_gfx1250), binary(ordinary));
}

TEST(DualGfx1250, MulDx9ZeroF32) {
  check(binary(rj_arith_v_dual_mul_dx9_zero_f32_w32_gfx1250),
        binary(rj_arith_v_mul_dx9_zero_f32_w32_gfx1250));
}

TEST(DualGfx1250, MinNumF32) {
  check(binary(rj_arith_v_dual_min_num_f32_w32_gfx1250), binary(rj_arith_v_min_num_f32_w32));
}

TEST(DualGfx1250, MaxNumF32) {
  check(binary(rj_arith_v_dual_max_num_f32_w32_gfx1250), binary(rj_arith_v_max_num_f32_w32));
}

TEST(DualGfx1250, FmaF32) {
  check(rj_arith_v_dual_fma_f32_w32_gfx1250, rj_arith_v_fma_f32_w32_gfx12);
}

TEST(DualGfx1250, FmacF32) {
  auto adapt = [](auto function) {
    return [function](auto f, auto e, auto m, auto d, auto a, auto b, [[maybe_unused]] auto c,
                      auto state) { return function(f, e, m, d, a, b, state); };
  };
  check(adapt(rj_arith_v_dual_fmac_f32_w32_gfx1250), adapt(rj_arith_v_fmac_f32_w32_gfx12));
}

TEST(DualGfx1250, SubNcU32) {
  auto adapt = [](auto function) {
    return [function](auto f, auto e, auto m, auto d, auto a, auto b, auto, auto) {
      return function(f, e, m, d, a, b);
    };
  };
  check(adapt(rj_arith_v_dual_sub_nc_u32_w32), adapt(rj_arith_v_sub_nc_u32_w32), 0);
}

TEST(DualGfx1250, LshrrevB32) {
  auto adapt = [](auto function) {
    return [function](auto f, auto e, auto m, auto d, auto a, auto b, auto, auto) {
      return function(f, e, m, d, a, b);
    };
  };
  check(adapt(rj_arith_v_dual_lshrrev_b32_w32), adapt(rj_arith_v_lshrrev_b32_w32), 0);
}

TEST(DualGfx1250, AshrrevI32) {
  auto adapt = [](auto function) {
    return [function](auto f, auto e, auto m, auto d, auto a, auto b, auto, auto) {
      return function(f, e, m, d, a, b);
    };
  };
  check(adapt(rj_arith_v_dual_ashrrev_i32_w32), adapt(rj_arith_v_ashrrev_i32_w32), 0);
}

TEST(DualGfx1250, MinI32) {
  auto adapt = [](auto function) {
    return [function](auto f, auto e, auto m, auto d, auto a, auto b, auto, auto) {
      return function(f, e, m, d, a, b);
    };
  };
  check(adapt(rj_arith_v_dual_min_i32_w32), adapt(rj_arith_v_min_i32_w32), 0);
}

TEST(DualGfx1250, MaxI32) {
  auto adapt = [](auto function) {
    return [function](auto f, auto e, auto m, auto d, auto a, auto b, auto, auto) {
      return function(f, e, m, d, a, b);
    };
  };
  check(adapt(rj_arith_v_dual_max_i32_w32), adapt(rj_arith_v_max_i32_w32), 0);
}

TEST(DualGfx1250, Bitop2) {
  check([](auto f, auto e, auto m, auto d, auto a, auto b, auto,
           auto) { return rj_arith_v_dual_bitop2_b32_w32(f, e, m, d, a, b); },
        [](auto f, auto e, auto m, auto d, auto a, auto b, auto, auto) {
          const uint32_t zero[32] = {};
          const uint32_t *c = zero;
          return rj_arith_v_bitop3_b32_w32(f, e, m, d, a, b, &c);
        },
        0x96);
}

TEST(DualGfx1250, Cndmask) {
  auto adapt = [](auto function) {
    return [function](auto f, auto e, auto m, auto d, auto a, auto b, auto, auto) {
      return function(f, e, m, d, a, b, 0xccccccccU);
    };
  };
  check(adapt(rj_arith_v_dual_cndmask_b32_w32_gfx1250), adapt(rj_arith_v_cndmask_b32_w32));
}
