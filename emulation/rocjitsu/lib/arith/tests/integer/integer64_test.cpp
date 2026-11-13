// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Wide integer arithmetic: overflow, masked shifts, selected lanes and overlapping VGPR pairs.
#include "common/word_test.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_integer.h"

#include <algorithm>
#include <gtest/gtest.h>

namespace {
enum class Op { Add, Sub, Mul, MinU, MaxU, MinI, MaxI, ShiftAdd };

// Independent wide-integer reference; extended arithmetic makes add overflow explicit.
uint64_t reference(Op op, uint64_t a, uint64_t b, uint64_t c, bool clamp) {
  switch (op) {
  case Op::Add: {
    __uint128_t sum = (__uint128_t)a + b;
    return clamp && sum > UINT64_MAX ? UINT64_MAX : uint64_t(sum);
  }
  case Op::Sub:
    return clamp && b > a ? 0 : a - b;
  case Op::Mul:
    return uint64_t((__uint128_t)a * b);
  case Op::MinU:
    return std::min(a, b);
  case Op::MaxU:
    return std::max(a, b);
  case Op::MinI:
    return uint64_t(std::min(int64_t(a), int64_t(b)));
  case Op::MaxI:
    return uint64_t(std::max(int64_t(a), int64_t(b)));
  case Op::ShiftAdd:
    // ISA decoding table, including the three encodings that mean no shift.
    constexpr int shifts[] = {0, 1, 2, 3, 4, 0, 0, 0};
    return (a << shifts[b % 8]) + c;
  }
  return 0;
}

// Exercise all available CPUs and semantics with shared deterministic data and register guards.
template <Op Operation, int Lanes, typename Call> void check(Call call) {
  using namespace rocjitsu::arith::test;
  using Mask = WaveMask<Lanes>;
  const auto input = make_integer_inputs<Lanes>();
  auto invoke = [&](uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d,
                    const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
    if constexpr (Operation == Op::ShiftAdd) {
      return call(flags, exec_mask, mode, d, a, b, c);
    } else {
      return call(flags, exec_mask, mode, d, a, b);
    }
  };
  expect_alu_invalid_flags<false>(
      [&](uint64_t flags, uint64_t mode) {
        auto r = input;
        int status = invoke(flags, ~Mask(0), mode, r.v + 8, r.v, r.v + 2, r.v + 4);
        expect_integer_registers(r, input);
        EXPECT_EQ(invoke(flags, ~Mask(0), mode, nullptr, nullptr, nullptr, nullptr),
                  RJ_ARITH_ERROR_INVALID_FLAGS);
        return status;
      },
      RJ_ARITH_ALU_CLAMP);
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (int scenario = 0; scenario < 7; ++scenario) {
        auto exec_mask = exec_mask_for_case<Lanes>(scenario < 5 ? scenario : 3);
        uint64_t mode = scenario >= 5 ? RJ_ARITH_ALU_CLAMP : 0;
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << scenario);
        for (int alias : {8, 0, 2, 4, -1}) {
          auto r = input, expected = input;
          int low = alias < 0 ? 0 : alias, high = alias < 0 ? 0 : alias + 1;
          for (int lane = 0; lane < Lanes; ++lane) {
            if (!(exec_mask >> lane & 1)) {
              continue;
            }
            auto value = [&](int reg) {
              return input.v[reg][lane] | (uint64_t(input.v[reg + 1][lane]) << 32);
            };
            uint64_t result = reference(Operation, value(0), value(2), value(4), mode != 0);
            expected.v[low][lane] = uint32_t(result);
            expected.v[high][lane] = uint32_t(result >> 32);
          }
          uint32_t *dest[] = {r.v[low], r.v[high]};
          ASSERT_EQ(invoke(cpu | semantics, exec_mask, mode, exec_mask ? dest : nullptr,
                           exec_mask ? r.v : nullptr, exec_mask ? r.v + 2 : nullptr,
                           exec_mask ? r.v + 4 : nullptr),
                    RJ_ARITH_SUCCESS);
          expect_integer_registers(r, expected);
        }
      }
    }
  }
}
} // namespace

TEST(Integer64AddW32, Arithmetic) { check<Op::Add, 32>(rj_arith_v_add_nc_u64_w32); }
TEST(Integer64AddW64, Arithmetic) { check<Op::Add, 64>(rj_arith_v_add_nc_u64_w64); }
TEST(Integer64SubW32, Arithmetic) { check<Op::Sub, 32>(rj_arith_v_sub_nc_u64_w32); }
TEST(Integer64SubW64, Arithmetic) { check<Op::Sub, 64>(rj_arith_v_sub_nc_u64_w64); }
TEST(Integer64MulW32, Arithmetic) { check<Op::Mul, 32>(rj_arith_v_mul_u64_w32); }
TEST(Integer64MulW64, Arithmetic) { check<Op::Mul, 64>(rj_arith_v_mul_u64_w64); }
TEST(Integer64MinUW32, Arithmetic) { check<Op::MinU, 32>(rj_arith_v_min_u64_w32); }
TEST(Integer64MinUW64, Arithmetic) { check<Op::MinU, 64>(rj_arith_v_min_u64_w64); }
TEST(Integer64MaxUW32, Arithmetic) { check<Op::MaxU, 32>(rj_arith_v_max_u64_w32); }
TEST(Integer64MaxUW64, Arithmetic) { check<Op::MaxU, 64>(rj_arith_v_max_u64_w64); }
TEST(Integer64MinIW32, Arithmetic) { check<Op::MinI, 32>(rj_arith_v_min_i64_w32); }
TEST(Integer64MinIW64, Arithmetic) { check<Op::MinI, 64>(rj_arith_v_min_i64_w64); }
TEST(Integer64MaxIW32, Arithmetic) { check<Op::MaxI, 32>(rj_arith_v_max_i64_w32); }
TEST(Integer64MaxIW64, Arithmetic) { check<Op::MaxI, 64>(rj_arith_v_max_i64_w64); }
TEST(Integer64ShiftAddW32, Arithmetic) { check<Op::ShiftAdd, 32>(rj_arith_v_lshl_add_u64_w32); }
TEST(Integer64ShiftAddW64, Arithmetic) { check<Op::ShiftAdd, 64>(rj_arith_v_lshl_add_u64_w64); }
