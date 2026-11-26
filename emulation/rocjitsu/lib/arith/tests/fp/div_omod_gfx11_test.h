// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Replay of captured gfx1100 lanes with active output modifiers through the RDNA3 division helpers.
#pragma once

#include "common/cpu.h"
#include "common/registers.h"
#include "common/wave.h"
#include "fp/captured_gfx11_test.h"
#include "fp/reference.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>

namespace rocjitsu::arith::test {

// One captured lane of a division helper. mode is the hardware MODE register as in CapturedGfx11
// (rounding 3:0, denormals 7:4, DX10_CLAMP bit 8, IEEE bit 9) plus FP16_OVFL in bit 23. For
// DIV_FMAS, condition is the lane's VCC bit; for DIV_SCALE it is the expected condition output.
struct DivOmodGfx11 {
  const char *name;
  uint32_t mode;
  uint64_t a, b, c, expected;
  uint32_t modifiers;
  bool condition = false;
};

// Translate MODE into guest flags, including FP16_OVFL for 16-bit results.
template <int Width> uint64_t gfx11_division_flags(uint32_t mode) {
  uint64_t flags = gfx11_guest_flags<Width>(mode);
  if (Width == 16 && (mode & 0x800000U)) {
    flags |= RJ_ARITH_FP16_OVFL;
  }
  return flags;
}

// Each lane reproduces the capture in Compatibility and strict exact at every CPU level, with the
// same operands in every wave lane. Mode 3 keeps rejecting these entries (no RDNA3 status model),
// leaving the destination untouched.
template <int Lanes, int Width, typename Fn, size_t N>
void check_div_fmas_omod_gfx11(Fn fn, const DivOmodGfx11 (&lanes)[N]) {
  for (const DivOmodGfx11 &c : lanes) {
    SCOPED_TRACE(c.name);
    for (uint64_t cpu : supported_cpu_levels()) {
      for (uint64_t semantics : {RJ_ARITH_SEMANTICS_COMPATIBILITY,
                                 RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
        Registers<8, Lanes> r;
        for (int lane = 0; lane < Lanes; ++lane) {
          write_fp(r.v, lane, Width, c.a);
          write_fp(r.v + 2, lane, Width, c.b);
          write_fp(r.v + 4, lane, Width, c.c);
          write_fp(r.v + 6, lane, Width, Width == 32 ? 0xa5a5a5a5ULL : 0xa5a5a5a5a5a5a5a5ULL);
        }
        const uint64_t flags = cpu | semantics | gfx11_division_flags<Width>(c.mode);
        const WaveMask<Lanes> condition = c.condition ? ~WaveMask<Lanes>(0) : WaveMask<Lanes>(0);
        uint32_t status = 0x80000000U;
        ASSERT_EQ(fn(flags, ~WaveMask<Lanes>(0), c.modifiers, r.v + 6, r.v, r.v + 2, r.v + 4,
                     condition, &status),
                  RJ_ARITH_SUCCESS)
            << std::hex << cpu << '/' << semantics;
        EXPECT_EQ(status, 0x80000000U);
        for (int lane = 0; lane < Lanes; ++lane) {
          ASSERT_EQ(read_fp(r.v + 6, lane, Width), c.expected)
              << std::hex << cpu << '/' << semantics << " lane " << std::dec << lane;
        }
        const auto before = r;
        ASSERT_EQ(
            fn(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | gfx11_division_flags<Width>(c.mode),
               ~WaveMask<Lanes>(0), c.modifiers, r.v + 6, r.v, r.v + 2, r.v + 4, condition,
               &status),
            RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
        r.expect_preserved(before, 0, 0);
      }
    }
  }
}

// DIV_FIXUP_F16 lanes: the low half carries the result and the high half keeps the 0xa5a5 fill.
template <int Lanes, typename Fn, size_t N>
void check_div_fixup_f16_omod_gfx11(Fn fn, const DivOmodGfx11 (&lanes)[N]) {
  for (const DivOmodGfx11 &c : lanes) {
    SCOPED_TRACE(c.name);
    for (uint64_t cpu : supported_cpu_levels()) {
      for (uint64_t semantics : {RJ_ARITH_SEMANTICS_COMPATIBILITY,
                                 RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
        Registers<4, Lanes> r;
        for (int lane = 0; lane < Lanes; ++lane) {
          r.v[0][lane] = 0xdc3a0000U | uint32_t(c.a);
          r.v[1][lane] = 0x6a9e0000U | uint32_t(c.b);
          r.v[2][lane] = 0x96e90000U | uint32_t(c.c);
          r.v[3][lane] = 0xa5a5a5a5U;
        }
        const uint64_t flags = cpu | semantics | gfx11_division_flags<16>(c.mode);
        uint32_t status = 0x80000000U;
        ASSERT_EQ(
            fn(flags, ~WaveMask<Lanes>(0), c.modifiers, r.v + 3, r.v, r.v + 1, r.v + 2, &status),
            RJ_ARITH_SUCCESS)
            << std::hex << cpu << '/' << semantics;
        EXPECT_EQ(status, 0x80000000U);
        for (int lane = 0; lane < Lanes; ++lane) {
          ASSERT_EQ(r.v[3][lane], 0xa5a50000U | uint32_t(c.expected))
              << std::hex << cpu << '/' << semantics << " lane " << std::dec << lane;
        }
        const auto before = r;
        ASSERT_EQ(
            fn(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | gfx11_division_flags<16>(c.mode),
               ~WaveMask<Lanes>(0), c.modifiers, r.v + 3, r.v, r.v + 1, r.v + 2, &status),
            RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
        r.expect_preserved(before, 0, 0);
      }
    }
  }
}

// DIV_SCALE lanes: the condition output must match the capture in every lane; mode 3 has no
// status output and succeeds like mode 2.
template <int Lanes, int Width, typename Fn, size_t N>
void check_div_scale_omod_gfx11(Fn fn, const DivOmodGfx11 (&lanes)[N]) {
  for (const DivOmodGfx11 &c : lanes) {
    SCOPED_TRACE(c.name);
    for (uint64_t cpu : supported_cpu_levels()) {
      for (uint64_t semantics :
           {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
            RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
        Registers<8, Lanes> r;
        for (int lane = 0; lane < Lanes; ++lane) {
          write_fp(r.v, lane, Width, c.a);
          write_fp(r.v + 2, lane, Width, c.b);
          write_fp(r.v + 4, lane, Width, c.c);
          write_fp(r.v + 6, lane, Width, Width == 32 ? 0xa5a5a5a5ULL : 0xa5a5a5a5a5a5a5a5ULL);
        }
        const uint64_t flags = cpu | semantics | gfx11_division_flags<Width>(c.mode);
        WaveMask<Lanes> condition = 0;
        ASSERT_EQ(
            fn(flags, ~WaveMask<Lanes>(0), c.modifiers, r.v + 6, &condition, r.v, r.v + 2, r.v + 4),
            RJ_ARITH_SUCCESS)
            << std::hex << cpu << '/' << semantics;
        EXPECT_EQ(condition, c.condition ? ~WaveMask<Lanes>(0) : WaveMask<Lanes>(0));
        for (int lane = 0; lane < Lanes; ++lane) {
          ASSERT_EQ(read_fp(r.v + 6, lane, Width), c.expected)
              << std::hex << cpu << '/' << semantics << " lane " << std::dec << lane;
        }
      }
    }
  }
}

} // namespace rocjitsu::arith::test
