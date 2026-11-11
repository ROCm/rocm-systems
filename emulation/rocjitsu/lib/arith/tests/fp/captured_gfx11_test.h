// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Replay of decoded gfx1100 lanes through unary, binary and ternary entry points.
#pragma once

#include "common/cpu.h"
#include "common/registers.h"
#include "common/wave.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>
#include <type_traits>

namespace rocjitsu::arith::test {

// One captured lane. mode is the hardware MODE register with DX10_CLAMP set: rounding in bits
// 3:0, denormals in bits 7:4 and IEEE in bit 9. Sources and result are raw register values; a
// 16-bit result includes the destination's untouched half, which the capture filled with 0xa5.
struct CapturedGfx11 {
  const char *name;
  uint64_t modifiers;
  uint32_t mode;
  uint64_t a, b, c, expected;
};

// Translate MODE into guest flags for an instruction whose FP format has Width bits.
template <int Width> uint64_t gfx11_guest_flags(uint32_t mode) {
  constexpr int shift = Width == 32 ? 0 : 2;
  uint64_t flags = uint64_t((mode >> shift) & 3) << 24;
  if (!(mode & (0x10U << shift))) {
    flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  }
  if (!(mode & (0x20U << shift))) {
    flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  }
  if (mode & 0x200) {
    flags |= RJ_ARITH_FP_IEEE_MODE;
  }
  return flags;
}

// Call a unary, binary or ternary entry point with its leading sources.
template <typename Fn, typename Mask>
int call_sources(Fn fn, uint64_t flags, Mask exec, uint64_t modifiers, uint32_t *const *d,
                 const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  using Out = uint32_t *const *;
  using In = const uint32_t *const *;
  if constexpr (std::is_invocable_v<Fn, uint64_t, Mask, uint64_t, Out, In, uint32_t *>) {
    return fn(flags, exec, modifiers, d, a, nullptr);
  } else if constexpr (std::is_invocable_v<Fn, uint64_t, Mask, uint64_t, Out, In, In, uint32_t *>) {
    return fn(flags, exec, modifiers, d, a, b, nullptr);
  } else {
    return fn(flags, exec, modifiers, d, a, b, c, nullptr);
  }
}

// Compatibility reproduces each lane in every wave lane at every CPU level. Width selects the
// MODE fields; 64-bit values occupy two registers, low word first.
template <int Lanes, int Width, typename Fn, size_t N>
void check_captured_gfx11(Fn fn, const CapturedGfx11 (&lanes)[N]) {
  for (const CapturedGfx11 &c : lanes) {
    SCOPED_TRACE(c.name);
    for (uint64_t cpu : supported_cpu_levels()) {
      Registers<8, Lanes> r;
      for (int lane = 0; lane < Lanes; ++lane) {
        int reg = 0;
        for (uint64_t value : {c.a, c.b, c.c, uint64_t(0xa5a5a5a5a5a5a5a5ULL)}) {
          r.v[reg++][lane] = uint32_t(value);
          r.v[reg++][lane] = uint32_t(value >> 32);
        }
      }
      ASSERT_EQ(call_sources(
                    fn, cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | gfx11_guest_flags<Width>(c.mode),
                    ~WaveMask<Lanes>(0), c.modifiers, r.v + 6, r.v, r.v + 2, r.v + 4),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < Lanes; ++lane) {
        uint64_t result = r.v[6][lane];
        if constexpr (Width == 64) {
          result |= uint64_t(r.v[7][lane]) << 32;
        }
        ASSERT_EQ(result, c.expected) << cpu << '/' << lane;
      }
    }
  }
}

} // namespace rocjitsu::arith::test
