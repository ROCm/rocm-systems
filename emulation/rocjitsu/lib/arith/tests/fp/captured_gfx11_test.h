// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Replay of decoded gfx1100 lanes through unary, binary and ternary entry points.
#pragma once

#include "common/cpu.h"
#include "common/registers.h"
#include "common/wave.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <gtest/gtest.h>
#include <initializer_list>
#include <stddef.h>
#include <stdint.h>
#include <type_traits>

namespace rocjitsu::arith::test {

// One captured lane. mode is the hardware MODE register: rounding in bits 3:0, denormals in bits
// 7:4, DX10_CLAMP in bit 8, IEEE in bit 9 and FP16_OVFL in bit 23. Sources and result are raw
// register values; a 16-bit result includes the destination's untouched half, which the capture
// filled with 0xa5. gfx1201 lanes use the same layout without DX10_CLAMP.
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
  if (!(mode & 0x100)) {
    flags |= RJ_ARITH_FP_DX10_CLAMP_OFF;
  }
  if (!(mode & (0x10U << shift))) {
    flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  }
  if (!(mode & (0x20U << shift))) {
    flags |= RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  }
  if (mode & 0x200) {
    flags |= RJ_ARITH_FP_IEEE_MODE;
  }
  if (mode & 0x800000) {
    flags |= RJ_ARITH_FP16_OVFL;
  }
  return flags;
}

// Translate a gfx1201 MODE word into guest flags: gfx12 has no DX10_CLAMP control, so bit 8 is
// ignored and NaN clamping is unconditional.
template <int Width> uint64_t gfx12_guest_flags(uint32_t mode) {
  return gfx11_guest_flags<Width>(mode | 0x100);
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

// Each requested semantics reproduces each lane in every wave lane at every CPU level. Width
// selects the MODE fields; 64-bit values occupy two registers, low word first. Gfx12 lanes
// translate MODE without the DX10_CLAMP control.
template <int Lanes, int Width, bool Gfx12 = false, typename Fn, size_t N>
void check_captured_gfx11(Fn fn, const CapturedGfx11 (&lanes)[N],
                          std::initializer_list<uint64_t> semantics = {
                              RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
  for (const CapturedGfx11 &c : lanes) {
    SCOPED_TRACE(c.name);
    for (uint64_t cpu : supported_cpu_levels()) {
      for (uint64_t mode : semantics) {
        Registers<8, Lanes> r;
        for (int lane = 0; lane < Lanes; ++lane) {
          int reg = 0;
          for (uint64_t value : {c.a, c.b, c.c, uint64_t(0xa5a5a5a5a5a5a5a5ULL)}) {
            r.v[reg++][lane] = uint32_t(value);
            r.v[reg++][lane] = uint32_t(value >> 32);
          }
        }
        const uint64_t guest =
            Gfx12 ? gfx12_guest_flags<Width>(c.mode) : gfx11_guest_flags<Width>(c.mode);
        ASSERT_EQ(call_sources(fn, cpu | mode | guest, ~WaveMask<Lanes>(0), c.modifiers, r.v + 6,
                               r.v, r.v + 2, r.v + 4),
                  RJ_ARITH_SUCCESS)
            << cpu << '/' << mode;
        for (int lane = 0; lane < Lanes; ++lane) {
          uint64_t result = r.v[6][lane];
          if constexpr (Width == 64) {
            result |= uint64_t(r.v[7][lane]) << 32;
          }
          ASSERT_EQ(result, c.expected) << cpu << '/' << mode << '/' << lane;
        }
      }
    }
  }
}

// Entry points whose Compatibility resolves to strict exact must reproduce captured lanes in
// both; a faster loose path cannot then change default-mode results.
template <int Lanes, int Width, typename Fn, size_t N>
void check_captured_gfx11_exact(Fn fn, const CapturedGfx11 (&lanes)[N]) {
  check_captured_gfx11<Lanes, Width>(
      fn, lanes,
      {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS});
}

// The gfx1201 form of check_captured_gfx11_exact.
template <int Lanes, int Width, typename Fn, size_t N>
void check_captured_gfx12_exact(Fn fn, const CapturedGfx11 (&lanes)[N]) {
  check_captured_gfx11<Lanes, Width, true>(
      fn, lanes,
      {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS});
}

} // namespace rocjitsu::arith::test
