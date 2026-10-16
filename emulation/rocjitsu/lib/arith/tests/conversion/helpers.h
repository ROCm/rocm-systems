// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Conversion test inputs, call shapes, modifiers, aliasing, and preservation checks.

#pragma once

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/dpp_reference.h"
#include "common/floating_point.h"
#include "common/integer.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "common/wave.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <type_traits>
#include <vector>

namespace rocjitsu::arith::test {

// A/B/C each have two slots, followed by D. Edge encodings supplement arbitrary
// high-bit draws, so narrow formats, rounding ties and saturation are not rare.
template <int Lanes> Registers<8, Lanes> conversion_inputs(int source_width) {
  Registers<8, Lanes> input;
  std::minstd_rand random;
  const uint32_t edge32[] = {0,           0x80000000U, 1,           0x80000001U,
                             0x3f000000U, 0xbf000000U, 0x3fc00000U, 0xbfc00000U,
                             0x4b800001U, 0x01000001U, 0x4f000000U, 0xcf000000U,
                             0x7f800000U, 0xff800000U, 0x7fc12345U, 0x7f800123U};
  const uint16_t edge16[] = {0,      0x8000, 1,      0x8001, 0x03ff, 0x0400, 0x3800, 0xb800,
                             0x3e00, 0xbe00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e01, 0x7c01};
  const uint64_t edge64[] = {0,
                             0x8000000000000000ULL,
                             1,
                             0x8000000000000001ULL,
                             0x3fe0000000000000ULL,
                             0xbfe0000000000000ULL,
                             0x3ff8000000000000ULL,
                             0xbff8000000000000ULL,
                             0x3ff0000010000000ULL,
                             0x3ff0000010000001ULL,
                             0x41e0000000000000ULL,
                             0xc1e0000000000000ULL,
                             0x7ff0000000000000ULL,
                             0xfff0000000000000ULL,
                             0x7ff8123456789abcULL,
                             0x7ff0000000000123ULL};
  for (int reg = 0; reg < 8; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      input.v[reg][lane] = random_word(random);
    }
  }
  // Rotate edge encodings between operands so exceptional values do not always coincide.
  for (int reg : {0, 2, 4}) {
    for (int lane = 0; lane < 16; ++lane) {
      int index = (lane + reg * 3) % 16;
      if (source_width == 64) {
        input.v[reg][lane] = uint32_t(edge64[index]);
        input.v[reg + 1][lane] = uint32_t(edge64[index] >> 32);
      } else if (source_width == 16) {
        input.v[reg][lane] = edge16[index] | (uint32_t(edge16[(index + 5) % 16]) << 16);
      } else {
        input.v[reg][lane] = edge32[index];
      }
    }
  }
  return input;
}

// Enumerate individual/combined supported modifiers and optional DPP cases.
inline std::vector<uint64_t> conversion_modes(uint64_t supported, bool dpp) {
  std::vector<uint64_t> result{0};
  for (int bit = 0; bit < 32; ++bit) {
    if (supported & (1ULL << bit)) {
      result.push_back(1ULL << bit);
    }
  }
  if (supported) {
    result.push_back(supported);
  }
  if (dpp) {
    for (uint64_t routing : dpp_modes) {
      result.push_back(routing | supported);
    }
  }
  return result;
}

// Apply the gfx11 guest-policy gate on output scaling to the requested mode.
inline uint32_t conversion_effective_mode(uint64_t flags, uint32_t mode, bool gfx12) {
  if (!gfx12 &&
      ((flags & RJ_ARITH_FP_IEEE_MODE) || !(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS))) {
    mode &= ~RJ_ARITH_ALU_OMOD_HALF;
  }
  return mode;
}

// The APIs differ only in operand count and presence of an FP-status output.
// Keep those call-shape details here, while family helpers specify support and
// independently compute the result.
template <class Call, class Mask, class... Sources>
int invoke_conversion(Call call, uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d,
                      uint32_t *status, Sources... sources) {
  if constexpr (std::is_invocable_v<Call, uint64_t, Mask, uint64_t, uint32_t *const *, Sources...,
                                    uint32_t *>) {
    return call(flags, exec_mask, mode, d, sources..., status);
  } else {
    return call(flags, exec_mask, mode, d, sources...);
  }
}

// A narrow conversion shares this memory/semantics contract with widening and
// packing. References return one or two words; comparators own numeric rules.
template <int Lanes, int Operands, int InputWords, int OutputWords, bool Dpp, class Call,
          class Reference, class Compare>
void check_conversion(Call call, const Registers<8, Lanes> &input, Reference reference,
                      Compare compare, uint64_t modifiers, bool exact, uint64_t guest_flags = 0) {
  using Mask = WaveMask<Lanes>;
  using Source = const uint32_t *const *;
  constexpr bool status_parameter =
      Operands == 1 ? std::is_invocable_v<Call, uint64_t, Mask, uint64_t, uint32_t *const *, Source,
                                          uint32_t *>
      : Operands == 2 ? std::is_invocable_v<Call, uint64_t, Mask, uint64_t, uint32_t *const *,
                                            Source, Source, uint32_t *>
                      : std::is_invocable_v<Call, uint64_t, Mask, uint64_t, uint32_t *const *,
                                            Source, Source, Source, uint32_t *>;
  auto invoke = [&](uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d, Source a,
                    Source b, Source c, uint32_t *status) {
    if constexpr (Operands == 1) {
      return invoke_conversion(call, flags, exec_mask, mode, d, status, a);
    } else if constexpr (Operands == 2) {
      return invoke_conversion(call, flags, exec_mask, mode, d, status, a, b);
    } else {
      return invoke_conversion(call, flags, exec_mask, mode, d, status, a, b, c);
    }
  };
  expect_alu_invalid_flags<Dpp>(
      [&](uint64_t flags, uint64_t mode) {
        auto r = input;
        uint32_t status = 0x80000055U;
        int error = invoke(flags, ~Mask(0), mode, r.v + 6, r.v, r.v + 2, r.v + 4, &status);
        r.expect_preserved(input, 0, 0);
        EXPECT_EQ(status, 0x80000055U);
        EXPECT_EQ(invoke(flags, ~Mask(0), mode, nullptr, nullptr, nullptr, nullptr, &status),
                  RJ_ARITH_ERROR_INVALID_FLAGS);
        return error;
      },
      modifiers);
  auto modes = conversion_modes(modifiers, Dpp);
  auto cpu_levels = supported_cpu_levels();
  // Independent samples of guest policies and modifiers, plus a full-mask
  // scaling witness under every guest policy (RDNA3 gates OMOD on these bits).
  int count = 5 + int(modes.size()) + (guest_flags & RJ_ARITH_FP_DX10_CLAMP_OFF ? 16 : 8);
  for (uint64_t cpu : cpu_levels) {
    for (uint64_t semantics : semantics_modes()) {
      for (int scenario = 0; scenario < count; ++scenario) {
        uint64_t mode = scenario < 5                       ? 0
                        : scenario < 5 + int(modes.size()) ? modes[scenario - 5]
                                                           : (modifiers & RJ_ARITH_ALU_OMOD_2);
        int guest =
            scenario >= 5 + int(modes.size()) ? scenario - 5 - int(modes.size()) : scenario % 8;
        if (guest >= 8) {
          mode |= modifiers & RJ_ARITH_ALU_CLAMP; // Exercise DX10 NaN preservation under CLAMP.
        }
        uint64_t policy = ((guest & 1 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0) |
                           (guest & 2 ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                           (guest & 4 ? RJ_ARITH_FP_IEEE_MODE : 0) |
                           (guest & 8 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0)) &
                          guest_flags;
        uint64_t flags = cpu | semantics | policy | (scenario % 2 ? RJ_ARITH_FP16_OVFL : 0);
        Mask exec_mask = exec_mask_for_case<Lanes>(scenario < 5                        ? scenario
                                                   : scenario >= 5 + int(modes.size()) ? 0
                                                   : scenario % 2                      ? 0
                                                                                       : 3);
        int error = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL && status_parameter
                        ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
                    : !exact && (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                                 semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL)
                        ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS
                        : RJ_ARITH_SUCCESS;
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << scenario);
        // Whole-source aliases and, for wide destinations, overlapping D words.
        for (int alias = 0; alias < Operands + 1 + (OutputWords == 2); ++alias) {
          int offset = alias == 0 ? 6 : alias > Operands ? 0 : 2 * (alias - 1);
          bool repeated = alias > Operands;
          auto r = input;
          uint32_t *d[] = {r.v[offset], r.v[offset + (repeated ? 0 : 1)]};
          // A nonzero sentinel detects unwanted writes when FP status is unused or reporting is
          // rejected.
          uint32_t status = 0x80000055U;
          uint32_t *status_ptr = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL || scenario % 2
                                     ? &status
                                     : nullptr;
          ASSERT_EQ(invoke(flags, exec_mask, mode, exec_mask ? d : nullptr,
                           exec_mask ? r.v : nullptr, exec_mask ? r.v + 2 : nullptr,
                           exec_mask ? r.v + 4 : nullptr, status_ptr),
                    error);
          EXPECT_EQ(status, 0x80000055U);
          if (error) {
            r.expect_preserved(input, 0, 0);
            continue;
          }
          r.expect_preserved(input, offset, repeated ? 1 : OutputWords);
          for (int lane = 0; lane < Lanes; ++lane) {
            int source = lane;
            bool write = mode & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16)
                             ? dpp_source(mode, exec_mask, lane, source)
                             : bool((exec_mask >> lane) & 1);
            if (!write) {
              for (int word = 0; word < (repeated ? 1 : OutputWords); ++word) {
                EXPECT_EQ(r.v[offset + word][lane], input.v[offset + word][lane]);
              }
              continue;
            }
            uint64_t a = source < 0 ? 0 : input.v[0][source];
            if constexpr (InputWords == 2) {
              if (source >= 0) {
                a |= uint64_t(input.v[1][source]) << 32;
              }
            }
            uint64_t want = reference(a, input.v[2][lane], input.v[4][lane], input.v[offset][lane],
                                      flags, uint32_t(mode));
            if (repeated) {
              // Only the high result survives; retain the family's NaN rules.
              compare((want & 0xffffffffULL) | (uint64_t(r.v[offset][lane]) << 32), want, flags,
                      uint32_t(mode));
            } else {
              uint64_t actual = r.v[offset][lane];
              if constexpr (OutputWords == 2) {
                actual |= uint64_t(r.v[offset + 1][lane]) << 32;
              }
              compare(actual, want, flags, uint32_t(mode));
            }
          }
        }
      }
    }
  }
}

// Compare raw conversion-result bits exactly.
inline void conversion_bits(uint64_t actual, uint64_t expected, uint64_t, uint32_t) {
  EXPECT_EQ(actual, expected);
}

// These conversions have an inherited correctly-rounded reference, hence zero
// tolerances for finite values. NaN payloads are unspecified in loose mode.
inline void conversion_float(uint64_t actual, uint64_t expected, int width) {
  double a = width == 64 ? bit_cast<double>(actual) : double(bit_cast<float>(uint32_t(actual)));
  double e = width == 64 ? bit_cast<double>(expected) : double(bit_cast<float>(uint32_t(expected)));
  if (std::isnan(e)) {
    EXPECT_TRUE(std::isnan(a));
    EXPECT_NE(actual & (width == 64 ? 1ULL << 51 : 1ULL << 22), 0ULL);
  } else if (std::isinf(e)) {
    EXPECT_EQ(a, e);
  } else {
    constexpr double atol = 0, rtol = 0;
    EXPECT_LE(std::abs(a - e), atol + rtol * std::abs(e));
    if (e == 0) {
      EXPECT_EQ(std::signbit(a), std::signbit(e));
    }
  }
}

} // namespace rocjitsu::arith::test
