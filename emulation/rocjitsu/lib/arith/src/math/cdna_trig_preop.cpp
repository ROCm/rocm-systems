// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited CDNA TRIG_PREOP uses 1201 bits of 2/pi and RTZ extraction.
// Keep the original compact table; all scaling is integer arithmetic, without FP-control writes.
#include "math/cdna_trig_preop.h"
#include "common/bits.h"
#include "common/validation.h"
#include "fp/float_output.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_math.h"
#include <stdint.h>

namespace {
using namespace rocjitsu::arith;
constexpr uint32_t chunks[] = {
    0xA2F983u, 0x6E4E44u, 0x1529FCu, 0x2757D1u, 0xF534DDu, 0xC0DB62u, 0x95993Cu, 0x439041u,
    0xFE5163u, 0xABDEBBu, 0xC561B7u, 0x246E3Au, 0x424DD2u, 0xE00649u, 0x2EEA09u, 0xD1921Cu,
    0xFE1DEBu, 0x1CB129u, 0xA73EE8u, 0x8235F5u, 0x2EBB44u, 0x84E99Cu, 0x7026B4u, 0x5F7E41u,
    0x3991D6u, 0x398353u, 0x39F49Cu, 0x845F8Bu, 0xBDF928u, 0x3B1FF8u, 0x97FFDEu, 0x05980Fu,
    0xEF2F11u, 0x8B5A0Au, 0x6D1F6Du, 0x367ECFu, 0x27CB09u, 0xB74F46u, 0x3F669Eu, 0x5FEA2Du,
    0x7527BAu, 0xC7EBE5u, 0xF17B3Du, 0x0739F7u, 0x8A5292u, 0xEA6BFBu, 0x5FB11Fu, 0x8D5D08u,
    0x560330u, 0x46FC7Bu, 0x6BABF0u};
static_assert(sizeof(chunks) == 204);

// Read a zero-padded 24-bit chunk of the inherited 2/pi expansion.
uint64_t chunk(int index) { return index < 51 ? chunks[index] : 0; }

// Extract 53 significant bits from the 1201-bit table, without a bit-at-a-time loop.
uint64_t segment(int shift) {
  if (shift >= 1201) {
    return 0;
  }
  const int index = shift / 24, offset = shift % 24;
  const uint64_t high = (chunk(index) << 40) | (chunk(index + 1) << 16) | (chunk(index + 2) >> 8);
  const uint64_t low = ((chunk(index + 2) & 255) << 56) | (chunk(index + 3) << 32);
  uint64_t value = ((high << offset) | (offset ? low >> (64 - offset) : 0)) >> 11;
  const int remaining = 1201 - shift;
  if (remaining < 53) {
    value &= ~((1ULL << (53 - remaining)) - 1);
  }
  return value;
}

// Encode an unsigned significand times a power of two, truncating tiny FP64 results.
uint64_t encode(uint64_t value, int scale) {
  if (!value) {
    return 0;
  }
  const int leading = bit_width(value) - 1;
  const int exponent = leading + scale;
  if (exponent >= -1022) {
    return (uint64_t(exponent + 1023) << 52) | ((value << (52 - leading)) & 0xfffffffffffffULL);
  }
  const int adjustment = scale + 1074;
  return adjustment >= 0 ? value << adjustment : adjustment > -64 ? value >> -adjustment : 0;
}

// Extract and scale pieces for a whole wave before publishing either destination VGPR.
template <int Lanes, bool Gfx1250, typename Mask>
int run(uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d, const uint32_t *const *a,
        const uint32_t *const *b) {
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  constexpr uint64_t controls =
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP;
  constexpr uint64_t fp = RJ_ARITH_FP_ROUND_MASK | RJ_ARITH_FP_IEEE_MODE |
                          RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                          RJ_ARITH_FP_DX10_CLAMP_OFF;
  if (int error = validate(flags, mode & ~controls, fp)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  int omod = int((mode >> 6) & 3);
  if (!Gfx1250 &&
      (!(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (flags & RJ_ARITH_FP_IEEE_MODE))) {
    omod = 0;
  }
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);

#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    cdna_preop_x86_64_v4(Lanes, a[1], b[0], chunks, omod, rounding, mode & RJ_ARITH_ALU_CLAMP,
                         exec_mask, d);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    cdna_preop_x86_64_v3(Lanes, a[1], b[0], chunks, omod, rounding, mode & RJ_ARITH_ALU_CLAMP,
                         exec_mask, d);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint64_t result[Lanes]{};
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!(exec_mask & (Mask(1) << lane))) {
      continue;
    }
    const int exponent = int((a[1][lane] >> 20) & 2047);
    const int shift = 53 * int(b[0][lane] & 31) + maximum(exponent - 1077, 0);
    uint64_t value = encode(segment(shift), -53 - shift + (exponent >= 1968 ? 128 : 0));
    if (omod) {
      // Legacy CDNA scales first: a tiny extracted value can become normal.
      value = division_scale_bits<64>(value, omod == 3 ? -1 : omod, rounding);
      if (value < 0x10000000000000ULL) {
        value = 0;
      }
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      value = minimum(value, uint64_t(0x3ff0000000000000ULL));
    }
    result[lane] = value;
  }
  for (int lane = 0; lane < Lanes; ++lane) {
    if (exec_mask & (Mask(1) << lane)) {
      d[0][lane] = uint32_t(result[lane]);
      d[1][lane] = uint32_t(result[lane] >> 32);
    }
  }
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Evaluate inherited CDNA range-reduction pieces; FP status is unused.
int rj_arith_v_trig_preop_f64_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         uint32_t * /*state*/) {
  return run<32, false>(flags, exec_mask, instruction_flags, d, a, b);
}

// Evaluate inherited CDNA range-reduction pieces; FP status is unused.
int rj_arith_v_trig_preop_f64_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         uint32_t * /*state*/) {
  return run<64, false>(flags, exec_mask, instruction_flags, d, a, b);
}

// Evaluate inherited CDNA range-reduction pieces; FP status is unused.
int rj_arith_v_trig_preop_f64_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                          uint64_t instruction_flags, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          uint32_t * /*state*/) {
  return run<32, true>(flags, exec_mask, instruction_flags, d, a, b);
}
