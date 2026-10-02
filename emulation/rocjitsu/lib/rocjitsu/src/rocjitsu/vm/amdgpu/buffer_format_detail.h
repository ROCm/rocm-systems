// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/buffer_format.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"

#include <algorithm>
#include <bit>
#include <cmath>

namespace rocjitsu::amdgpu::buffer_format_detail {

using Number = BufferNumberFormat;
using Format = BufferFormat;

// Shared conversions for formatted buffer and image loads. The caller owns
// the fixed-function FP environment for the whole instruction, not each texel.
inline uint32_t mask(uint32_t bits) { return bits == 32 ? ~0u : (1u << bits) - 1; }
inline bool integer(Number n) { return n == Number::Uint || n == Number::Sint; }

// Independent of the host floating-point rounding mode.
inline double round_even(double value) {
  const double lo = std::floor(value);
  const double fraction = value - lo;
  return lo + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lo, 2.0) != 0));
}

inline uint32_t read_bits(std::span<const uint8_t> bytes, uint32_t offset, uint32_t width) {
  uint64_t value = 0;
  for (uint32_t i = offset / 8; i < (offset + width + 7) / 8; ++i)
    value |= uint64_t{bytes[i]} << ((i - offset / 8) * 8);
  return static_cast<uint32_t>(value >> (offset % 8)) & mask(width);
}

inline uint32_t unpack(uint32_t value, uint32_t width, Number n) {
  const int32_t signed_value = static_cast<int32_t>(value << (32 - width)) >> (32 - width);
  if (n == Number::Uint || (n == Number::Float && width == 32))
    return value;
  if (n == Number::Sint)
    return static_cast<uint32_t>(signed_value);
  float result = 0;
  switch (n) {
  case Number::Unorm:
    result = static_cast<float>(value) / mask(width);
    break;
  case Number::Snorm:
    result = std::max(-1.0f, static_cast<float>(signed_value) / mask(width - 1));
    break;
  case Number::Uscaled:
    result = static_cast<float>(value);
    break;
  case Number::Sscaled:
    result = static_cast<float>(signed_value);
    break;
  case Number::Float:
    if (width == 16)
      return std::bit_cast<uint32_t>(util::f16_to_f32(static_cast<uint16_t>(value)));
    // Unsigned 10/11-bit floats have five exponent bits and bias 15.
    {
      const uint32_t mantissa_bits = width - 5;
      const uint32_t exponent = value >> mantissa_bits;
      const uint32_t mantissa = value & mask(mantissa_bits);
      if (exponent == 31)
        return 0x7f800000u | (mantissa << (23 - mantissa_bits));
      result = std::ldexp(static_cast<float>(mantissa + (exponent ? 1u << mantissa_bits : 0)),
                          (exponent ? static_cast<int>(exponent) - 15 : -14) -
                              static_cast<int>(mantissa_bits));
    }
    break;
  default:
    break;
  }
  return std::bit_cast<uint32_t>(result);
}

inline std::array<uint32_t, 4> unpack_format(const Format &f, uint32_t selectors,
                                             std::span<const uint8_t> bytes) {
  std::array<uint32_t, 4> channels{}, result{};
  uint32_t offset = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    if (f.widths[i] && !bytes.empty())
      channels[i] = unpack(read_bits(bytes, offset, f.widths[i]), f.widths[i], f.number);
    offset += f.widths[i];
  }
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t sel = (selectors >> (3 * i)) & 7;
    result[i] = sel >= 4   ? channels[sel - 4]
                : sel == 1 ? (integer(f.number) ? 1u : 0x3f800000u)
                           : 0;
  }
  return result;
}

inline void write_format_load_lane(Wavefront &wf, ComputeUnitCore &cu, const VectorMemState &d,
                                   uint32_t lane, const std::array<uint32_t, 4> &values) {
  const uint32_t registers = d.buffer_d16 ? (d.buffer_components + 1) / 2 : d.buffer_components;
  const auto &format = d.decoded_buffer_format;
  for (uint32_t reg = 0; reg < registers; ++reg) {
    if (!d.buffer_d16) {
      if (d.lds_dst)
        wf.lds().write(d.lds_base + (lane * registers + reg) * 4,
                       reinterpret_cast<const uint8_t *>(&values[reg]), 4);
      else
        cu.write_vgpr(d.dst_reg_base + reg, lane, values[reg]);
      continue;
    }
    uint32_t packed = 0, write_mask = 0;
    for (uint32_t i = reg * 2; i < std::min(reg * 2 + 2, d.buffer_components); ++i) {
      const uint32_t shift = d.d16_hi ? 16 : (i % 2) * 16;
      uint16_t half = static_cast<uint16_t>(values[i]);
      if (!integer(format.number)) {
        const float value = std::bit_cast<float>(values[i]);
        // Only FLOAT32 -> D16 truncates. Other conversions round to nearest even.
        half = format.number == Number::Float && format.widths[0] == 32
                   ? util::f32_to_f16_rtz(value)
                   : util::f32_to_f16(value);
      }
      packed |= uint32_t{half} << shift;
      write_mask |= 0xffffu << shift;
    }
    if (write_mask != ~0u && !cu.sram_ecc() && !d.lds_dst)
      packed |= cu.read_vgpr_storage(d.dst_reg_base + reg, lane) & ~write_mask;
    if (d.lds_dst)
      wf.lds().write(d.lds_base + (lane * registers + reg) * 4,
                     reinterpret_cast<const uint8_t *>(&packed), 4);
    else
      cu.write_vgpr(d.dst_reg_base + reg, lane, packed);
  }
}

} // namespace rocjitsu::amdgpu::buffer_format_detail
