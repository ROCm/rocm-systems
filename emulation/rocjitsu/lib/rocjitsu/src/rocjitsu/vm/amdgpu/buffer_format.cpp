// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/buffer_format.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>

namespace rocjitsu::amdgpu {
namespace {
enum class Number { Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Float };
struct Format {
  std::array<uint32_t, 4> widths;
  Number number;
};

Format decode(uint32_t id, BufferFormatEncoding encoding) {
  // Consecutive format groups use the same numeric encodings. Packed format
  // names list widths from the high bits; widths here run from R in the low bits.
  auto group = [](uint32_t n, uint32_t bits, uint32_t count) {
    Format f{{}, static_cast<Number>(n)};
    std::fill_n(f.widths.begin(), count, bits);
    return f;
  };
  if (encoding == BufferFormatEncoding::Gfx9) {
    constexpr std::array<std::array<uint32_t, 4>, 15> widths{{{},
                                                              {8},
                                                              {16},
                                                              {8, 8},
                                                              {32},
                                                              {16, 16},
                                                              {11, 11, 10},
                                                              {10, 11, 11},
                                                              {2, 10, 10, 10},
                                                              {10, 10, 10, 2},
                                                              {8, 8, 8, 8},
                                                              {32, 32},
                                                              {16, 16, 16, 16},
                                                              {32, 32, 32},
                                                              {32, 32, 32, 32}}};
    const uint32_t dfmt = id & 15, nfmt = id >> 4;
    if (!dfmt)
      return {{}, Number::Uint};
    if (dfmt >= widths.size() || nfmt > 7 || nfmt == 6)
      throw std::runtime_error("Unsupported GFX9 buffer data format");
    return {widths[dfmt], nfmt == 7 ? Number::Float : static_cast<Number>(nfmt)};
  }
  if (encoding == BufferFormatEncoding::Rdna1 || encoding == BufferFormatEncoding::Rdna2) {
    if (id > 77 || (encoding == BufferFormatEncoding::Rdna2 &&
                    ((id >= 30 && id <= 35) || (id >= 37 && id <= 42) || id == 46 || id == 47)))
      throw std::runtime_error("Unsupported GFX10 buffer data format");
    if (id >= 30 && id <= 36)
      return {{11, 11, 10, 0}, static_cast<Number>(id - 30)};
    if (id >= 37 && id <= 43)
      return {{10, 11, 11, 0}, static_cast<Number>(id - 37)};
    if (id >= 44 && id <= 49)
      return {{2, 10, 10, 10}, static_cast<Number>(id - 44)};
    // The remaining GFX10 formats have the same order as GFX11, shifted by 14.
    if (id >= 50)
      id -= 14;
  }
  if (id == 0)
    return {{}, Number::Uint};
  if (id <= 6)
    return group(id - 1, 8, 1);
  if (id <= 13)
    return group(id - 7, 16, 1);
  if (id <= 19)
    return group(id - 14, 8, 2);
  if (id <= 22)
    return group(id - 20 + 4, 32, 1);
  if (id <= 29)
    return group(id - 23, 16, 2);
  if (id == 30)
    return {{11, 11, 10, 0}, Number::Float};
  if (id == 31)
    return {{10, 11, 11, 0}, Number::Float};
  if (id <= 35)
    return {{2, 10, 10, 10}, static_cast<Number>(id - 32 + (id >= 34 ? 2 : 0))};
  if (id <= 41)
    return {{10, 10, 10, 2}, static_cast<Number>(id - 36)};
  if (id <= 47)
    return group(id - 42, 8, 4);
  if (id <= 50)
    return group(id - 48 + 4, 32, 2);
  if (id <= 57)
    return group(id - 51, 16, 4);
  if (id <= 60)
    return group(id - 58 + 4, 32, 3);
  if (id <= 63)
    return group(id - 61 + 4, 32, 4);
  throw std::runtime_error("Unsupported buffer data format");
}

uint32_t mask(uint32_t bits) { return bits == 32 ? ~0u : (1u << bits) - 1; }
bool integer(Number n) { return n == Number::Uint || n == Number::Sint; }

// Independent of the host floating-point rounding mode.
double round_even(double value) {
  const double lo = std::floor(value);
  const double fraction = value - lo;
  return lo + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lo, 2.0) != 0));
}

uint32_t read_bits(std::span<const uint8_t> bytes, uint32_t offset, uint32_t width) {
  uint64_t value = 0;
  for (uint32_t i = offset / 8; i < (offset + width + 7) / 8; ++i)
    value |= uint64_t{bytes[i]} << ((i - offset / 8) * 8);
  return static_cast<uint32_t>(value >> (offset % 8)) & mask(width);
}

uint32_t unpack(uint32_t value, uint32_t width, Number n) {
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

uint32_t pack(uint32_t value, uint32_t width, Number n) {
  if (integer(n) || (n == Number::Float && width == 32))
    return value & mask(width);
  const float input = std::bit_cast<float>(value);
  if (n == Number::Float) {
    if (width == 16)
      return util::f32_to_f16(input);
    const uint32_t mantissa_bits = width - 5;
    if (std::isnan(input))
      return (31u << mantissa_bits) | (1u << (mantissa_bits - 1));
    if (input <= 0)
      return 0;
    if (std::isinf(input))
      return 31u << mantissa_bits;
    const double maximum = std::ldexp(2.0 - std::ldexp(1.0, -static_cast<int>(mantissa_bits)), 15);
    if (input >= maximum)
      return (31u << mantissa_bits) - 1;
    int exponent;
    std::frexp(input, &exponent);
    exponent = std::max(exponent - 1, -14);
    const uint32_t significand = static_cast<uint32_t>(
        round_even(std::ldexp(input, static_cast<int>(mantissa_bits) - exponent)));
    return (static_cast<uint32_t>(exponent + 14) << mantissa_bits) + significand;
  }
  double number = std::isnan(input) ? 0 : input;
  if (n == Number::Unorm)
    number = round_even(std::clamp(number, 0.0, 1.0) * mask(width));
  else if (n == Number::Snorm)
    number = round_even(std::clamp(number, -1.0, 1.0) * mask(width - 1));
  else if (n == Number::Uscaled)
    number = std::trunc(std::clamp(number, 0.0, static_cast<double>(mask(width))));
  else
    number = std::trunc(std::clamp(number, -static_cast<double>(1u << (width - 1)),
                                   static_cast<double>(mask(width - 1))));
  return static_cast<uint32_t>(static_cast<int64_t>(number)) & mask(width);
}
} // namespace

uint32_t buffer_format_bytes(uint32_t format, BufferFormatEncoding encoding) {
  const auto f = decode(format, encoding);
  uint32_t bits = 0;
  for (auto width : f.widths)
    bits += width;
  return bits / 8;
}

std::array<uint32_t, 4> unpack_buffer_format(uint32_t format, uint32_t selectors,
                                             std::span<const uint8_t> bytes,
                                             BufferFormatEncoding encoding) {
  const auto f = decode(format, encoding);
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

void pack_buffer_format(uint32_t format, uint32_t selectors, std::span<const uint32_t> components,
                        std::span<uint8_t> bytes, BufferFormatEncoding encoding) {
  const auto f = decode(format, encoding);
  std::fill(bytes.begin(), bytes.end(), 0);
  uint32_t offset = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t width = f.widths[i];
    if (!width)
      continue;
    // Stores invert the descriptor's load mapping: an A8 view maps shader
    // W to physical R, for example. Unprovided shader channels replicate X.
    uint32_t component = 0;
    for (uint32_t shader = 0; shader < 4; ++shader) {
      if (((selectors >> (3 * shader)) & 7) == i + 4) {
        component = components[shader < components.size() ? shader : 0];
        break;
      }
    }
    const uint64_t bits = uint64_t{pack(component, width, f.number)} << (offset % 8);
    for (uint32_t b = offset / 8; b < (offset + width + 7) / 8; ++b)
      bytes[b] |= static_cast<uint8_t>(bits >> ((b - offset / 8) * 8));
    offset += width;
  }
}

bool prepare_buffer_format(Wavefront &wf, VectorMemState &d, uint32_t resource, int format,
                           uint32_t components) {
  d.buffer_components = components;
  d.wf_size = wf.wf_size();
  d.exec_mask = wf.exec();
  d.elem_size = d.num_elems = 1;
  if (!addr_calc::buffer_resource_range_is_backed(wf, resource))
    return false;
  const uint32_t word3 = read_scalar_selector(wf, resource + 3);
  const auto arch = wf.cu().arch();
  d.buffer_format_encoding = arch_is_cdna_4_or_lower(arch)      ? BufferFormatEncoding::Gfx9
                             : arch == ROCJITSU_CODE_ARCH_RDNA1 ? BufferFormatEncoding::Rdna1
                             : arch == ROCJITSU_CODE_ARCH_RDNA2 ? BufferFormatEncoding::Rdna2
                                                                : BufferFormatEncoding::Gfx11;
  d.buffer_format = format < 0
                        ? (word3 >> 12) & (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 0x3f : 0x7f)
                        : static_cast<uint32_t>(format);
  if (format < 0 && d.buffer_format_encoding == BufferFormatEncoding::Gfx9)
    d.buffer_format = ((word3 >> 15) & 15) | (((word3 >> 12) & 7) << 4);
  d.buffer_selectors = format < 0 ? word3 & 0xfff : 4 | (5 << 3) | (6 << 6) | (7 << 9);
  // An explicit typed format cannot bind an INVALID resource descriptor.
  const uint32_t resource_format =
      d.buffer_format_encoding == BufferFormatEncoding::Gfx9
          ? (word3 >> 15) & 15
          : (word3 >> 12) & (arch == ROCJITSU_CODE_ARCH_RDNA4 ? 63 : 127);
  if (resource_format == 0)
    return false;
  const uint32_t bytes = buffer_format_bytes(d.buffer_format, d.buffer_format_encoding);
  if (!bytes)
    return false;
  d.elem_size = bytes;
  return true;
}

void capture_buffer_format_store(Wavefront &wf, VectorMemState &d, uint32_t data_base) {
  RegisterAccess regs(wf);
  const uint32_t registers = d.buffer_d16 ? (d.buffer_components + 1) / 2 : d.buffer_components;
  auto data = regs.read_vgpr_region(data_base, registers, d.lane_mask);
  const auto format = decode(d.buffer_format, d.buffer_format_encoding);
  d.store_data.resize(d.wf_size * d.elem_size);
  for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
    if (!(d.lane_mask & (1ULL << lane)))
      continue;
    std::array<uint32_t, 4> components{};
    for (uint32_t i = 0; i < d.buffer_components; ++i) {
      if (!d.buffer_d16) {
        components[i] = data.lane(i, lane);
        continue;
      }
      const uint32_t shift = d.d16_hi ? 16 : (i % 2) * 16;
      const uint16_t half = data.lane(i / 2, lane) >> shift;
      components[i] =
          integer(format.number)
              ? format.number == Number::Sint
                    ? static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(half)))
                    : half
              : std::bit_cast<uint32_t>(util::f16_to_f32(half));
    }
    pack_buffer_format(
        d.buffer_format, d.buffer_selectors, std::span(components).first(d.buffer_components),
        std::span(d.store_data).subspan(lane * d.elem_size, d.elem_size), d.buffer_format_encoding);
  }
}

void complete_buffer_format_load(Wavefront &wf, ComputeUnitCore &cu, const VectorMemState &d) {
  const uint32_t registers = d.buffer_d16 ? (d.buffer_components + 1) / 2 : d.buffer_components;
  if (!d.lds_dst && !cu.owns_vgpr_range(wf, d.dst_reg_base, registers))
    return;
  const auto format = decode(d.buffer_format, d.buffer_format_encoding);
  for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
    if (!(d.exec_mask & (1ULL << lane)))
      continue;
    const auto texel = [&](uint32_t tap) {
      const bool valid = d.lane_mask & (uint64_t{1} << lane);
      const bool border =
          d.image_sample && !(d.image_sample->taps[tap].lane_mask & (uint64_t{1} << lane));
      const auto bytes = valid && !border
                             ? std::span(d.response_data)
                                   .subspan((tap * d.wf_size + lane) * d.elem_size, d.elem_size)
                             : std::span<const uint8_t>{};
      auto values = unpack_buffer_format(d.buffer_format, d.buffer_selectors, bytes,
                                         d.buffer_format_encoding);
      for (uint32_t i = 0; i < d.buffer_components; ++i) {
        const uint32_t selector = (d.buffer_selectors >> (3 * i)) & 7;
        if (border && valid && selector >= 4) {
          const uint32_t color = d.image_sample->border_color;
          const bool one = color == 2 || (color == 1 && selector == 7);
          values[i] = one ? (integer(format.number) ? 1u : 0x3f800000u) : 0;
        } else if (d.image_srgb && selector >= 4 && selector <= 6) {
          const float value = std::bit_cast<float>(values[i]);
          const float linear =
              value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
          // RDNA3/4 texture decoding rounds sRGB channels to BF16 precision.
          values[i] = std::bit_cast<uint32_t>(util::bf16_to_f32(util::f32_to_bf16_rne(linear)));
        }
      }
      return values;
    };
    auto values = texel(0);
    if (d.image_sample && d.image_sample->tap_count >= 4) {
      std::array<std::array<uint32_t, 4>, ImageSampleAccess::kMaxTaps> texels{};
      for (uint32_t tap = 0; tap < d.image_sample->tap_count; ++tap)
        texels[tap] = texel(tap);
      for (uint32_t c = 0; c < d.buffer_components; ++c) {
        const uint32_t selector = (d.buffer_selectors >> (3 * c)) & 7;
        const bool unorm8 = !d.image_srgb || selector == 7;
        const auto filter_level = [&](uint32_t level) {
          const double x = d.image_sample->fractions[lane][level][0];
          const double y = d.image_sample->fractions[lane][level][1];
          std::array<double, 4> channels{};
          for (uint32_t tap = 0; tap < 4; ++tap) {
            channels[tap] = std::bit_cast<float>(texels[level * 4 + tap][c]);
            if (unorm8)
              channels[tap] = round_even(channels[tap] * 255);
          }
          const std::array weights{(1 - x) * (1 - y), x * (1 - y), (1 - x) * y, x * y};
          if (!unorm8) {
            double maximum = 0;
            for (uint32_t tap = 0; tap < 4; ++tap)
              if (weights[tap] != 0)
                maximum = std::max(maximum, channels[tap]);
            // The sRGB filter aligns contributing texels to a shared exponent
            // with twelve significant bits before multiplying by the weights.
            const int exponent = maximum > 0 ? std::ilogb(maximum) : 0;
            const double scale = std::ldexp(1.0, 11 - exponent);
            for (auto &channel : channels)
              channel = std::floor(channel * scale) / scale;
          }
          return channels[0] * weights[0] + channels[1] * weights[1] + channels[2] * weights[2] +
                 channels[3] * weights[3];
        };
        double filtered = filter_level(0);
        if (d.image_sample->tap_count == 8) {
          const double fraction = d.image_sample->mip_fractions[lane];
          if (unorm8) {
            // Each weighted mip retains nineteen fractional texel-value bits
            // before the two contributions are added.
            filtered = (round_even(std::ldexp(filtered * (1 - fraction), 19)) +
                        round_even(std::ldexp(filter_level(1) * fraction, 19))) /
                       std::ldexp(1.0, 19);
          } else {
            filtered = filtered * (1 - fraction) + filter_level(1) * fraction;
          }
        }
        if (unorm8) {
          // Preserve thirteen fractional texel bits, then normalize by byte replication.
          // Convert its 34 fractional bits to FP32 with midpoints rounded up.
          const uint64_t numerator = static_cast<uint64_t>(round_even(std::ldexp(filtered, 13)))
                                     << 13;
          uint64_t normalized = numerator + (numerator >> 8) + (numerator >> 16) +
                                (numerator >> 24) + (numerator >> 32);
          const uint32_t bits = std::bit_width(normalized);
          const uint32_t shift = bits > 24 ? bits - 24 : 0;
          if (shift)
            normalized = (normalized + (uint64_t{1} << (shift - 1))) >> shift;
          filtered = std::ldexp(static_cast<double>(normalized), static_cast<int>(shift) - 34);
        } else if (filtered > 0) {
          // The sRGB filter rounds its normalized result to 29 significant
          // bits before conversion to FP32. This intermediate rounding can
          // turn a value on either side of an FP32 midpoint into an exact tie.
          const double scale = std::ldexp(1.0, 28 - std::ilogb(filtered));
          filtered = round_even(filtered * scale) / scale;
        }
        values[c] = std::bit_cast<uint32_t>(static_cast<float>(filtered));
      }
    }
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
}
} // namespace rocjitsu::amdgpu
