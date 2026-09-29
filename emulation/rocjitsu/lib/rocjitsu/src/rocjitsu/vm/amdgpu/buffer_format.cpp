// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/buffer_format.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/hwfloat/unorm.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/image_bc.h"
#include "rocjitsu/vm/amdgpu/image_filter.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace rocjitsu::amdgpu {
namespace {
using Number = BufferNumberFormat;
using Format = BufferFormat;

util::FailureOr<Format> decode(uint32_t id, BufferFormatEncoding encoding) {
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
      return Format{{}, Number::Uint};
    if (dfmt >= widths.size() || nfmt > 7 || nfmt == 6)
      return util::Result::failure();
    return Format{widths[dfmt], nfmt == 7 ? Number::Float : static_cast<Number>(nfmt)};
  }
  if (encoding == BufferFormatEncoding::Rdna1 || encoding == BufferFormatEncoding::Rdna2) {
    if (id > 77 || (encoding == BufferFormatEncoding::Rdna2 &&
                    ((id >= 30 && id <= 35) || (id >= 37 && id <= 42) || id == 46 || id == 47)))
      return util::Result::failure();
    if (id >= 30 && id <= 36)
      return Format{{11, 11, 10, 0}, static_cast<Number>(id - 30)};
    if (id >= 37 && id <= 43)
      return Format{{10, 11, 11, 0}, static_cast<Number>(id - 37)};
    if (id >= 44 && id <= 49)
      return Format{{2, 10, 10, 10}, static_cast<Number>(id - 44)};
    // The remaining GFX10 formats have the same order as GFX11, shifted by 14.
    if (id >= 50)
      id -= 14;
  }
  if (id == 0)
    return Format{{}, Number::Uint};
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
    return Format{{11, 11, 10, 0}, Number::Float};
  if (id == 31)
    return Format{{10, 11, 11, 0}, Number::Float};
  if (id <= 35)
    return Format{{2, 10, 10, 10}, static_cast<Number>(id - 32 + (id >= 34 ? 2 : 0))};
  if (id <= 41)
    return Format{{10, 10, 10, 2}, static_cast<Number>(id - 36)};
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
  return util::Result::failure();
}

uint32_t mask(uint32_t bits) { return bits == 32 ? ~0u : (1u << bits) - 1; }
bool integer(Number n) { return n == Number::Uint || n == Number::Sint; }

// Independent of the host floating-point rounding mode.
double round_even(double value) {
  const double lo = std::floor(value);
  const double fraction = value - lo;
  return lo + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lo, 2.0) != 0));
}

uint32_t encode_filtered_unorm(uint64_t rounded_unorm, uint32_t unorm_width) {
  // Normalize by repeating seven-, eight- or ten-bit fields, then round the 34
  // fractional bits to FP32 with midpoints rounded up. This is the existing
  // texture normalization, not ordinary division by the UNORM maximum.
  const uint64_t numerator = rounded_unorm << (34 - 13 - unorm_width);
  uint64_t normalized = 0;
  for (uint32_t offset = 0; offset < 34; offset += unorm_width)
    normalized += numerator >> offset;
  const uint32_t bits = std::bit_width(normalized);
  const uint32_t shift = bits > 24 ? bits - 24 : 0;
  if (shift)
    normalized = (normalized + (uint64_t{1} << (shift - 1))) >> shift;
  if (!normalized)
    return 0;
  const uint32_t leading = std::bit_width(normalized) - 1;
  const uint32_t significand = static_cast<uint32_t>((normalized << (63 - leading)) >> 40);
  const uint32_t exponent = leading + shift + (127 - 34);
  return (exponent << 23) | (significand & 0x7fffffu);
}

// Inspect bits without converting a nonintegral value or touching host FP
// flags. Preparation emits fractions in [0,1] at exact multiples of 1/256;
// externally supplied or observer-modified fractions retain the general path.
bool sample_fraction_q8(float value, uint32_t &result) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  if (!(bits & 0x7fffffffu)) {
    result = 0;
    return true;
  }
  const uint32_t exponent = (bits >> 23) & 255;
  if ((bits >> 31) || exponent < 119 || exponent > 127)
    return false;
  const uint32_t shift = 142 - exponent;
  const uint32_t significand = (bits & 0x7fffffu) | 0x800000u;
  if (significand & ((1u << shift) - 1))
    return false;
  result = significand >> shift;
  return result <= 256;
}

uint32_t round_even_shift(uint32_t value, uint32_t shift) {
  const uint32_t integer = value >> shift;
  const uint32_t remainder = value & ((uint32_t{1} << shift) - 1);
  const uint32_t half = uint32_t{1} << (shift - 1);
  return integer + (remainder > half || (remainder == half && (integer & 1)));
}

bool filter_unorm8_single(const ImageSampleAccess &access, uint32_t lane, uint32_t components,
                          std::span<const std::array<uint32_t, 4>> texels,
                          std::array<uint32_t, 4> &values) {
  if (access.filter_counts[lane] != 1 || access.texels_per_tap != 1 ||
      (access.taps_per_filter != 4 && access.taps_per_filter != 8))
    return false;
  const uint32_t levels = access.taps_per_filter / 4;
  std::array<std::array<uint32_t, 4>, 2> weights;
  uint32_t mip = 0;
  if (levels == 2 && !sample_fraction_q8(access.mip_fractions[lane], mip))
    return false;
  for (uint32_t level = 0; level < levels; ++level) {
    uint32_t x, y;
    if (access.filters[0].cube_corners[lane][level] ||
        !sample_fraction_q8(access.filters[0].fractions[lane][level][0], x) ||
        !sample_fraction_q8(access.filters[0].fractions[lane][level][1], y))
      return false;
    weights[level] = {(256 - x) * (256 - y), x * (256 - y), (256 - x) * y, x * y};
  }
  for (uint32_t c = 0; c < components; ++c) {
    std::array<uint32_t, 2> filtered{};
    for (uint32_t level = 0; level < levels; ++level)
      for (uint32_t tap = 0; tap < 4; ++tap)
        filtered[level] += texels[4 * level + tap][c] * weights[level][tap];
    // UNORM8 texels are at most 255; Q16 weights sum to 65536. Each
    // level is at most 255 * 2^16, and either mip product is below 2^32.
    // Round each mip contribution separately to Q19, exactly as the double
    // path does. A single level retains Q16 until the final Q13 rounding.
    const uint32_t rounded = levels == 2
                                 ? round_even_shift(round_even_shift(filtered[0] * (256 - mip), 5) +
                                                        round_even_shift(filtered[1] * mip, 5),
                                                    6)
                                 : round_even_shift(filtered[0], 3);
    values[c] = encode_filtered_unorm(rounded, 8);
  }
  return true;
}

constexpr float kFilterNan = std::bit_cast<float>(0xffc00000u);

double filter_float_texels(std::array<double, 4> channels, const std::array<double, 4> &weights,
                           uint32_t unaligned = 0, uint32_t precision = 12) {
  double maximum = 0;
  uint32_t active = 0, selected = 0;
  bool positive_inf = false, negative_inf = false;
  for (uint32_t tap = 0; tap < channels.size(); ++tap) {
    // Zero-weight texels do not contribute even when their value is NaN or infinity.
    if (weights[tap] == 0) {
      channels[tap] = 0;
      continue;
    }
    ++active;
    selected = tap;
    const double value = channels[tap];
    if (std::isnan(value))
      return kFilterNan;
    positive_inf |= value == std::numeric_limits<double>::infinity();
    negative_inf |= value == -std::numeric_limits<double>::infinity();
    maximum = std::max(maximum, std::abs(value));
  }
  if (positive_inf && negative_inf)
    return kFilterNan;
  if (positive_inf || negative_inf)
    return negative_inf ? -std::numeric_limits<double>::infinity()
                        : std::numeric_limits<double>::infinity();
  if (active == 1)
    return channels[selected]; // Preserve signed zero at an exact texel center.

  // RDNA3/4 filters align contributing texels to a shared exponent,
  // truncating toward zero: twelve bits for sRGB/FP16, twenty-five for FP32.
  const int exponent = maximum > 0 ? std::ilogb(maximum) : 0;
  const double scale = std::ldexp(1.0, int(precision) - 1 - exponent);
  for (uint32_t i = 0; i < channels.size(); ++i)
    if (!(unaligned & (1u << i)))
      channels[i] = std::trunc(channels[i] * scale) / scale;
  // Weighted zero sums are positive, including sums of only negative zeros.
  return 0.0 + channels[0] * weights[0] + channels[1] * weights[1] + channels[2] * weights[2] +
         channels[3] * weights[3];
}

/// The floating-point footprint accumulator aligns signed operands to 35 bits
/// before each addition. A carry increases its exponent; cancellation does not
/// decrease it. Final rounding must retain that exponent as well.
class ImageFilterAccumulator {
public:
  explicit ImageFilterAccumulator(double first) : value(first) {
    if (std::isfinite(first) && first != 0) {
      std::frexp(first, &exponent);
      has_exponent = true;
    }
  }

  void add(double term) {
    if (!std::isfinite(value) || !std::isfinite(term)) {
      value += term;
      if (std::isnan(value))
        value = kFilterNan;
      return;
    }
    if (term != 0) {
      int term_exponent;
      std::frexp(term, &term_exponent);
      exponent = has_exponent ? std::max(exponent, term_exponent) : term_exponent;
      has_exponent = true;
    }
    if (!has_exponent) {
      value = 0; // Weighted sums of zeros are positive, including only negative zeros.
      return;
    }
    double mantissa =
        std::floor(std::ldexp(value, 35 - exponent)) + std::floor(std::ldexp(term, 35 - exponent));
    if (std::abs(mantissa) >= 0x1p35) {
      mantissa = std::floor(mantissa / 2);
      ++exponent;
    }
    value = std::ldexp(mantissa, exponent - 35);
  }

  double value;
  int exponent = 0;

private:
  bool has_exponent = false;
};

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
  if (n == Number::Unorm && hwfloat::supports_unorm_width(width))
    return hwfloat::unorm_from_f32(value, width);
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

util::FailureOr<BufferFormat> decode_buffer_format(uint32_t format, BufferFormatEncoding encoding) {
  return decode(format, encoding);
}

util::FailureOr<uint32_t> buffer_format_bytes(uint32_t format, BufferFormatEncoding encoding) {
  const auto decoded = decode(format, encoding);
  if (decoded.failed())
    return util::Result::failure();
  return decoded.value().byte_size();
}

namespace {
std::array<uint32_t, 4> unpack_format(const Format &f, uint32_t selectors,
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

void pack_format(const Format &f, uint32_t selectors, std::span<const uint32_t> components,
                 std::span<uint8_t> bytes) {
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

} // namespace

util::FailureOr<std::array<uint32_t, 4>> unpack_buffer_format(uint32_t format, uint32_t selectors,
                                                              std::span<const uint8_t> bytes,
                                                              BufferFormatEncoding encoding) {
  const auto decoded = decode(format, encoding);
  if (decoded.failed())
    return util::Result::failure();
  const fp_mode::detail::ScopedFenv environment(0);
  return unpack_format(decoded.value(), selectors, bytes);
}

util::Result pack_buffer_format(uint32_t format, uint32_t selectors,
                                std::span<const uint32_t> components, std::span<uint8_t> bytes,
                                BufferFormatEncoding encoding) {
  const auto decoded = decode(format, encoding);
  if (decoded.failed())
    return util::Result::failure();
  // Every channel of this path uses integer bits, including NaN classification.
  // Other formats and the shader-store conversion scope retain their FP policy.
  if (decoded.value().number == Number::Unorm &&
      std::ranges::all_of(decoded.value().widths, [](uint32_t width) {
        return width == 0 || hwfloat::supports_unorm_width(width);
      })) {
    pack_format(decoded.value(), selectors, components, bytes);
    return util::Result::success();
  }
  const fp_mode::detail::ScopedFenv environment(0);
  pack_format(decoded.value(), selectors, components, bytes);
  return util::Result::success();
}

util::FailureOr<bool> prepare_buffer_format(Wavefront &wf, VectorMemState &d, uint32_t resource,
                                            int format, uint32_t components) {
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
  const auto decoded = decode(d.buffer_format, d.buffer_format_encoding);
  if (decoded.failed())
    return util::Result::failure();
  d.decoded_buffer_format = decoded.value();
  d.elem_size = d.decoded_buffer_format.byte_size();
  return d.elem_size != 0;
}

void capture_buffer_format_store(Wavefront &wf, VectorMemState &d, uint32_t data_base) {
  RegisterAccess regs(wf);
  const uint32_t registers = d.buffer_d16 ? (d.buffer_components + 1) / 2 : d.buffer_components;
  auto data = regs.read_vgpr_region(data_base, registers, d.lane_mask);
  const auto &format = d.decoded_buffer_format;
  const fp_mode::detail::ScopedFenv environment(0);
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
    pack_format(format, d.buffer_selectors, std::span(components).first(d.buffer_components),
                std::span(d.store_data).subspan(lane * d.elem_size, d.elem_size));
  }
}

void complete_buffer_format_load(Wavefront &wf, ComputeUnitCore &cu, const VectorMemState &d) {
  const uint32_t registers = d.buffer_d16 ? (d.buffer_components + 1) / 2 : d.buffer_components;
  if (!d.lds_dst && !cu.owns_vgpr_range(wf, d.dst_reg_base, registers))
    return;
  const auto &format = d.decoded_buffer_format;
  const fp_mode::detail::ScopedFenv environment(0);
  // Fixed UNORM8 filtering consumes integer texel units. Decode those units
  // directly instead of normalizing to FP32 and rounding back for each tap.
  const bool unorm8_texels =
      d.image_sample && d.image_sample->tap_count >= 4 && !d.image_srgb &&
      !d.image_sample->comparison && !d.image_sample->gather && format.number == Number::Unorm &&
      format.widths[0] == 8 &&
      std::ranges::all_of(format.widths, [](uint32_t width) { return width == 0 || width == 8; });
  // The first active lane touches every destination register with the original
  // arithmetic state, including any lazy-storage allocation. Later VGPR lanes
  // reuse those chunks. LDS can still grow at later lanes, so retain its filter.
  const bool integer_filter =
      unorm8_texels && !d.lds_dst && !cu.observes_register_access() && !cu.debug_active();
  bool destinations_materialized = false;
  std::array<uint32_t, 4> channel_offsets{};
  for (uint32_t c = 1; c < channel_offsets.size(); ++c)
    channel_offsets[c] = channel_offsets[c - 1] + format.widths[c - 1] / 8;
  for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
    if (!(d.exec_mask & (1ULL << lane)))
      continue;
    const bool integer_texels = integer_filter && destinations_materialized;
    const auto texel = [&](uint32_t tap) {
      const bool valid = d.lane_mask & (uint64_t{1} << lane);
      const bool border =
          d.image_sample && !(d.image_sample->taps[tap].lane_mask & (uint64_t{1} << lane));
      auto bytes = valid && !border
                       ? std::span(d.response_data)
                             .subspan((tap * d.wf_size + lane) * d.elem_size, d.elem_size)
                       : std::span<const uint8_t>{};
      std::array<uint8_t, 8> decoded_texel{};
      if (d.image_bc_format && !bytes.empty()) {
        const uint32_t coordinate =
            d.image_sample ? d.image_sample->taps[tap].coordinates[lane] : 0;
        const uint32_t index =
            d.image_sample ? (coordinate & 3) + ((coordinate >> 14) & 12) : d.image_bc_texels[lane];
        if (d.image_bc_format <= 114) {
          const auto rgba = decode_image_bc(d.image_bc_format, index, bytes);
          std::copy(rgba.begin(), rgba.end(), decoded_texel.begin());
        } else {
          const bool signed_format = !(d.image_bc_format & 1);
          const uint32_t channels = d.image_bc_format <= 116 ? 1 : 2;
          for (uint32_t c = 0; c < channels; ++c) {
            const int32_t value =
                decode_image_bc_scalar(bytes.subspan(c * 8, 8), index, signed_format);
            const uint32_t bits =
                encode_filtered_unorm(uint32_t(std::abs(value)) << 7, signed_format ? 7 : 8) |
                (value < 0 ? 0x80000000u : 0);
            std::memcpy(decoded_texel.data() + c * 4, &bits, 4);
          }
        }
        bytes = std::span(decoded_texel).first(format.byte_size());
      }
      std::array<uint32_t, 4> values{};
      if (d.image_sample && d.image_sample->gather && !valid)
        return values;
      if (integer_texels) {
        for (uint32_t c = 0; c < d.buffer_components; ++c) {
          const uint32_t selector = (d.buffer_selectors >> (3 * c)) & 7;
          if (selector == 1) {
            values[c] = 255;
          } else if (selector >= 4) {
            if (border && valid) {
              const uint32_t color = d.image_sample->border_color;
              values[c] = color == 2 || (color == 1 && selector == 7) ? 255 : 0;
            } else if (!bytes.empty() && format.widths[selector - 4]) {
              values[c] = bytes[channel_offsets[selector - 4]];
            }
          }
        }
        return values;
      }
      values = unpack_format(format, d.buffer_selectors, bytes);
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
        if (d.image_sampling && format.number == Number::Float) {
          // Before filtering, sampling canonicalizes decoded FP32 NaNs and
          // flushes subnormals to signed zero, independently of VALU MODE.
          // Image and buffer loads preserve the decoded texel bits.
          const uint32_t magnitude = values[i] & 0x7fffffffu;
          if (magnitude > 0x7f800000u)
            values[i] = std::bit_cast<uint32_t>(kFilterNan);
          else if (magnitude < 0x00800000u)
            values[i] &= 0x80000000u;
        }
      }
      if (d.image_sample && d.image_sample->comparison) {
        const auto &comparison = *d.image_sample->comparison;
        const float reference = comparison.references[lane];
        for (uint32_t i = 0; i < d.buffer_components; ++i) {
          const float value = std::bit_cast<float>(values[i]);
          const bool comparisons[] = {false,
                                      (reference < value),
                                      reference == value,
                                      reference <= value,
                                      (reference > value),
                                      reference != value,
                                      reference >= value,
                                      true};
          values[i] = comparisons[comparison.function] ? 0x3f800000u : 0;
        }
      }
      return values;
    };
    auto values = texel(0);
    if (d.image_sample && d.image_sample->gather) {
      // Ordinary gather starts at lower left and proceeds counterclockwise.
      // Horizontal gather returns left to right.
      constexpr uint32_t order[] = {2, 3, 1, 0};
      for (uint32_t i = 0; i < 4; ++i) {
        const auto &access = *d.image_sample;
        const uint32_t tap = access.horizontal ? i : order[i];
        const uint32_t first = tap * access.texels_per_tap;
        values[i] = texel(first)[0];
        if (access.filters[0].cube_corners[lane][0] & (1u << tap)) {
          // Integer cube gathers return zero at the missing corner.
          if (integer(format.number)) {
            values[i] = 0;
            continue;
          }
          const double a = std::bit_cast<float>(values[i]);
          const double b = std::bit_cast<float>(texel(first + 1)[0]);
          const double c = std::bit_cast<float>(texel(first + 2)[0]);
          values[i] = std::bit_cast<uint32_t>(
              static_cast<float>((a * 21846 + b * 21845 + c * 21845) / 65536));
        }
      }
    } else if (d.image_sample && d.image_sample->tap_count >= 4) {
      const uint32_t filter_count = d.image_sample->filter_counts[lane];
      const uint32_t tap_count = filter_count * d.image_sample->taps_per_filter;
      // Initialize only the texels consumed by this lane's filters, reusing
      // the first texel already decoded above.
      std::array<std::array<uint32_t, 4>, ImageSampleAccess::kMaxTaps> texels;
      texels[0] = values;
      for (uint32_t tap = 1; tap < tap_count; ++tap)
        texels[tap] = texel(tap);
      if (!(integer_texels && filter_unorm8_single(*d.image_sample, lane, d.buffer_components,
                                                   std::span(texels).first(tap_count), values))) {
        std::array<std::array<std::array<double, 4>, 2>, ImageSampleAccess::kMaxFilters> weights;
        const uint32_t levels =
            d.image_sample->taps_per_filter == 8 * d.image_sample->texels_per_tap ? 2 : 1;
        for (uint32_t filter = 0; filter < filter_count; ++filter)
          for (uint32_t level = 0; level < levels; ++level) {
            const double x = d.image_sample->filters[filter].fractions[lane][level][0];
            const double y = d.image_sample->filters[filter].fractions[lane][level][1];
            weights[filter][level] = {(1 - x) * (1 - y), x * (1 - y), (1 - x) * y, x * y};
          }
        for (uint32_t c = 0; c < d.buffer_components; ++c) {
          const uint32_t selector = (d.buffer_selectors >> (3 * c)) & 7;
          const bool bc_scalar =
              !d.image_sample->comparison && d.image_bc_format >= 115 && d.image_bc_format <= 118;
          const bool bc_signed = bc_scalar && !(d.image_bc_format & 1);
          const bool fixed_unorm = !d.image_sample->comparison && format.number == Number::Unorm &&
                                   (format.widths[0] == 8 || format.widths[0] == 10) &&
                                   (!d.image_srgb || selector == 7);
          const uint32_t unorm_width = bc_signed ? 7 : format.widths[0] == 10 ? 10 : 8;
          const uint32_t unorm_max = (1u << unorm_width) - 1;
          const auto filter_level = [&](uint32_t filter_index, uint32_t level) {
            std::array<double, 4> channels{};
            const auto &access = *d.image_sample;
            const uint32_t corners = access.filters[filter_index].cube_corners[lane][level];
            for (uint32_t tap = 0; tap < 4; ++tap) {
              const uint32_t first =
                  filter_index * access.taps_per_filter + (level * 4 + tap) * access.texels_per_tap;
              const auto channel = [&](uint32_t source) {
                if (integer_texels)
                  return double(texels[first + source][c]);
                const double value = std::bit_cast<float>(texels[first + source][c]);
                // BC4/5 retain six fractional bits in decoded texel units.
                if (bc_scalar)
                  return round_even(value * unorm_max * 64) / 64;
                return fixed_unorm ? round_even(value * unorm_max) : value;
              };
              channels[tap] = channel(0);
              if (corners & (1u << tap))
                channels[tap] =
                    (channels[tap] * 21846 + channel(1) * 21845 + channel(2) * 21845) / 65536;
            }
            const auto &level_weights = weights[filter_index][level];
            if (!fixed_unorm && !bc_scalar)
              return filter_float_texels(
                  channels, level_weights, corners,
                  format.number == Number::Float && format.widths[0] == 32 ? 25 : 12);
            return channels[0] * level_weights[0] + channels[1] * level_weights[1] +
                   channels[2] * level_weights[2] + channels[3] * level_weights[3];
          };
          if (bc_scalar) {
            // BC4/5 accumulate weighted mips with 27 fractional texel bits.
            // Keep mip contributions separate: RDNA3 consumes the smaller mip
            // first, RDNA4 the larger one. Cancellation and exponent growth
            // make that order observable even after texture normalization.
            ImageFilterAccumulator accumulator(0);
            const uint32_t normalization = std::bit_floor(filter_count);
            for (uint32_t pass = 0; pass < levels; ++pass) {
              const uint32_t level =
                  cu.arch() == ROCJITSU_CODE_ARCH_RDNA4 ? pass : levels - 1 - pass;
              const double mip_weight = levels == 1 ? 1.0
                                        : level     ? d.image_sample->mip_fractions[lane]
                                                    : 1.0 - d.image_sample->mip_fractions[lane];
              for (uint32_t filter = 0; filter < filter_count; ++filter) {
                const double value = filter_level(filter, level) * mip_weight *
                                     image_anisotropic_filter_weight(filter_count, filter) *
                                     normalization;
                accumulator.add(round_even(value * 0x1p27) * 0x1p-27);
              }
            }
            // Round to 29 significant bits, retaining at least eight integer
            // bits, then discard signed low bits before magnitude conversion.
            const double scale = std::ldexp(1.0, 29 - std::max(8, accumulator.exponent));
            const double filtered = round_even(accumulator.value * scale) / scale / normalization;
            const double units = std::floor(filtered * 0x1p13);
            values[c] = encode_filtered_unorm(static_cast<uint64_t>(std::abs(units)), unorm_width) |
                        (units < 0 ? 0x80000000u : 0);
            continue;
          }
          const auto filter_sample = [&](uint32_t filter_index) {
            double filtered = filter_level(filter_index, 0);
            if (d.image_sample->taps_per_filter == 8 * d.image_sample->texels_per_tap) {
              const double fraction = d.image_sample->mip_fractions[lane];
              if (fixed_unorm) {
                // Each weighted mip retains nineteen fractional texel-value bits
                // before the two contributions are added. Fixed texel values
                // and Q8 fractions keep these binary scales exact and normal.
                filtered = (round_even(filtered * (1 - fraction) * 0x1p19) +
                            round_even(filter_level(filter_index, 1) * fraction * 0x1p19)) *
                           0x1p-19;
              } else {
                // Do not multiply an unused mip's NaN/infinity by zero, or lose
                // signed zero at an exact mip level.
                if (fraction == 1)
                  filtered = filter_level(filter_index, 1);
                else if (fraction != 0)
                  filtered =
                      0.0 + filtered * (1 - fraction) + filter_level(filter_index, 1) * fraction;
                if (std::isnan(filtered))
                  filtered = kFilterNan;
              }
            }
            return filtered;
          };
          double filtered;
          int accumulation_exponent = 0;
          if (filter_count > 1) {
            const auto weighted_filter = [&](uint32_t index) {
              const double value =
                  filter_sample(index) * image_anisotropic_filter_weight(filter_count, index);
              // The UNORM accumulator normalizes after summation. Each weighted
              // contribution retains nineteen fractional texel-value bits.
              return fixed_unorm
                         ? round_even(value * std::bit_floor(filter_count) * 0x1p19) * 0x1p-19
                         : value;
            };
            filtered = weighted_filter(0);
            if (fixed_unorm) {
              for (uint32_t filter_index = 1; filter_index < filter_count; ++filter_index)
                filtered += weighted_filter(filter_index);
            } else {
              ImageFilterAccumulator accumulator(filtered);
              for (uint32_t filter_index = 1; filter_index < filter_count; ++filter_index)
                accumulator.add(weighted_filter(filter_index));
              filtered = accumulator.value;
              accumulation_exponent = accumulator.exponent;
            }
          } else {
            filtered = filter_sample(0);
          }
          if (fixed_unorm) {
            // Anisotropic accumulation rounds half up before normalization, then
            // discards the normalization remainder. A single filter rounds even.
            const uint64_t rounded_unorm =
                filter_count > 1 ? static_cast<uint64_t>(std::floor(filtered * 0x1p13 + 0.5)) /
                                       std::bit_floor(filter_count)
                                 : static_cast<uint64_t>(round_even(filtered * 0x1p13));
            values[c] = encode_filtered_unorm(rounded_unorm, unorm_width);
            continue;
          } else if (std::isfinite(filtered) && filtered != 0) {
            const int exponent =
                filter_count > 1 ? accumulation_exponent : 1 + std::ilogb(std::abs(filtered));
            if (format.number == Number::Float && format.widths[0] == 32) {
              // FP32 filtering retains 35 signed significant bits before the
              // final rounding. Discarding signed low bits rounds downward.
              const double scale = std::ldexp(1.0, 35 - exponent);
              filtered = std::floor(filtered * scale) / scale;
            }
            // The floating-point filter rounds its result to 29 significant
            // bits before conversion to FP32. This intermediate rounding can
            // turn a value on either side of an FP32 midpoint into an exact tie.
            const double scale = std::ldexp(1.0, 29 - exponent);
            filtered = round_even(filtered * scale) / scale;
          }
          values[c] = std::bit_cast<uint32_t>(static_cast<float>(filtered));
          // Sampling flushes FP32 underflow at the output as well as the input.
          if (format.number == Number::Float && format.widths[0] == 32 &&
              (values[c] & 0x7fffffffu) < 0x00800000u)
            values[c] &= 0x80000000u;
        }
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
    destinations_materialized = true;
  }
}
} // namespace rocjitsu::amdgpu
