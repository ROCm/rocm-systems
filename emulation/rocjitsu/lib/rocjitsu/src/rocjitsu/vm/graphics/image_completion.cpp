// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/graphics/image_completion.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/vm/amdgpu/buffer_format_detail.h"
#include "rocjitsu/vm/graphics/image_bc.h"
#include "rocjitsu/vm/graphics/image_filter.h"

#include <cstring>
#include <limits>

namespace rocjitsu::amdgpu {
using namespace buffer_format_detail;
namespace {

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

} // namespace

void complete_image_load(Wavefront &wf, ComputeUnitCore &cu, const VectorMemState &d) {
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
    write_format_load_lane(wf, cu, d, lane, values);
    destinations_materialized = true;
  }
}
} // namespace rocjitsu::amdgpu
