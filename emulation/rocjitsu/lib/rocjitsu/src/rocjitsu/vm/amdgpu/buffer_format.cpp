// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/buffer_format.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/hwfloat/unorm.h"
#include "rocjitsu/vm/amdgpu/buffer_format_detail.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/graphics/image_bc.h"
#include "rocjitsu/vm/graphics/image_completion.h"
#include "util/data_types.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <optional>
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
    Format format{widths[dfmt], nfmt == 7 ? Number::Float : static_cast<Number>(nfmt)};
    if (format.number == Number::Float && std::ranges::any_of(format.widths, [](uint32_t width) {
          return width != 0 && width != 10 && width != 11 && width != 16 && width != 32;
        }))
      return util::Result::failure();
    return format;
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

uint32_t pack(uint32_t value, uint32_t width, Number n, BufferFormatEncoding encoding) {
  // Native gfx1100/gfx1201 integer stores saturate before narrowing, including D16 sources.
  if (encoding == BufferFormatEncoding::Gfx11 && buffer_format_is_integer(n) && width < 32) {
    if (n == Number::Uint)
      return std::min(value, buffer_format_mask(width));
    const int32_t maximum = static_cast<int32_t>(buffer_format_mask(width - 1));
    return static_cast<uint32_t>(std::clamp(std::bit_cast<int32_t>(value), -maximum - 1, maximum)) &
           buffer_format_mask(width);
  }
  if (buffer_format_is_integer(n) || (n == Number::Float && width == 32))
    return value & buffer_format_mask(width);
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
        buffer_format_round_even(std::ldexp(input, static_cast<int>(mantissa_bits) - exponent)));
    return (static_cast<uint32_t>(exponent + 14) << mantissa_bits) + significand;
  }
  double number = std::isnan(input) ? 0 : input;
  if (n == Number::Unorm)
    number = buffer_format_round_even(std::clamp(number, 0.0, 1.0) * buffer_format_mask(width));
  else if (n == Number::Snorm) {
    number = std::clamp(number, -1.0, 1.0) * buffer_format_mask(width - 1);
    // Native gfx1100/gfx1201 encode +/-0.5 as +/-1 in two-bit SNORM.
    number = encoding == BufferFormatEncoding::Gfx11 && width == 2
                 ? std::round(number)
                 : buffer_format_round_even(number);
  } else if (n == Number::Uscaled)
    number = std::trunc(std::clamp(number, 0.0, static_cast<double>(buffer_format_mask(width))));
  else
    number = std::trunc(std::clamp(number, -static_cast<double>(1u << (width - 1)),
                                   static_cast<double>(buffer_format_mask(width - 1))));
  return static_cast<uint32_t>(static_cast<int64_t>(number)) & buffer_format_mask(width);
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
// Native gfx1100/gfx1201 sweeps qualify exact selector patterns by memory
// channel count. General DST_SEL routing is not an inverse of load swizzling.
// CDNA1-4, RDNA1/2 and RDNA3.5 use this policy by extrapolation, without native
// measurements. Rejection marks unsupported emulation, not a hardware fault.
util::FailureOr<uint32_t> resolve_store_selectors(const Format &f, uint32_t selectors,
                                                  bool image_store = false) {
  constexpr uint32_t identity = 4 | (5 << 3) | (6 << 6) | (7 << 9);
  constexpr uint32_t reversed = 7 | (6 << 3) | (5 << 6) | (4 << 9);
  const uint32_t channels =
      std::count_if(f.widths.begin(), f.widths.end(), [](uint32_t width) { return width != 0; });
  if (channels == 4) {
    constexpr uint32_t rgb = 4 | (5 << 3) | (6 << 6);
    constexpr uint32_t bgra = 6 | (5 << 3) | (4 << 6) | (7 << 9);
    constexpr uint32_t zyx1 = 6 | (5 << 3) | (4 << 6) | (1 << 9);
    if (selectors == identity || selectors == reversed || selectors == bgra || selectors == rgb ||
        selectors == (rgb | (1 << 9)) || (image_store && selectors == zyx1))
      return selectors;
    return util::Result::failure();
  }
  if (!image_store && channels == 1 && f.widths[0] == 32 && f.number == Number::Float &&
      selectors == 0)
    return 4u << 9; // Native gfx1100/gfx1201 resolve this view to A8 (W to R).
  if (channels == 1 && (selectors == identity || selectors == reversed || selectors == (4 << 9)))
    return selectors; // Single-channel identity, WZYX, and A8 views were measured.
  if (!channels)
    return util::Result::failure();
  for (uint32_t channel = 0; channel < 4; ++channel) {
    const uint32_t selector = (selectors >> (channel * 3)) & 7;
    if (channel < channels ? selector != channel + 4 : selector > 1)
      return util::Result::failure();
  }
  return selectors;
}

void pack_format(const Format &f, uint32_t selectors, std::span<const uint32_t> components,
                 std::span<uint8_t> bytes, BufferStoreMissingChannel missing,
                 BufferFormatEncoding encoding, uint32_t component_mask, bool raw = false) {
  // Native Gfx11-format stores encode SCALED as NORM, unlike SCALED loads.
  const auto number = encoding != BufferFormatEncoding::Gfx11 ? f.number
                      : f.number == Number::Uscaled           ? Number::Unorm
                      : f.number == Number::Sscaled           ? Number::Snorm
                                                              : f.number;
  std::fill(bytes.begin(), bytes.end(), 0);
  uint32_t offset = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t width = f.widths[i];
    if (!width)
      continue;
    // Qualified permutations invert the load mapping (A8 maps W to R).
    // A selector of one writes a constant to physical alpha for both buffer
    // and image XYZ1/ZYX1 stores, independently of DMASK.
    uint32_t component = 0;
    if (i == 3 && ((selectors >> 9) & 7) == 1) {
      component = buffer_format_is_integer(f.number) ? 1u : 0x3f800000u;
    } else {
      for (uint32_t shader = 0; shader < 4; ++shader) {
        if (((selectors >> (3 * shader)) & 7) == i + 4) {
          if (component_mask & (1u << shader))
            component = components[shader];
          else if (missing == BufferStoreMissingChannel::ReplicateFirst && (component_mask & 1u))
            component = components[0];
          break;
        }
      }
    }
    const uint64_t bits = uint64_t{raw ? (component & buffer_format_mask(width))
                                       : pack(component, width, number, encoding)}
                          << (offset % 8);
    for (uint32_t b = offset / 8; b < (offset + width + 7) / 8; ++b)
      bytes[b] |= static_cast<uint8_t>(bits >> ((b - offset / 8) * 8));
    offset += width;
  }
}

// Integer UNORM conversion is independent of the caller's FP environment.
void pack_components(const Format &format, uint32_t selectors, std::span<const uint32_t> components,
                     std::span<uint8_t> bytes, BufferStoreMissingChannel missing,
                     BufferFormatEncoding encoding) {
  const uint32_t component_mask = (1u << components.size()) - 1u;
  if (format.number == Number::Unorm && std::ranges::all_of(format.widths, [](uint32_t width) {
        return width == 0 || hwfloat::supports_unorm_width(width);
      })) {
    pack_format(format, selectors, components, bytes, missing, encoding, component_mask);
    return;
  }
  const fp_mode::detail::ScopedFenv environment(0);
  pack_format(format, selectors, components, bytes, missing, encoding, component_mask);
}

} // namespace

bool image_store_selectors_supported(const BufferFormat &format, uint32_t selectors, bool gfx10) {
  if (gfx10) {
    // A one-to-one mapping of all storage channels can be inverted for stores,
    // including a component supplied by a later shader VGPR slot.
    unsigned mapped = 0;
    const unsigned channels = std::count_if(format.widths.begin(), format.widths.end(),
                                            [](unsigned width) { return width != 0; });
    for (unsigned shader = 0; shader < 4; ++shader) {
      const unsigned selector = (selectors >> (3 * shader)) & 7;
      if (selector <= 1)
        continue;
      if (selector < 4 || selector - 4 >= channels || (mapped & (1u << (selector - 4)))) {
        mapped = 0;
        break;
      }
      mapped |= 1u << (selector - 4);
    }
    if (mapped && mapped == ((1u << channels) - 1))
      return true;
  }
  return resolve_store_selectors(format, selectors, true).succeeded();
}

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
                                BufferFormatEncoding encoding, BufferStoreMissingChannel missing) {
  const auto decoded = decode(format, encoding);
  if (decoded.failed() || components.empty() || components.size() > 4)
    return util::Result::failure();
  const auto resolved = resolve_store_selectors(decoded.value(), selectors);
  if (resolved.failed())
    return util::Result::failure();
  pack_components(decoded.value(), resolved.value(), components, bytes, missing, encoding);
  return util::Result::success();
}

util::FailureOr<bool> prepare_buffer_format_state(VectorMemState &d) {
  const auto decoded = decode(d.buffer_format, d.buffer_format_encoding);
  if (decoded.failed())
    return util::Result::failure();
  d.decoded_buffer_format = decoded.value();
  uint32_t bits = 0;
  for (uint32_t width : d.decoded_buffer_format.widths)
    bits += width;
  if (!bits)
    return false;
  d.elem_size = bits / 8;
  return true;
}

util::FailureOr<bool> prepare_gfx10_image_format(VectorMemState &d) {
  using enum BufferNumberFormat;
  if (d.buffer_format >= 169 && d.buffer_format <= 178) {
    d.image_bc = std::make_unique<VectorMemState::BlockCompressedImage>();
    d.image_bc->format = d.buffer_format;
    const unsigned kind = (d.buffer_format - 169) / 2; // BC1 through BC5
    d.image_srgb = kind < 3 && d.buffer_format % 2 == 0;
    d.elem_size = kind == 0 || kind == 3 ? 8 : 16;
    d.decoded_buffer_format = kind < 3    ? Format{{8, 8, 8, 8}, Unorm}
                              : kind == 3 ? Format{{32, 0, 0, 0}, Float}
                                          : Format{{32, 32, 0, 0}, Float};
    return true;
  }
  switch (d.buffer_format) {
  case 133:
    d.decoded_buffer_format = {{5, 6, 5, 0}, Unorm};
    break;
  case 134:
    d.decoded_buffer_format = {{5, 5, 5, 1}, Unorm};
    break;
  case 135:
    d.decoded_buffer_format = {{1, 5, 5, 5}, Unorm};
    break;
  case 136:
    d.decoded_buffer_format = {{4, 4, 4, 4}, Unorm};
    break;
  case 137:
    d.decoded_buffer_format = {{4, 4, 0, 0}, Unorm};
    break;
  case 141:
    d.decoded_buffer_format = {{24, 8, 0, 0}, Unorm};
    break;
  case 142:
    d.decoded_buffer_format = {{24, 8, 0, 0}, Uint};
    break;
  case 143:
    d.decoded_buffer_format = {{8, 24, 0, 0}, Unorm};
    break;
  case 144:
    d.decoded_buffer_format = {{8, 24, 0, 0}, Uint};
    break;
  default:
    return prepare_buffer_format_state(d);
  }
  unsigned bits = 0;
  for (unsigned width : d.decoded_buffer_format.widths)
    bits += width;
  d.elem_size = bits / 8;
  return true;
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
  if ((arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
       arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5) &&
      (word3 >> 30) != 0) {
    d.exec_mask = 0;
    return false;
  }
  d.buffer_format_encoding = arch_is_cdna_4_or_lower(arch)      ? BufferFormatEncoding::Gfx9
                             : arch == ROCJITSU_CODE_ARCH_RDNA1 ? BufferFormatEncoding::Rdna1
                             : arch == ROCJITSU_CODE_ARCH_RDNA2 ? BufferFormatEncoding::Rdna2
                                                                : BufferFormatEncoding::Gfx11;
  d.buffer_format = format < 0 ? addr_calc::rdna_buffer_resource_format(arch, word3)
                               : static_cast<uint32_t>(format);
  if (format < 0 && d.buffer_format_encoding == BufferFormatEncoding::Gfx9)
    d.buffer_format = ((word3 >> 15) & 15) | (((word3 >> 12) & 7) << 4);
  d.buffer_selectors = format < 0 ? word3 & 0xfff : 4 | (5 << 3) | (6 << 6) | (7 << 9);
  // An explicit typed format cannot bind an INVALID resource descriptor.
  const uint32_t resource_format = d.buffer_format_encoding == BufferFormatEncoding::Gfx9
                                       ? (word3 >> 15) & 15
                                       : addr_calc::rdna_buffer_resource_format(arch, word3);
  if (resource_format == 0)
    return false;
  const auto prepared = prepare_buffer_format_state(d);
  if (prepared.failed() || !prepared.value())
    return prepared;
  if (d.is_load && format < 0 && d.buffer_format_encoding == BufferFormatEncoding::Gfx11) {
    // RDNA3/4 untyped loads wrap channel selectors to the memory format's
    // channel count, for both ordinary and D16 destinations. Typed loads
    // instead zero absent channels. RDNA3.5 shares the Gfx11 format policy.
    const uint32_t channels =
        std::count_if(d.decoded_buffer_format.widths.begin(), d.decoded_buffer_format.widths.end(),
                      [](uint32_t width) { return width != 0; });
    if (channels) {
      uint32_t resolved = 0;
      for (uint32_t i = 0; i < 4; ++i) {
        uint32_t selector = (d.buffer_selectors >> (3 * i)) & 7;
        if (selector >= 4)
          selector = 4 + (selector - 4) % channels;
        resolved |= selector << (3 * i);
      }
      d.buffer_selectors = resolved;
    }
  }
  if (!d.is_load && format < 0) {
    const auto resolved = resolve_store_selectors(d.decoded_buffer_format, d.buffer_selectors);
    if (resolved.failed())
      return util::Result::failure();
    d.buffer_selectors = resolved.value();
  }
  return true;
}

void capture_buffer_format_store(Wavefront &wf, VectorMemState &d, uint32_t data_base, bool raw) {
  // Native gfx1201 replicates X and gfx1100 zero-fills. Other families use the
  // extrapolated zero-fill policy described by BufferStoreMissingChannel.
  const auto missing = wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4
                           ? BufferStoreMissingChannel::ReplicateFirst
                           : BufferStoreMissingChannel::Zero;
  RegisterAccess regs(wf);
  const uint32_t registers = d.buffer_d16 ? (d.buffer_components + 1) / 2 : d.buffer_components;
  // Image stores on GFX11 and GFX12 substitute VGPR0 for each unallocated source word.
  const bool image = d.image_store_component_mask && has_gfx11_image_operands(wf.cu().arch());
  const bool partial_word = d.buffer_d16 && (d.buffer_components & 1);
  const uint8_t half_mask = d.d16_hi ? 0xc : 0x3;
  uint32_t allocated = registers;
  if (image) {
    const uint32_t first = data_base - wf.vgpr_alloc().base;
    allocated = std::min(registers, wf.num_vgprs() - std::min(first, wf.num_vgprs()));
  }
  // Buffer tuples crossing the wave's VGPR block read as zero, including their owned prefix.
  const bool allocated_half = partial_word && allocated == registers &&
                              (image || regs.owns_vgpr_range(data_base, registers));
  const auto data = regs.read_vgpr_region(data_base, allocated - allocated_half, d.lane_mask);
  std::optional<RegisterAccess::VgprReadRegion> tail;
  if (allocated_half)
    tail.emplace(regs.read_vgpr_region(data_base + allocated - 1, 1, d.lane_mask, half_mask));
  else if (allocated < registers) {
    const bool only_half = partial_word && allocated + 1 == registers;
    tail.emplace(
        regs.read_vgpr_region(wf.vgpr_alloc().base, 1, d.lane_mask, only_half ? half_mask : 0xf));
  }
  const auto source_word = [&](uint32_t word, uint32_t lane) {
    return word < allocated - allocated_half ? data.lane(word, lane) : tail->lane(0, lane);
  };
  const auto &format = d.decoded_buffer_format;
  const fp_mode::detail::ScopedFenv environment(0);
  const auto *transfer =
      d.image_sample && d.image_sample->transfer ? &*d.image_sample->transfer : nullptr;
  if (transfer) {
    for (auto &tap : d.image_sample->taps)
      tap.response.resize(d.wf_size * d.elem_size);
  } else {
    d.store_data.resize(d.wf_size * d.elem_size);
  }
  for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
    if (!(d.lane_mask & (1ULL << lane)))
      continue;
    std::array<uint32_t, 4> components{};
    uint32_t remaining_image_components = transfer ? transfer->dmask : d.image_store_component_mask;
    for (uint32_t i = 0; i < d.buffer_components; ++i) {
      uint32_t shader_component = i;
      if (remaining_image_components) {
        shader_component = std::countr_zero(remaining_image_components);
        remaining_image_components &= remaining_image_components - 1;
      }
      if (!d.buffer_d16) {
        components[shader_component] = source_word(i, lane);
        continue;
      }
      const uint32_t shift = d.d16_hi ? 16 : (i % 2) * 16;
      const uint16_t half = source_word(i / 2, lane) >> shift;
      components[shader_component] =
          buffer_format_is_integer(format.number)
              ? format.number == Number::Sint
                    ? static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(half)))
                    : half
              : std::bit_cast<uint32_t>(util::f16_to_f32(half));
    }
    if (transfer) {
      const unsigned width = transfer->slots_per_texel;
      for (unsigned i = 0; i < d.image_sample->taps.size(); ++i) {
        auto &tap = d.image_sample->taps[i];
        if (!(tap.lane_mask & (uint64_t{1} << lane)))
          continue;
        auto bytes = std::span(tap.response).subspan(lane * d.elem_size, d.elem_size);
        if (transfer->raw) {
          std::memcpy(bytes.data(), &components[i], bytes.size());
        } else {
          const unsigned mask = (transfer->dmask >> (i * width)) & ((1u << width) - 1);
          pack_format(format, d.buffer_selectors, std::span(components).subspan(i * width, width),
                      bytes, missing, d.buffer_format_encoding, mask);
        }
      }
      continue;
    }
    const bool image_store = d.image_store_component_mask != 0;
    const uint32_t component_mask =
        image_store ? d.image_store_component_mask : (1u << d.buffer_components) - 1u;
    const auto shader_components =
        image_store ? std::span(components) : std::span(components).first(d.buffer_components);
    pack_format(format, d.buffer_selectors, shader_components,
                std::span(d.store_data).subspan(lane * d.elem_size, d.elem_size), missing,
                d.buffer_format_encoding, component_mask, raw);
  }
}

namespace {
std::array<uint32_t, 4> image_texel_values(const VectorMemState &state,
                                           std::span<const uint8_t> bytes,
                                           uint8_t block_texel = 0) {
  std::array<uint8_t, 8> decoded_bytes{};
  if (state.image_bc && !bytes.empty()) {
    const uint32_t format = state.image_bc->format;
    const uint32_t kind = (format - 169) / 2;
    if (kind < 3) {
      const auto rgba = decode_image_bc(kind, block_texel, bytes);
      std::copy(rgba.begin(), rgba.end(), decoded_bytes.begin());
      bytes = std::span(decoded_bytes).first(4);
    } else {
      const bool signed_format = format % 2 == 0;
      const unsigned count = kind == 3 ? 1 : 2;
      for (unsigned c = 0; c < count; ++c) {
        const int32_t value =
            decode_image_bc_scalar(bytes.subspan(c * 8, 8), block_texel, signed_format);
        const float channel = float(value) / (64 * (signed_format ? 127 : 255));
        std::memcpy(decoded_bytes.data() + c * 4, &channel, sizeof(channel));
      }
      bytes = std::span(decoded_bytes).first(count * 4);
    }
  }
  auto values = unpack_format(state.decoded_buffer_format, state.buffer_selectors, bytes);
  const uint32_t channels = state.image_sample && state.image_sample->transfer
                                ? state.image_sample->transfer->slots_per_texel
                                : state.buffer_components;
  for (uint32_t component = 0; component < channels; ++component) {
    const uint32_t selector = (state.buffer_selectors >> (3 * component)) & 7;
    if (state.image_srgb && selector >= 4 && selector <= 6) {
      const float value = std::bit_cast<float>(values[component]);
      float linear = value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
      // GFX9/10 use the analytic IEC sRGB transfer. Their hardware precision is unqualified.
      // gfx1100 and gfx1201 captures agree with rounding decoded channels to BF16.
      if (state.buffer_format_encoding == BufferFormatEncoding::Gfx11)
        linear = util::bf16_to_f32(util::f32_to_bf16_rne(linear));
      values[component] = std::bit_cast<uint32_t>(linear);
    }
    if (state.image_sample && !state.image_sample->transfer && !state.image_sample->gather &&
        state.decoded_buffer_format.number == Number::Float) {
      const uint32_t magnitude = values[component] & 0x7fffffffu;
      if (magnitude > 0x7f800000u)
        values[component] = 0xffc00000u;
      else if (magnitude < 0x00800000u)
        values[component] &= 0x80000000u;
    }
  }
  return values;
}

// A horizontal transfer packs DMASK-selected slots in increasing order.
std::array<uint32_t, 4> group_image_values(const VectorMemState &state, uint32_t lane) {
  const auto &group = *state.image_sample;
  const auto &transfer = *group.transfer;
  std::array<uint32_t, 4> values{};
  for (unsigned tap_index = 0; tap_index < group.taps.size(); ++tap_index) {
    const auto &tap = group.taps[tap_index];
    if (!(tap.lane_mask & (uint64_t{1} << lane)))
      continue;
    const auto bytes = std::span(tap.response).subspan(lane * state.elem_size, state.elem_size);
    std::array<uint32_t, 4> texel{};
    if (transfer.raw) {
      std::memcpy(&texel[0], bytes.data(), bytes.size());
      if (transfer.sign_extend && bytes.size() < sizeof(uint32_t)) {
        const unsigned width = bytes.size() * 8;
        const uint32_t sign = 1u << (width - 1);
        texel[0] = (texel[0] ^ sign) - sign;
      }
    } else {
      texel = image_texel_values(state, bytes,
                                 state.image_bc ? state.image_bc->texels[tap_index][lane] : 0);
    }
    for (unsigned component = 0; component < transfer.slots_per_texel; ++component) {
      const unsigned slot = tap_index * transfer.slots_per_texel + component;
      if (transfer.dmask & (1u << slot))
        values[std::popcount(transfer.dmask & ((1u << slot) - 1))] = texel[component];
    }
  }
  return values;
}

// Compare each tap before spatial or mip filtering.
std::array<uint32_t, 4> image_tap_values(const VectorMemState &state, uint32_t lane,
                                         const ImageSampleAccess::Tap &tap) {
  std::array<uint32_t, 4> values{};
  if (tap.lane_mask & (uint64_t{1} << lane)) {
    const auto bytes = std::span(tap.response).subspan(lane * state.elem_size, state.elem_size);
    const auto index = &tap - state.image_sample->taps.data();
    values =
        image_texel_values(state, bytes, state.image_bc ? state.image_bc->texels[index][lane] : 0);
  }
  if (state.image_sample->border_color && (tap.border_mask & (uint64_t{1} << lane))) {
    const auto number = state.decoded_buffer_format.number;
    const uint32_t one =
        number == Number::Uint || number == Number::Sint ? 1u : std::bit_cast<uint32_t>(1.0f);
    for (uint32_t i = 0; i < values.size(); ++i) {
      const uint32_t selector = (state.buffer_selectors >> (3 * i)) & 7u;
      if (image_border_component_is_one(state.image_sample->border_color, selector))
        values[i] = one;
    }
  }
  if (const auto &compare = state.image_sample->depth_compare) {
    const float ref = compare->reference[lane];
    for (auto &value : values) {
      const float depth = std::bit_cast<float>(value);
      const bool pass = image_depth_comparison(compare->function, ref, depth);
      value = std::bit_cast<uint32_t>(pass ? 1.0f : 0.0f);
    }
  }
  return values;
}

std::array<uint32_t, 4> image_sample_values(const VectorMemState &state, uint32_t lane) {
  if (state.image_status && state.image_status->zero_sample_result(lane))
    return {};
  std::array<uint32_t, 4> selected{};
  if (state.image_sample->packed_row) {
    uint32_t byte = 0;
    for (const auto &tap : state.image_sample->taps) {
      for (uint32_t i = 0; i < state.elem_size; ++i, ++byte)
        if (tap.lane_mask & (uint64_t{1} << lane))
          selected[byte / 4] |= uint32_t{tap.response[lane * state.elem_size + i]}
                                << (8 * (byte % 4));
    }
    return selected;
  }
  if (state.image_sample->gather) {
    constexpr std::array<uint32_t, 4> order{2, 3, 1, 0};
    for (uint32_t component = 0; component < 4; ++component) {
      const auto &tap =
          state.image_sample->taps[state.image_sample->horizontal ? component : order[component]];
      selected[component] = image_tap_values(state, lane, tap)[0];
    }
    return selected;
  }
  const bool preserve_single_tap = !(state.image_sample->mip_blend_mask & (uint64_t{1} << lane));
  std::array<double, 4> sum{};
  // Each group of four taps is one bilinear footprint. Its weights already
  // include the mip weight, so shared-exponent alignment remains per level.
  if (state.image_sample->fp32_bilinear) {
    for (size_t first = 0; first < state.image_sample->taps.size(); first += 4) {
      std::array<std::array<double, 4>, 4> channels{};
      std::array<double, 4> weights{};
      unsigned contributors = 0;
      for (unsigned tap = 0; tap < 4; ++tap) {
        const auto &source = state.image_sample->taps[first + tap];
        weights[tap] = source.weights[lane];
        if (!weights[tap])
          continue;
        ++contributors;
        selected = image_tap_values(state, lane, source);
        if (weights[tap] == 1 && preserve_single_tap)
          return selected;
        for (unsigned component = 0; component < state.buffer_components; ++component)
          channels[component][tap] = std::bit_cast<float>(selected[component]);
      }
      if (!contributors)
        continue;
      for (unsigned component = 0; component < state.buffer_components; ++component) {
        // One spatial contributor still receives the fractional mip weight.
        if (contributors == 1) {
          for (unsigned tap = 0; tap < 4; ++tap)
            if (weights[tap])
              sum[component] += channels[component][tap] * weights[tap];
        } else {
          sum[component] += filter_image_float_texels(channels[component], weights, 0, 25);
        }
      }
    }
    for (unsigned component = 0; component < state.buffer_components; ++component) {
      const double filtered = sum[component];
      if (filtered != 0 && std::isfinite(filtered))
        sum[component] =
            round_image_filter_result(filtered, 1 + std::ilogb(std::abs(filtered)), true);
    }
  } else {
    uint32_t contributors = 0;
    for (const auto &tap : state.image_sample->taps) {
      const float weight = tap.weights[lane];
      if (weight == 0)
        continue;
      auto values = image_tap_values(state, lane, tap);
      selected = values;
      ++contributors;
      if (weight == 1 && contributors == 1 && preserve_single_tap)
        continue;
      for (uint32_t component = 0; component < state.buffer_components; ++component)
        sum[component] += double(std::bit_cast<float>(values[component])) * weight;
    }
    if (contributors == 1 && preserve_single_tap) {
      // prepare_image_sample deliberately collapses duplicate spatial texels into
      // one weight-1 tap. Preserve its integer payload and signed zero, except
      // when an active fractional-mip lane makes its GFX12 quad blend.
      for (const auto &tap : state.image_sample->taps)
        if (tap.weights[lane] == 1)
          return selected;
    }
  }
  for (uint32_t component = 0; component < state.buffer_components; ++component) {
    uint32_t bits = std::bit_cast<uint32_t>(static_cast<float>(sum[component]));
    const uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude < 0x00800000u)
      bits &= 0x80000000u;
    else if (magnitude > 0x7f800000u)
      bits = 0xffc00000u;
    selected[component] = bits;
  }
  return selected;
}
} // namespace

void complete_buffer_format_load(Wavefront &wf, ComputeUnitCore &cu, const VectorMemState &d) {
  if (d.image_status) {
    const auto &status = *d.image_status;
    const uint64_t misses = status.written_misses();
    const uint64_t warnings = status.written_warnings();
    if (warnings && (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA3 ||
                     wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA3_5))
      wf.set_mode_raw(wf.mode_raw() | (1u << 10)); // MODE.LOD_CLAMPED is sticky.
    const uint32_t data_words = d.buffer_d16 ? (d.buffer_components + 1) / 2 : d.buffer_components;
    for (uint64_t lanes = misses | warnings; lanes; lanes &= lanes - 1) {
      const uint32_t lane = std::countr_zero(lanes);
      const uint32_t cause = (misses & (uint64_t{1} << lane)) ? 1u : 2u;
      cu.write_vgpr(d.dst_reg_base + data_words, lane,
                    (uint32_t{status.levels[lane]} << 16) | cause);
    }
  }
  if (d.image_sampling || d.graphics_sample || d.image_bc_format ||
      (d.image_status && !d.image_sample)) {
    complete_image_load(wf, cu, d);
    return;
  }
  const uint32_t registers = d.buffer_d16 ? (d.buffer_components + 1) / 2 : d.buffer_components;
  if (!d.lds_dst && !cu.owns_vgpr_range(wf, d.dst_reg_base, registers))
    return;
  const fp_mode::detail::ScopedFenv environment(0);
  const uint64_t lane_write_mask =
      d.lds_dst ? lds_destination_write_mask(d, cu.arch()) : d.exec_mask;
  for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
    if (!(lane_write_mask & (uint64_t{1} << lane)))
      continue;
    std::array<uint32_t, 4> values;
    if (d.image_sample) {
      values =
          d.image_sample->transfer ? group_image_values(d, lane) : image_sample_values(d, lane);
    } else {
      const auto bytes = d.lane_mask & (1ULL << lane)
                             ? std::span(d.response_data).subspan(lane * d.elem_size, d.elem_size)
                             : std::span<const uint8_t>{};
      values = image_texel_values(d, bytes,
                                  d.image_bc && !bytes.empty() ? d.image_bc->texels[0][lane] : 0);
    }
    write_format_load_lane(wf, cu, d, lane, values);
  }
}

} // namespace rocjitsu::amdgpu
