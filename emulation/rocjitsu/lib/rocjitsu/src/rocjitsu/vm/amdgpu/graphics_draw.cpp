// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/graphics_draw.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/vm/amdgpu/buffer_format.h"
#include "rocjitsu/vm/amdgpu/cpu_dispatch_pool.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/image_address.h"
#include "rocjitsu/vm/amdgpu/image_metadata.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/raster_math.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/log.h"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <format>
#include <stdexcept>
#include <type_traits>

namespace rocjitsu::amdgpu {
namespace {
struct RestoreErrno {
  int value = errno;
  ~RestoreErrno() { errno = value; }
};

constexpr uint32_t kPointList = 1, kTriangleList = 4, kTriangleStrip = 6, kRectangleList = 17;

std::array<uint32_t, 3> strip_vertex_offsets(bool reverse, bool last_provoking) {
  if (!reverse)
    return {0, 1, 2};
  // Reverse winding without moving the selected first or last provoking vertex.
  return last_provoking ? std::array<uint32_t, 3>{1, 0, 2} : std::array<uint32_t, 3>{0, 2, 1};
}

// Factors 13-16 select the unsupported second-source export.
constexpr uint32_t kBlendZero = 0, kBlendOne = 1, kBlendSrcColor = 2, kBlendInvSrcColor = 3,
                   kBlendSrcAlpha = 4, kBlendInvSrcAlpha = 5, kBlendDstAlpha = 6,
                   kBlendInvDstAlpha = 7, kBlendDstColor = 8, kBlendInvDstColor = 9,
                   kBlendSrcAlphaSaturate = 10, kBlendConstantColor = 11,
                   kBlendInvConstantColor = 12, kBlendConstantAlpha = 17,
                   kBlendInvConstantAlpha = 18;
constexpr uint32_t kBlendAdd = 0, kBlendSubtract = 1, kBlendMin = 2, kBlendMax = 3,
                   kBlendReverseSubtract = 4;
constexpr uint32_t kBlendSeparateAlpha = 1u << 29, kBlendEnable = 1u << 30, kDisableRop3 = 1u << 31;
constexpr uint32_t kRopCopy = 12;
constexpr uint32_t kNumberUnorm = 0, kNumberUint = 4, kNumberSint = 5, kNumberSrgb = 6,
                   kNumberFloat = 7;
constexpr uint32_t kBufR8Unorm = 1, kBufR32Uint = 20, kBufRgb10A2Unorm = 36, kBufRgba8Unorm = 42,
                   kBufRg32Uint = 48, kBufRgba16Unorm = 51, kBufRgba32Uint = 61;

constexpr uint32_t kExport32R = 1, kExport32Gr = 2, kExport32Ar = 3, kExportFp16Abgr = 4,
                   kExportUnorm16Abgr = 5, kExportSnorm16Abgr = 6, kExportUint16Abgr = 7,
                   kExportSint16Abgr = 8, kExport32Abgr = 9;

// CB color formats use separate data and number fields; pack_buffer_format uses
// the combined GFX11 buffer format table.
uint32_t color_buffer_format(uint32_t data_format, uint32_t number_format) {
  switch (data_format) {
  case 1: // COLOR_8
    if (number_format <= kNumberSint)
      return 1 + number_format;
    break;
  case 9: // COLOR_2_10_10_10
    if (number_format <= kNumberSint)
      return 36 + number_format;
    break;
  case 10: // COLOR_8_8_8_8
    if (number_format == kNumberSrgb)
      return kBufRgba8Unorm; // Encode RGB separately; alpha remains linear UNORM.
    if (number_format == kNumberUnorm || number_format == kNumberUint ||
        number_format == kNumberSint)
      return kBufRgba8Unorm + number_format;
    break;
  case 2:  // COLOR_16
  case 5:  // COLOR_16_16
  case 12: // COLOR_16_16_16_16
    if (number_format <= kNumberSint || number_format == kNumberFloat) {
      const uint32_t base = data_format == 2 ? 7 : data_format == 5 ? 23 : kBufRgba16Unorm;
      // FLOAT follows SINT in the buffer table, skipping NUMBER_SRGB.
      return base + (number_format == kNumberFloat ? kNumberSint + 1 : number_format);
    }
    break;
  case 4:  // COLOR_32
  case 11: // COLOR_32_32
  case 14: // COLOR_32_32_32_32
    if (number_format == kNumberUint || number_format == kNumberSint ||
        number_format == kNumberFloat) {
      const uint32_t base = data_format == 4    ? kBufR32Uint
                            : data_format == 11 ? kBufRg32Uint
                                                : kBufRgba32Uint;
      return base + (number_format == kNumberFloat ? kNumberSint - kNumberUint + 1
                                                   : number_format - kNumberUint);
    }
    break;
  }
  return 0;
}

bool supported_export_format(uint32_t format, uint32_t components) {
  switch (format) {
  case kExport32R:
    return components == 1;
  case kExport32Gr:
    return components == 2;
  case kExport32Ar:
    return components <= 2;
  case kExportFp16Abgr:
  case kExportUnorm16Abgr:
  case kExportSnorm16Abgr:
  case kExportUint16Abgr:
  case kExportSint16Abgr:
  case kExport32Abgr:
    return true;
  default:
    return false;
  }
}

bool supported_blend(uint32_t control) {
  const uint32_t operation = (control >> 5) & 7;
  const auto supported_factor = [](uint32_t value) {
    return value <= kBlendInvConstantColor || value == kBlendConstantAlpha ||
           value == kBlendInvConstantAlpha;
  };
  return operation <= kBlendReverseSubtract &&
         (operation == kBlendMin || operation == kBlendMax ||
          (supported_factor(control & 31) && supported_factor((control >> 8) & 31)));
}

template <typename T>
T blend_factor(uint32_t factor, uint32_t component, const std::array<T, 4> &source,
               const std::array<T, 4> &destination, const std::array<T, 4> &constant) {
  switch (factor) {
  case kBlendZero:
    return 0;
  case kBlendOne:
    return 1;
  case kBlendSrcColor:
    return source[component];
  case kBlendInvSrcColor:
    return 1 - source[component];
  case kBlendSrcAlpha:
    return source[3];
  case kBlendInvSrcAlpha:
    return 1 - source[3];
  case kBlendDstAlpha:
    return destination[3];
  case kBlendInvDstAlpha:
    return 1 - destination[3];
  case kBlendDstColor:
    return destination[component];
  case kBlendInvDstColor:
    return 1 - destination[component];
  case kBlendSrcAlphaSaturate:
    return component == 3 ? 1 : std::min(source[3], 1 - destination[3]);
  case kBlendConstantColor:
    return constant[component];
  case kBlendInvConstantColor:
    return 1 - constant[component];
  case kBlendConstantAlpha:
    return constant[3];
  case kBlendInvConstantAlpha:
    return 1 - constant[3];
  default:
    throw std::runtime_error("unsupported graphics blend factor");
  }
}

enum class BlendCopy { None, Source, Destination };

// SX can bypass arithmetic when the result equals the source or destination. RGB flags consider
// every exported RGB channel, including channels masked off by CB_TARGET_MASK.
template <typename T>
bool blend_can_copy(BlendCopy from, uint32_t opt, uint32_t component, uint32_t export_mask,
                    const std::array<T, 4> &source) {
  const bool copy_source = from == BlendCopy::Source;
  const uint32_t operation = (opt >> 8) & 7;
  // ADD can copy either input; SUBTRACT and REVSUBTRACT copy source/destination.
  if (operation != 1 && operation != (copy_source ? 2u : 5u))
    return false;
  bool c0 = (export_mask & 7) != 0, c1 = c0;
  for (unsigned i = 0; i < 3; ++i) {
    if (export_mask & (1u << i)) {
      c0 &= source[i] == 0;
      c1 &= source[i] == 1;
    }
  }
  const bool a0 = (export_mask & 8) && source[3] == 0;
  const bool a1 = (export_mask & 8) && source[3] == 1;
  // SX_BLEND_OPT: preserve/ignore all, color zero/one, alpha zero/one, or none.
  const bool ignore[]{true, false, c0, c1, a0, a1, a0, false};
  const bool preserve[]{false, true, c1, c0, a1, a0, false, false};
  if (copy_source)
    return (preserve[opt & 7] || (component == 3 ? a0 : c0)) && ignore[(opt >> 4) & 7];
  return (ignore[opt & 7] || (component == 3 ? a0 : c0)) && preserve[(opt >> 4) & 7];
}

// Constant zero/one equations can bypass arithmetic for the whole pixel.
// Color constants are checked across RGB even for partial exports or writes.
BlendCopy fixed_blend_copy(uint32_t control, uint32_t write_mask,
                           const std::array<float, 4> &constant) {
  const auto factor_is = [&](uint32_t factor, bool one, bool alpha) {
    if (factor == kBlendZero)
      return !one;
    if (factor == kBlendOne)
      return one;
    const bool color = factor == kBlendConstantColor || factor == kBlendInvConstantColor;
    if (!color && factor != kBlendConstantAlpha && factor != kBlendInvConstantAlpha)
      return false;
    const bool inverse = factor == kBlendInvConstantColor || factor == kBlendInvConstantAlpha;
    const uint32_t bits = one != inverse ? 0x3f800000u : 0;
    if (alpha || !color)
      return std::bit_cast<uint32_t>(constant[3]) == bits;
    return std::bit_cast<uint32_t>(constant[0]) == bits &&
           std::bit_cast<uint32_t>(constant[1]) == bits &&
           std::bit_cast<uint32_t>(constant[2]) == bits;
  };
  bool source = true, destination = true;
  for (bool alpha : {false, true}) {
    if (!(write_mask & (alpha ? 8u : 7u)))
      continue;
    const uint32_t equation = alpha && (control & kBlendSeparateAlpha) ? control >> 16 : control;
    const uint32_t operation = (equation >> 5) & 7;
    source &= (operation == kBlendAdd || operation == kBlendSubtract) &&
              factor_is(equation & 31, true, alpha) &&
              factor_is((equation >> 8) & 31, false, alpha);
    destination &= (operation == kBlendAdd || operation == kBlendReverseSubtract) &&
                   factor_is(equation & 31, false, alpha) &&
                   factor_is((equation >> 8) & 31, true, alpha);
  }
  return source ? BlendCopy::Source : destination ? BlendCopy::Destination : BlendCopy::None;
}

template <typename T>
T blend_component(uint32_t control, uint32_t component, const std::array<T, 4> &source,
                  const std::array<T, 4> &destination, const std::array<T, 4> &constant,
                  raster::BlendPrecision precision) {
  const bool fp32 = precision == raster::BlendPrecision::Float32;
  const uint32_t operation = (control >> 5) & 7;
  if (operation == kBlendMin)
    return fp32 ? raster::blend_minmax(source[component], destination[component], false)
                : std::min(source[component], destination[component]);
  if (operation == kBlendMax)
    return fp32 ? raster::blend_minmax(source[component], destination[component], true)
                : std::max(source[component], destination[component]);
  const uint32_t src = control & 31, dst = (control >> 8) & 31;
  const auto factor_mode = [](uint32_t factor) {
    if (factor == kBlendOne)
      return raster::BlendFactorMode::One;
    if (factor == kBlendInvSrcColor || factor == kBlendInvSrcAlpha || factor == kBlendInvDstAlpha ||
        factor == kBlendInvDstColor || factor == kBlendInvConstantColor ||
        factor == kBlendInvConstantAlpha)
      return raster::BlendFactorMode::Inverse;
    return raster::BlendFactorMode::Direct;
  };
  if (precision != raster::BlendPrecision::Unaligned) {
    const auto resolve = [&](uint32_t factor) {
      if (factor != kBlendSrcAlphaSaturate)
        return factor;
      if (component == 3)
        return kBlendOne;
      return raster::blend_saturate_uses_source(source[3], destination[3]) ? kBlendSrcAlpha
                                                                           : kBlendInvDstAlpha;
    };
    const uint32_t sfactor = resolve(src), dfactor = resolve(dst);
    const auto sm = factor_mode(sfactor), dm = factor_mode(dfactor);
    // Inverse factors immediately follow their base factors in the encoding.
    const T sf = blend_factor(sfactor - (sm == raster::BlendFactorMode::Inverse), component, source,
                              destination, constant);
    const T df = blend_factor(dfactor - (dm == raster::BlendFactorMode::Inverse), component, source,
                              destination, constant);
    return raster::blend_products(source[component], sf, sm, destination[component], df, dm,
                                  operation == kBlendReverseSubtract, operation == kBlendSubtract,
                                  precision);
  }
  const T source_term =
      source[component] * blend_factor(src, component, source, destination, constant);
  const T destination_term =
      destination[component] * blend_factor(dst, component, source, destination, constant);
  if (operation == kBlendSubtract)
    return source_term - destination_term;
  if (operation == kBlendReverseSubtract)
    return destination_term - source_term;
  if (operation == kBlendAdd)
    return source_term + destination_term;
  throw std::runtime_error("unsupported graphics blend operation");
}

template <typename T> bool depth_stencil_compare(uint32_t function, T source, T destination) {
  switch (function) {
  case 0:
    return false;
  case 1:
    return source < destination;
  case 2:
    return source == destination;
  case 3:
    return source <= destination;
  case 4:
    return source > destination;
  case 5:
    return source != destination;
  case 6:
    return source >= destination;
  case 7:
    return true;
  }
  return false;
}

uint8_t stencil_operation(uint32_t operation, uint8_t value, uint8_t reference, uint8_t operand) {
  switch (operation) {
  case 0:
    return value;
  case 1:
    return 0;
  case 2:
    return 255;
  case 3:
    return reference;
  case 4:
    return operand;
  case 5:
    return value == 255 ? 255 : value + 1;
  case 6:
    return value == 0 ? 0 : value - 1;
  case 7:
    return ~value;
  case 8:
    return value + 1;
  case 9:
    return value - 1;
  case 10:
    return value & operand;
  case 11:
    return value | operand;
  case 12:
    return value ^ operand;
  case 13:
    return ~(value & operand);
  case 14:
    return ~(value | operand);
  case 15:
    return ~(value ^ operand);
  }
  return value;
}

uint32_t unorm_color_bits(double value, uint32_t width = 8) {
  const double scaled = (std::isnan(value) ? 0 : std::clamp(value, 0.0, 1.0)) * ((1u << width) - 1);
  const auto lower = static_cast<uint32_t>(scaled);
  const double fraction = scaled - lower;
  return lower + (fraction > 0.5 || (fraction == 0.5 && (lower & 1)));
}

// Color-buffer sRGB destinations decode to FP16, independently of the texture
// decoder. These rounded levels reproduce physical RDNA3/4 blend results.
float srgb_color_linear(uint8_t value) {
  static constexpr uint16_t levels[] = {
      0x0000, 0x0cf9, 0x10f9, 0x1376, 0x14f9, 0x1637, 0x1776, 0x185a, 0x18f9, 0x1998, 0x1a37,
      0x1adb, 0x1b88, 0x1c1f, 0x1c7f, 0x1ce4, 0x1d4e, 0x1dbd, 0x1e32, 0x1eab, 0x1f2a, 0x1fae,
      0x201c, 0x2063, 0x20ad, 0x20fa, 0x214a, 0x219d, 0x21f2, 0x224a, 0x22a6, 0x2304, 0x2365,
      0x23c9, 0x2418, 0x244d, 0x2484, 0x24bc, 0x24f6, 0x2532, 0x256f, 0x25ad, 0x25ed, 0x262f,
      0x2673, 0x26b8, 0x26ff, 0x2747, 0x2791, 0x27dd, 0x2815, 0x283d, 0x2865, 0x288f, 0x28b9,
      0x28e4, 0x2910, 0x293d, 0x296a, 0x2999, 0x29c9, 0x29f9, 0x2a2a, 0x2a5d, 0x2a90, 0x2ac4,
      0x2af9, 0x2b2f, 0x2b66, 0x2b9e, 0x2bd7, 0x2c08, 0x2c26, 0x2c44, 0x2c62, 0x2c81, 0x2ca0,
      0x2cc0, 0x2ce0, 0x2d01, 0x2d22, 0x2d44, 0x2d66, 0x2d89, 0x2dad, 0x2dd0, 0x2df5, 0x2e1a,
      0x2e3f, 0x2e65, 0x2e8b, 0x2eb2, 0x2ed9, 0x2f01, 0x2f2a, 0x2f53, 0x2f7c, 0x2fa7, 0x2fd1,
      0x2ffc, 0x3014, 0x302a, 0x3040, 0x3057, 0x306e, 0x3085, 0x309d, 0x30b4, 0x30cc, 0x30e5,
      0x30fd, 0x3116, 0x312f, 0x3149, 0x3162, 0x317c, 0x3197, 0x31b1, 0x31cc, 0x31e7, 0x3203,
      0x321e, 0x323a, 0x3257, 0x3273, 0x3290, 0x32ad, 0x32cb, 0x32e8, 0x3306, 0x3325, 0x3343,
      0x3362, 0x3381, 0x33a1, 0x33c1, 0x33e1, 0x3401, 0x3411, 0x3422, 0x3432, 0x3443, 0x3454,
      0x3465, 0x3476, 0x3488, 0x3499, 0x34ab, 0x34bd, 0x34cf, 0x34e1, 0x34f4, 0x3506, 0x3519,
      0x352c, 0x353f, 0x3552, 0x3565, 0x3578, 0x358c, 0x35a0, 0x35b4, 0x35c8, 0x35dc, 0x35f1,
      0x3605, 0x361a, 0x362f, 0x3644, 0x3659, 0x366f, 0x3684, 0x369a, 0x36b0, 0x36c6, 0x36dc,
      0x36f2, 0x3709, 0x3720, 0x3736, 0x374d, 0x3765, 0x377c, 0x3794, 0x37ab, 0x37c3, 0x37db,
      0x37f3, 0x3806, 0x3812, 0x381f, 0x382b, 0x3838, 0x3844, 0x3851, 0x385e, 0x386b, 0x3877,
      0x3885, 0x3892, 0x389f, 0x38ac, 0x38ba, 0x38c7, 0x38d5, 0x38e2, 0x38f0, 0x38fe, 0x390c,
      0x391a, 0x3928, 0x3936, 0x3944, 0x3953, 0x3961, 0x3970, 0x397e, 0x398d, 0x399c, 0x39ab,
      0x39ba, 0x39c9, 0x39d8, 0x39e7, 0x39f7, 0x3a06, 0x3a16, 0x3a25, 0x3a35, 0x3a45, 0x3a55,
      0x3a65, 0x3a75, 0x3a85, 0x3a95, 0x3aa5, 0x3ab6, 0x3ac6, 0x3ad7, 0x3ae8, 0x3af9, 0x3b09,
      0x3b1a, 0x3b2c, 0x3b3d, 0x3b4e, 0x3b5f, 0x3b71, 0x3b82, 0x3b94, 0x3ba6, 0x3bb8, 0x3bca,
      0x3bdc, 0x3bee, 0x3c00,
  };
  return util::f16_to_f32(levels[value]);
}

// Color-buffer quantization thresholds for FP16 sRGB exports. Each entry is
// the first positive half encoding producing the next byte value. Captures of
// all 65536 FP16 inputs agree on RDNA3 and RDNA4; applying the ideal sRGB curve
// instead changes values near these boundaries.
uint8_t srgb_color_byte(double value) {
  if (!(value > 0))
    return 0;
  if (value >= 1)
    return 255;
  static constexpr uint16_t boundaries[] = {
      0x0900, 0x0f80, 0x1200, 0x1440, 0x1580, 0x16c0, 0x1800, 0x18a0, 0x1940, 0x19e0, 0x1a80,
      0x1b30, 0x1be0, 0x1c50, 0x1cb0, 0x1d20, 0x1d80, 0x1e00, 0x1e70, 0x1ee0, 0x1f60, 0x1ff0,
      0x2040, 0x2088, 0x20d0, 0x2120, 0x2170, 0x21c0, 0x2220, 0x2270, 0x22d0, 0x2330, 0x2390,
      0x2400, 0x2430, 0x2468, 0x24a0, 0x24d8, 0x2510, 0x2550, 0x2590, 0x25d0, 0x2610, 0x2650,
      0x2690, 0x26e0, 0x2720, 0x2770, 0x27b8, 0x2800, 0x2828, 0x2850, 0x2878, 0x28a0, 0x28d0,
      0x28f8, 0x2928, 0x2950, 0x2980, 0x29b0, 0x29e0, 0x2a10, 0x2a40, 0x2a78, 0x2aa8, 0x2ae0,
      0x2b10, 0x2b48, 0x2b80, 0x2bb8, 0x2bf0, 0x2c18, 0x2c34, 0x2c54, 0x2c70, 0x2c90, 0x2cb0,
      0x2cd0, 0x2cf0, 0x2d10, 0x2d34, 0x2d58, 0x2d78, 0x2d98, 0x2dc0, 0x2de0, 0x2e08, 0x2e2c,
      0x2e50, 0x2e78, 0x2ea0, 0x2ec8, 0x2ef0, 0x2f18, 0x2f40, 0x2f68, 0x2f90, 0x2fbc, 0x2fe8,
      0x3008, 0x3020, 0x3034, 0x304c, 0x3064, 0x3078, 0x3090, 0x30a8, 0x30c0, 0x30d8, 0x30f0,
      0x3108, 0x3124, 0x313c, 0x3154, 0x3170, 0x3188, 0x31a4, 0x31c0, 0x31d8, 0x31f4, 0x3210,
      0x322c, 0x3248, 0x3264, 0x3280, 0x32a0, 0x32bc, 0x32d8, 0x32f8, 0x3318, 0x3334, 0x3354,
      0x3370, 0x3390, 0x33b0, 0x33d0, 0x33f0, 0x3408, 0x3418, 0x342a, 0x343c, 0x344c, 0x345c,
      0x346e, 0x3480, 0x3490, 0x34a2, 0x34b4, 0x34c6, 0x34d8, 0x34ea, 0x34fc, 0x3510, 0x3522,
      0x3534, 0x3548, 0x355c, 0x3570, 0x3582, 0x3596, 0x35aa, 0x35be, 0x35d2, 0x35e8, 0x35fc,
      0x3610, 0x3624, 0x3638, 0x3650, 0x3664, 0x3678, 0x3690, 0x36a4, 0x36bc, 0x36d0, 0x36e8,
      0x36fc, 0x3714, 0x372c, 0x3740, 0x3758, 0x3770, 0x3788, 0x37a0, 0x37b8, 0x37d0, 0x37e8,
      0x3800, 0x380c, 0x3818, 0x3824, 0x3832, 0x383e, 0x384a, 0x3858, 0x3864, 0x3870, 0x387e,
      0x388c, 0x3898, 0x38a6, 0x38b4, 0x38c0, 0x38ce, 0x38dc, 0x38e8, 0x38f8, 0x3904, 0x3914,
      0x3920, 0x3930, 0x393c, 0x394c, 0x395a, 0x3968, 0x3978, 0x3986, 0x3994, 0x39a4, 0x39b2,
      0x39c0, 0x39d0, 0x39e0, 0x39f0, 0x39fe, 0x3a0e, 0x3a1c, 0x3a2c, 0x3a3c, 0x3a4c, 0x3a5c,
      0x3a6c, 0x3a7c, 0x3a8c, 0x3a9c, 0x3aae, 0x3abe, 0x3ad0, 0x3ae0, 0x3af0, 0x3b00, 0x3b12,
      0x3b24, 0x3b34, 0x3b44, 0x3b58, 0x3b68, 0x3b7a, 0x3b8c, 0x3b9c, 0x3bb0, 0x3bc0, 0x3bd4,
      0x3be4, 0x3bf8,
  };
  // The color buffer truncates to FP16 before sRGB encoding. Each boundary is
  // the first half encoding that produces the next byte value; blended inputs
  // have already undergone accumulator rounding.
  return static_cast<uint8_t>(std::upper_bound(std::begin(boundaries), std::end(boundaries), value,
                                               [](double linear, uint16_t half) {
                                                 return linear < util::f16_to_f32(half);
                                               }) -
                              std::begin(boundaries));
}
} // namespace

GraphicsDraw::GraphicsDraw(const Pm4QueueState &state, rj_code_arch_t arch, uint32_t vertices,
                           std::vector<uint32_t> indices)
    : arch_(arch), total_vertices_(vertices), instance_count_(state.num_instances),
      primitive_type_(state.uconfig_registers[0x242]), indices_(std::move(indices)),
      sh_(state.sh_registers), context_(state.context_registers),
      attribute_ring_base_(
          addr_calc::buffer_virtual_address(uint64_t{state.uconfig_registers[0x446]} << 16)),
      attribute_ring_bytes_(((state.uconfig_registers[0x447] & 255) + 1) << 16),
      gs_registers_(state.gs_registers) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA3_5 &&
      arch != ROCJITSU_CODE_ARCH_RDNA4)
    throw std::runtime_error("graphics draw requires RDNA3 or RDNA4");
  geometry_shader_ = context_[arch == ROCJITSU_CODE_ARCH_RDNA4 ? 0x2a6 : 0x2d5] & (1u << 5);
  geometry_vertices_ = context_[0x2ce] & 0x7ff;
  if (!instance_count_ || !vertices || vertices > (1u << 20) ||
      (primitive_type_ != kTriangleList && primitive_type_ != kTriangleStrip &&
       primitive_type_ != kRectangleList && !(geometry_shader_ && primitive_type_ == kPointList)))
    throw std::runtime_error("unsupported graphics primitive=" + std::to_string(primitive_type_) +
                             " vertices=" + std::to_string(vertices) +
                             " instances=" + std::to_string(instance_count_));
  if (geometry_shader_ && (state.uconfig_registers[0x266] != 2 || !geometry_vertices_ ||
                           geometry_vertices_ > 64 || (context_[0x2cf] & 1)))
    throw std::runtime_error("unsupported graphics geometry shader output or invocation count");
  const auto &ctx = state.context_registers;
  const bool gfx12 = arch == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t count_control = ctx[gfx12 ? 0x18 : 1];
  count_occlusion_ = count_control & (1u << 8);
  if (count_occlusion_ &&
      ((count_control & ~0x1100010eu) || (count_control & 0x11000000u) != 0x11000000u))
    throw std::runtime_error("unsupported graphics occlusion counter mode");
  const uint32_t shader_control = ctx[gfx12 ? 0x1b : 0x203];
  // RADV programs early/late Z and clears both EXEC preservation bits only
  // when an implicit early test may omit the shader. Explicit early tests,
  // discard, depth/coverage exports and ordered shaders keep the original path.
  // EARLY_Z_THEN_LATE_Z, ALPHA_TO_MASK_DISABLE and DUAL_QUAD_DISABLE.
  constexpr uint32_t allowed_shader_control = (1u << 4) | (1u << 11) | (1u << 15);
  const uint32_t aa_control = ctx[0x2f8];
  // PA_SC_AA_CONFIG: sample count, exposed samples, detail-to-exposed mode,
  // then GFX12 sample iteration or GFX11 coverage selection/encoding.
  const uint32_t sample_modes = 7u | (7u << 20) | (3u << 24) | (gfx12 ? 3u << 30 : 7u << 26);
  // GFX12 RADV requests stencil reads even with the effective stencil test
  // disabled. GFX11 instead uses this register for viewport-clamp control.
  const uint32_t allowed_override = gfx12 ? 1u << 12 : 1u << 16;
  // DB_RENDER_CONTROL: compression-disable hints and maximum wave tile count.
  constexpr uint32_t allowed_render_control = (3u << 5) | (15u << 20);
  // DB_EQAA: sample counts and over-rasterization, including post-Z coverage.
  constexpr uint32_t eqaa_coverage = 0x7777u | (15u << 24);
  // Conservative rasterization: over/under enable and coverage/pre/post-Z masks.
  // Bit20 (NULL_SQUAD_AA_MASK_ENABLE) is RADV's ordinary disabled-mode value.
  constexpr uint32_t conservative_coverage = (1u << 0) | (1u << 5) | (7u << 21);
  // DB_RENDER_OVERRIDE2: validation/decompression and hierarchical comparisons.
  constexpr uint32_t override_comparisons = (3u << 7) | (0x3ffu << 11);
  // Pipeline statistics count actual invocations; enabling them does not forbid
  // the early omission permitted by the shader state below.
  early_depth_state_ =
      (shader_control & ~allowed_shader_control) == 0 && (shader_control & (1u << 4)) &&
      !state.performance_counters_active && !state.unsupported_pixel_counter_mode &&
      ctx[gfx12 ? 0x18 : 1] == 0 &&      // DB_COUNT_CONTROL
      ctx[gfx12 ? 0x194 : 0x1c4] == 0 && // SPI_SHADER_Z_FORMAT
      !(aa_control & sample_modes) && !(ctx[gfx12 ? 0x1e : 0x201] & eqaa_coverage) &&
      !(ctx[0x293] & (1u << 16)) && // PA_SC_MODE_CNTL_1.PS_ITER_SAMPLE
      !(ctx[gfx12 ? 0x315 : 0x313] & conservative_coverage) &&
      !(ctx[0] & ~allowed_render_control) && !(ctx[3] & ~allowed_override) &&
      !(ctx[4] & override_comparisons);
  for (uint32_t target = 0; target < colors_.size(); ++target) {
    auto &color = colors_[target];
    const uint32_t dst = 0x318 + 9 * target;
    if (arch == ROCJITSU_CODE_ARCH_RDNA4) {
      color.max_mip = (ctx[dst + 7] >> 19) & 31;
      color.mip = ctx[dst + 2] & 31;
      continue;
    }
    // GFX11 color blocks have fifteen registers; GFX12 blocks have nine.
    // Always read the original snapshot because the two layouts overlap.
    const uint32_t src = 0x318 + 15 * target;
    if (ctx[src + 6] & (1u << 22))
      color.metadata = addr_calc::buffer_virtual_address(
          ((uint64_t{ctx[0x3a8 + target] & 255} << 32) | ctx[src + 13]) << 8);
    context_[dst] = ctx[src];
    context_[dst + 1] = ctx[src + 3];
    context_[dst + 2] = 0;
    context_[dst + 3] = ctx[src + 5];
    const uint32_t attrib2 = ctx[0x3b0 + target];
    context_[dst + 6] = (attrib2 & 0x3fff) | (((attrib2 >> 14) & 0x3fff) << 16);
    context_[dst + 7] = ctx[0x3b8 + target];
    context_[0x3b0 + target] = ctx[src + 4];
    color.max_mip = attrib2 >> 28;
    color.mip = (ctx[src + 3] >> 26) & 15;
  }
  if (arch != ROCJITSU_CODE_ARCH_RDNA4) {
    // Normalize relocated registers into the GFX12 slots used below. Read the
    // original snapshot because several source and destination slots overlap.
    if ((ctx[0x200] & 3) && (ctx[0x10] & (1u << 29))) {
      if (!(ctx[0x2af] & (1u << 18)))
        throw std::runtime_error("unsupported GFX11 unaligned HTILE surface");
      depth_metadata_ =
          addr_calc::buffer_virtual_address(((uint64_t{ctx[0x1e] & 255} << 32) | ctx[5]) << 8);
      depth_clear_ = ctx[0xb];
      depth_metadata_has_stencil_ = (ctx[0x11] & 1) && !(ctx[0x11] & (1u << 29));
      stencil_clear_ = ctx[0xa];
    }
    for (const auto &[dst, src] : {std::pair{0x1b, 0x203},
                                   {0x1c, 0x200},
                                   {0x190, 0x1b6},
                                   {0x194, 0x1c4},
                                   {0x195, 0x1c5},
                                   {0x197, 0x1b3},
                                   {0x198, 0x1b4},
                                   {0x115, 0xb4},
                                   {0x116, 0xb5},
                                   {0x205, 0x206},
                                   {0x207, 0x205},
                                   {0x206, 0x207},
                                   {0x214, 0x8e},
                                   {0x215, 0x8f},
                                   {0x216, 0x202}})
      context_[dst] = ctx[src];
    context_[0x19] = (ctx[3] >> 16) & 1; // DISABLE_VIEWPORT_CLAMP
    for (uint32_t i = 0; i < 32; ++i)
      context_[0x199 + i] = ctx[0x191 + i];
    sh_[0x31] =
        ((ctx[0x1b1] >> 1) & 31) | (((ctx[0x1b1] >> 8) & 31) << 5) | ((ctx[0x1b6] & 63) << 11);
    sh_[0x84] = state.sh_registers[0x88];
    sh_[0x85] = state.sh_registers[0x89];
    context_[5] = (ctx[7] & 0x3fff) | (((ctx[7] >> 16) & 0x3fff) << 16);
    // MAXMIP moves from bits 16:19 to 15:19 on GFX12.
    context_[6] = (ctx[0x10] & ~0xf8000u) | ((ctx[0x10] & 0xf0000u) >> 1);
    context_[7] = ctx[0x11]; // DB_STENCIL_INFO
    context_[0xc] = ctx[0x13];
    context_[0xd] = ctx[0x1b];
    context_[0xe] = ctx[0x15];
    context_[0xf] = ctx[0x1d];
    context_[0x1d] = ctx[0x10b];
    for (const auto &[dst, shift] : {std::pair{0x22, 0}, {0x23, 24}, {0x24, 8}, {0x25, 16}})
      context_[dst] = ((ctx[0x10c] >> shift) & 255) | (((ctx[0x10d] >> shift) & 255) << 8);
    context_[8] = ctx[0x12];
    context_[9] = ctx[0x1a];
    context_[10] = ctx[0x14];
    context_[11] = ctx[0x1c];
    context_[1] = (ctx[2] & 0x1fff) | (((ctx[2] >> 13) & 0x7ff) << 16) | ((ctx[2] >> 30) << 27);
    context_[2] = ctx[2] & 0x3f000000;
  }
  attribute_ram_candidate_ = classify_attribute_ram_reads();
  if (!indices_.empty() && indices_.size() != total_vertices_)
    throw std::runtime_error("graphics index count does not match draw");
  if (!indices_.empty() && (state.uconfig_registers[0x24b] & 1)) {
    const uint32_t index_type = state.uconfig_registers[0x243] & 3;
    const uint32_t index_bytes = index_type == 0 ? 2 : index_type == 1 ? 4 : 1;
    const uint32_t index_mask = index_bytes == 4 ? ~0u : (1u << (8 * index_bytes)) - 1;
    const uint32_t restart =
        context_[0x103] & ((state.uconfig_registers[0x24b] & 2) ? ~0u : index_mask);
    if (std::find(indices_.begin(), indices_.end(), restart) != indices_.end()) {
      if (primitive_type_ == kRectangleList)
        throw std::runtime_error("graphics rectangle restart is not implemented");
      std::vector<uint32_t> triangles;
      size_t start = 0;
      const bool strip = primitive_type_ == kTriangleStrip;
      const bool last_provoking = context_[0x207] & (1u << 19);
      for (size_t end = 0; end <= indices_.size(); ++end) {
        if (end != indices_.size() && indices_[end] != restart)
          continue;
        for (size_t i = start; i + 2 < end; i += strip ? 1 : 3) {
          const bool reverse = strip && ((i - start) & 1);
          for (const uint32_t offset : strip_vertex_offsets(reverse, last_provoking))
            triangles.push_back(indices_[i + offset]);
        }
        start = end + 1;
      }
      indices_ = std::move(triangles);
      total_vertices_ = indices_.size();
      primitive_type_ = kTriangleList;
    }
  }
  select_vertex_groups();
}

bool GraphicsDraw::enable_vertex_batching(const GpuVmAccess &memory, uint32_t limit) {
  const bool gfx12 = arch_ == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t stages = context_[gfx12 ? 0x2a6 : 0x2d5];
  // NGG_WAVE_ID_EN is used by streamout and mesh scratch. Those programs,
  // tessellation, real GS, and ordered pixel shaders retain serial launches.
  constexpr uint32_t unsupported_stages =
      (1u << 2) | (1u << 5) | (1u << 12) | (1u << 14) | (3u << 19) | (1u << 24);
  if (limit < 2 || vertex_groups_.empty() || vertex_groups_.front().first_vertex ||
      vertex_groups_.front().instance || (stages & unsupported_stages) ||
      (context_[0x1b] & ((1u << 12) | (1u << 16))) || ((context_[0x1b] >> 4) & 3) != 1 ||
      (sh_[0xb] & ((1u << 6) | (3u << 25))) || (sh_[0x8b] & (1u << 6)) || !attribute_ring_base_ ||
      (sh_[0x31] >> 17))
    return false;
  const uint32_t wave = (stages & (1u << 22)) ? 32 : 64;
  const uint32_t vertices = primitive_type_ == kTriangleStrip ? wave : (wave / 3) * 3;
  if (instance_count_ == 1 && total_vertices_ <= vertices)
    return false;
  // Attribute stores round threads to eight; swizzling interleaves blocks of
  // 32 vertices. Reserve the whole wave, including trailing and padding lanes.
  const uint32_t slot_bytes = wave * attribute_stride();
  limit = std::min({limit, 32u, attribute_ring_bytes_ / slot_bytes});
  const uint64_t groups_per_instance = primitive_type_ == kTriangleStrip
                                           ? (uint64_t{total_vertices_} - 3) / (wave - 2) + 1
                                           : (uint64_t{total_vertices_} + vertices - 1) / vertices;
  limit = std::min<uint64_t>(limit, groups_per_instance * instance_count_);
  if (limit < 2)
    return false;
  const uint64_t bytes = uint64_t{slot_bytes} * limit;
  if (memory.query_access(attribute_ring_base_, bytes, VmAccessKind::Write) !=
      VmAccessOutcome::Complete)
    return false;
  // These borrowed pointers are identity queries only. Every actual shader and
  // raster access still uses the ordinary VM path, including fault delivery.
  const auto ring_host =
      reinterpret_cast<uintptr_t>(memory.resolve_host_pointer(attribute_ring_base_));
  if (!ring_host || ring_host + bytes < ring_host)
    return false;
  const auto visit_host_spans = [&](uint64_t address, uint64_t size, const auto &visit) {
    if (address + size < address)
      return false;
    for (uint64_t done = 0; done < size;) {
      const uint64_t count = std::min(size - done, 4096 - ((address + done) & 4095));
      const auto host =
          reinterpret_cast<uintptr_t>(memory.resolve_host_pointer(address + done, count));
      if (!host || host + count < host || !visit(host, count, done))
        return false;
      done += count;
    }
    return true;
  };
  if (!visit_host_spans(
          attribute_ring_base_, bytes,
          [&](uintptr_t host, uint64_t, uint64_t offset) { return host == ring_host + offset; }))
    return false;
  const auto disjoint = [&](uint64_t address, uint64_t size) {
    return visit_host_spans(address, size, [&](uintptr_t host, uint64_t count, uint64_t) {
      return host >= ring_host + bytes || ring_host >= host + count;
    });
  };
  const auto disjoint_allocation = [&](uint64_t address) {
    const auto [host, size] = memory.host_range(address);
    return host && size && host + size >= host &&
           (host >= ring_host + bytes || ring_host >= host + size);
  };
  // No code extent is carried by the PM4 ABI. An unknown or overlapping code
  // allocation therefore declines batching instead of guessing a shader size.
  if (!disjoint_allocation(vertex_dispatch().kernel_entry_pc))
    return false;
  const uint64_t fragment_pc =
      addr_calc::buffer_virtual_address(((uint64_t{sh_[0x9]} << 32) | sh_[0x8]) << 8);
  if (!disjoint_allocation(fragment_pc))
    return false;
  try {
    prepare_attachments();
  } catch (const std::runtime_error &) {
    // Preserve ordinary validation and failure delivery at stage retirement.
    return false;
  }
  const auto disjoint_surface = [&](uint64_t base, uint64_t slice, uint32_t first, uint32_t last,
                                    uint32_t swizzle) {
    // Pipe/bank XOR bits stay within a swizzle block. The admitted layouts
    // have no mip offsets, and their slice sizes include full storage blocks.
    base &= ~((uint64_t{1} << image_block_log2(gfx12, swizzle)) - 1);
    const uint64_t begin = base + slice * first, end = base + slice * (uint64_t{last} + 1);
    if (begin < base || end < begin)
      return false;
    return disjoint(begin, end - begin);
  };
  const auto disjoint_metadata = [&](uint64_t base, uint32_t width, uint32_t height, uint32_t bytes,
                                     bool depth, bool pipe_aligned, uint32_t first, uint32_t last) {
    const auto range =
        gfx11_metadata_range(base, width, height, bytes, depth, pipe_aligned, first, last);
    return range && disjoint(range->address, range->size);
  };
  for (const auto &color : colors_) {
    if (color.write_mask &&
        (color.max_mip || color.volume ||
         !disjoint_surface(color.base, color.slice_size, color.first_layer, color.last_layer,
                           color.swizzle) ||
         (color.metadata &&
          !disjoint_metadata(*color.metadata, color.width, color.height, color.bytes, false,
                             color.pipe_aligned, color.first_layer, color.last_layer))))
      return false;
  }
  if ((depth_control_ & 3) && ((context_[6] >> 15) & 31))
    return false;
  if ((depth_control_ & 2) && !disjoint_surface(depth_base_, depth_slice_size_, depth_first_layer_,
                                                depth_last_layer_, depth_swizzle_))
    return false;
  if ((depth_control_ & 1) &&
      !disjoint_surface(stencil_base_, stencil_slice_size_, depth_first_layer_, depth_last_layer_,
                        stencil_swizzle_))
    return false;
  if (depth_metadata_ &&
      !disjoint_metadata(*depth_metadata_, depth_width_, depth_height_, depth_bytes_, true, true,
                         depth_first_layer_, depth_last_layer_))
    return false;
  vertex_group_limit_ = limit;
  attribute_slot_bytes_ = slot_bytes;
  first_vertex_ = instance_ = 0;
  select_vertex_groups();
  return true;
}

uint32_t GraphicsDraw::primitive_count(const VertexGroup &group) const {
  return primitive_type_ == kPointList       ? group.count
         : primitive_type_ == kTriangleStrip ? (group.count > 2 ? group.count - 2 : 0)
                                             : group.count / 3;
}

void GraphicsDraw::select_vertex_groups() {
  const bool gfx12 = arch_ == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t wave = (context_[gfx12 ? 0x2a6 : 0x2d5] & (1u << 22)) ? 32 : 64;
  uint32_t limit =
      primitive_type_ == kTriangleStrip || primitive_type_ == kPointList ? wave : (wave / 3) * 3;
  if (geometry_shader_) {
    const uint32_t primitives = wave / geometry_vertices_;
    if (!primitives)
      throw std::runtime_error("geometry shader output exceeds one wave");
    limit = primitive_type_ == kPointList       ? primitives
            : primitive_type_ == kTriangleStrip ? primitives + 2
                                                : primitives * 3;
    limit = std::min(limit, wave);
  }
  vertex_groups_.clear();
  next_raster_group_ = 0;
  for (uint32_t i = 0; i < vertex_group_limit_ && instance_ < instance_count_; ++i) {
    auto &group = vertex_groups_.emplace_back();
    group.count = std::min(total_vertices_ - first_vertex_, limit);
    group.first_vertex = first_vertex_;
    group.instance = instance_;
    group.attribute_offset = i * attribute_slot_bytes_;
    group.output_vertices = group.count;
    group.output_primitives = primitive_count(group);
    if (first_vertex_ + group.count == total_vertices_) {
      first_vertex_ = 0;
      ++instance_;
    } else {
      first_vertex_ += primitive_type_ == kTriangleStrip ? group.count - 2 : group.count;
    }
  }
}

std::optional<DispatchEntry> GraphicsDraw::next_vertex_group() {
  if (instance_ == instance_count_)
    return std::nullopt;
  select_vertex_groups();
  fragments_.clear();
  fragment_exports_.clear();
  fragment_exports_prepared_ = false;
  fragment_stage_ = false;
  return vertex_dispatch();
}

DispatchEntry GraphicsDraw::vertex_dispatch() const {
  DispatchEntry dp;
  const bool gfx12 = arch_ == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t rsrc1 = sh_[0x8a], rsrc2 = sh_[0x8b];
  dp.kernel_wave_size = (context_[gfx12 ? 0x2a6 : 0x2d5] & (1u << 22)) ? 32 : 64;
  if (vertex_groups_.empty() || vertex_groups_.front().count > dp.kernel_wave_size)
    throw std::runtime_error("unsupported graphics wave count");
  uint64_t pc = (uint64_t{sh_[gfx12 ? 0x86 : 0xc9]} << 32) | sh_[gfx12 ? 0x89 : 0xc8];
  pc <<= 8;
  dp.kernel_entry_pc = static_cast<uint64_t>(static_cast<int64_t>(pc << 16) >> 16);
  dp.vgprs_per_wf = ((rsrc1 & 63) + 1) * (dp.kernel_wave_size == 32 ? 8 : 4);
  dp.initial_mode_raw = (rsrc1 >> 12) & 0xff;
  if (rsrc1 & (1u << 31))
    dp.initial_mode_raw |= Wavefront::FP16_OVFL_BIT;
  dp.group_segment_fixed_size = ((rsrc2 >> 19) & 255) * 512;
  dp.wgp_mode = (rsrc1 >> 27) & 1;
  dp.total_wgs = dp.grid_wgs_x = vertex_groups_.size();
  dp.grid_wgs_y = dp.grid_wgs_z = 1;
  dp.grid_yz_valid = true;
  dp.workgroup_size_x = dp.kernel_wave_size;
  dp.grid_size_x = dp.total_wgs * dp.kernel_wave_size;
  dp.grid_size_y = dp.grid_size_z = 1;
  dp.wait_for_predecessors = true;
  return dp;
}

void GraphicsDraw::initialize(Wavefront &wave, uint32_t workgroup, uint32_t wave_index) {
  if (fragment_stage_) {
    auto &batch = fragment_for_dispatch(wave.wg_coord()[0]);
    const uint32_t users = ((sh_[0xb] >> 1) & 31) | ((sh_[0xb] >> 22) & 32);
    for (uint32_t i = 0; i < users; ++i)
      wave.debug_write_sgpr(i, sh_[0xc + i]);
    // One primitive per wave, with parameters at offset zero in its LDS allocation.
    // Parameter loads add the wave's LDS base to this shader-visible offset.
    wave.debug_write_sgpr(users, 0);
    for (uint32_t i = 0; i < batch.parameters.size(); ++i)
      wave.lds().write32(wave.lds_base() + i * 4, batch.parameters[i]);
    // Resolve each register only at its original first write, preserving lazy
    // allocation and FP evaluation order. Other registers never move its lanes.
    std::array<uint32_t *, 22> input_lanes;
    uint32_t resolved_inputs = 0;
    const auto write_input = [&](uint32_t reg, uint32_t lane, uint32_t value) {
      auto *&lanes = input_lanes[reg];
      // Every lane emits a dense prefix, so each new register is the next one.
      // Remember refused storage too, without initializing unused array entries.
      if (reg >= resolved_inputs) {
        lanes = wave.initialization_vgpr_lanes(reg).data();
        ++resolved_inputs;
      }
      if (lanes)
        lanes[lane] = value;
      else
        wave.debug_write_vgpr(reg, lane, value);
    };
    // Decode the lane-invariant barycentric input selection once. The plan
    // contains only byte offsets; register storage is still acquired at the
    // original first write, before the unchanged position arithmetic.
    constexpr uint8_t source_offsets[7][3] = {
        {offsetof(Fragment, i), offsetof(Fragment, j)},
        {offsetof(Fragment, i), offsetof(Fragment, j)},
        {offsetof(Fragment, i), offsetof(Fragment, j)},
        {offsetof(Fragment, pull_model), offsetof(Fragment, pull_model) + sizeof(float),
         offsetof(Fragment, pull_model) + 2 * sizeof(float)},
        {offsetof(Fragment, linear_i), offsetof(Fragment, linear_j)},
        {offsetof(Fragment, linear_i), offsetof(Fragment, linear_j)},
        {offsetof(Fragment, linear_i), offsetof(Fragment, linear_j)}};
    std::array<uint8_t, 15> input_offsets;
    uint32_t input_count = 0;
    for (uint32_t input = 0; input < 7; ++input)
      if (context_[0x198] & (1u << input))
        for (uint32_t component = 0; component < (input == 3 ? 3u : 2u); ++component)
          input_offsets[input_count++] = source_offsets[input][component];
    uint64_t exec = 0;
    for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
      const auto &f = batch.lanes[lane];
      // INPUT_ADDR reserves the VGPRs in architectural order, including
      // inputs whose calculation is disabled in INPUT_ENA.
      uint32_t vgpr = 0;
      const auto put = [&](float value) {
        write_input(vgpr++, lane, std::bit_cast<uint32_t>(value));
      };
      for (uint32_t input = 0; input < input_count; ++input) {
        uint32_t value;
        std::memcpy(&value, reinterpret_cast<const std::byte *>(&f) + input_offsets[input],
                    sizeof(value));
        write_input(vgpr++, lane, value);
      }
      // POS_W_FLOAT supplies W; PERSP_PULL_MODEL above supplies 1/W.
      const std::array<float, 4> position{float(f.x) + 0.5f, float(f.y) + 0.5f, f.z,
                                          raster::reciprocal(f.pull_model[2])};
      for (uint32_t component = 0; component < 4; ++component)
        if (context_[0x198] & (1u << (8 + component)))
          put(position[component]);
      // Single-sample rasterization: sample ID and primitive type are zero.
      if (context_[0x198] & (1u << 13))
        write_input(vgpr++, lane, batch.relative_layer << 16);
      if (context_[0x198] & (1u << 14))
        write_input(vgpr++, lane, f.covered ? 1u : 0u);
      if (context_[0x198] & (1u << 15))
        write_input(vgpr++, lane, (uint32_t(f.x) & 0xffff) | (uint32_t(f.y) << 16));
      if (f.covered)
        exec |= uint64_t{1} << lane;
    }
    wave.set_exec(exec);
    return;
  }
  if (workgroup >= vertex_groups_.size() || wave_index)
    throw std::runtime_error("graphics wave outside supported primitive group");
  const auto &group = vertex_groups_[workgroup];
  const bool gfx12 = arch_ == ROCJITSU_CODE_ARCH_RDNA4;
  // GFX11+ merged GS repurposes PGM_LO/HI_GS as the first two user SGPRs.
  wave.debug_write_sgpr(0, sh_[gfx12 ? 0x84 : 0x88]);
  wave.debug_write_sgpr(1, sh_[gfx12 ? 0x85 : 0x89]);
  // Ordinary merged GS launches carry group counts as well as per-wave counts.
  // The ordered wave ID is zero for our single-wave primitive groups.
  const uint32_t max_outputs =
      geometry_shader_ ? primitive_count(group) * geometry_vertices_ : group.count;
  wave.debug_write_sgpr(2, (max_outputs << 12) | (primitive_count(group) << 22));
  wave.debug_write_sgpr(3, group.count | (primitive_count(group) << 8) | (1u << 28));
  for (uint32_t i = 4; i < 8; ++i)
    wave.debug_write_sgpr(i, 0);
  // Attribute-ring offsets use 512-byte units, independently for each group.
  wave.debug_write_sgpr(5, group.attribute_offset >> 9);
  const uint32_t user_count = ((sh_[0x8b] >> 1) & 31) | ((sh_[0x8b] >> 22) & 32);
  // The merged-stage USER_SGPR count starts at s8; the ring pair is separate.
  for (uint32_t i = 0; i < user_count; ++i)
    wave.debug_write_sgpr(i + 8, sh_[0x8c + i]);
  const uint32_t index_bits = gfx12 ? 9 : 10;
  const bool last_provoking = context_[0x207] & (1u << 19);
  for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
    // Primitive connectivity is local to the primitive group's position exports.
    const uint32_t first =
        primitive_type_ == kTriangleStrip || primitive_type_ == kPointList ? lane : 3 * lane;
    const bool reverse = primitive_type_ == kTriangleStrip && ((group.first_vertex + lane) & 1);
    const auto offsets = primitive_type_ == kPointList
                             ? std::array<uint32_t, 3>{0, 0, 0}
                             : strip_vertex_offsets(reverse, last_provoking);
    wave.debug_write_vgpr(0, lane,
                          (first + offsets[0]) | ((first + offsets[1]) << index_bits) |
                              ((first + offsets[2]) << (2 * index_bits)));
    for (uint32_t i = 1; i < (gfx12 ? 3u : 5u); ++i)
      wave.debug_write_vgpr(i, lane, 0);
    if (geometry_shader_ && !gfx12) {
      wave.debug_write_vgpr(0, lane, (first + offsets[0]) | ((first + offsets[1]) << 16));
      wave.debug_write_vgpr(1, lane, first + offsets[2]);
    }
    if (geometry_shader_)
      wave.debug_write_vgpr(
          gfx12 ? 1 : 2, lane,
          (primitive_type_ == kTriangleList ? group.first_vertex / 3 : group.first_vertex) + lane);
    const uint32_t index = indices_.empty()     ? group.first_vertex + lane
                           : lane < group.count ? indices_[group.first_vertex + lane]
                                                : 0;
    wave.debug_write_vgpr(gfx12 ? 3 : 5, lane, index);
    const uint32_t components = (sh_[0x8b] >> 16) & 3;
    if (components >= (gfx12 ? 1u : 3u))
      wave.debug_write_vgpr(gfx12 ? 4 : 8, lane, group.instance);
  }
}

bool GraphicsDraw::allocate_exports(Wavefront &wave, uint32_t vertices, uint32_t primitives) {
  if (fragment_stage_ || !geometry_shader_)
    return false;
  if (vertices > 64 || primitives > 64) {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return true;
  }
  auto &group = vertex_groups_.at(wave.wg_coord()[0]);
  group.output_vertices = vertices;
  group.output_primitives = primitives;
  return true;
}

void GraphicsDraw::export_mask(Wavefront &wave, uint64_t mask) {
  if (!fragment_stage_)
    return;
  auto &batch = fragment_for_dispatch(wave.wg_coord()[0]);
  for (uint32_t lane = 0; lane < fragment_wave_size_; ++lane)
    batch.lanes[lane].covered &= (mask & (uint64_t{1} << lane)) != 0;
}

void GraphicsDraw::export_lane(Wavefront &wave, uint32_t lane, uint32_t target, uint32_t mask,
                               const std::array<uint32_t, 4> &values) {
  if (fragment_stage_) {
    if (target == 8 && !(mask & ~3u) &&
        (!(mask & 2) || (context_[0x194] == 2 && !(context_[0x1b] & 4)))) {
      auto &fragment = fragment_for_dispatch(wave.wg_coord()[0]).lanes[lane];
      if (mask & 1)
        fragment.z = std::bit_cast<float>(values[0]);
      // SPI_SHADER_32_GR carries depth in X and the stencil test value in Y.
      if (mask & 2) {
        fragment.stencil_reference = values[1];
        fragment.stencil_exported = true;
      }
      return;
    }
    if (target >= colors_.size()) {
      wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
      return;
    }
    auto &batch = fragment_for_dispatch(wave.wg_coord()[0]);
    assert(fragment_exports_prepared_);
    const uint32_t slot = export_slots_[target];
    // Valid exports unused by every attachment have no output consumer. Keep
    // operand validation/reads and coverage updates in export_graphics unchanged.
    if (slot == kColorTargets)
      return;
    assert(batch.export_offset != size_t(-1));
    auto &exported = fragment_exports_[batch.export_offset + slot * 64 + lane];
    for (uint32_t i = 0; i < 4; ++i)
      if (mask & (1u << i))
        exported.values[i] = values[i];
    exported.mask |= mask;
    return;
  }
  auto &group = vertex_groups_.at(vertex_groups_.size() == 1 ? 0 : wave.wg_coord()[0]);
  if (target == 12 && lane < group.output_vertices) {
    for (uint32_t i = 0; i < 4; ++i)
      if (mask & (1u << i))
        group.positions[lane][i] = values[i];
    group.position_masks[lane] |= mask;
  } else if (target == 13 && lane < group.output_vertices && !(context_[0x206] & 0x1813ffffu)) {
    // Unused miscellaneous components may still be exported. Only Z carries
    // the layer and viewport indices; the other enabled consumers need support.
    if (mask & 4)
      group.layer_viewport[lane] = values[2];
  } else if (target == 20 && lane < group.output_primitives && mask == 1) {
    group.primitives[lane] = values[0];
    group.primitive_valid[lane] = true;
  } else {
    wave.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
  }
}

void GraphicsDraw::finish_vertices(const VertexGroup &group) {
  for (uint32_t i = 0; i < group.output_primitives; ++i)
    if (!group.primitive_valid[i])
      throw std::runtime_error("missing graphics primitive export");
  for (uint32_t i = 0; i < group.output_vertices; ++i) {
    if ((context_[0x206] & (1u << 19)) && (group.layer_viewport[i] >> 16))
      throw std::runtime_error("graphics viewport selection is not implemented");
    if (group.position_masks[i] != 15)
      throw std::runtime_error("missing graphics position export");
    util::Logger::vm("position ", i, ": ", std::bit_cast<float>(group.positions[i][0]), ", ",
                     std::bit_cast<float>(group.positions[i][1]), ", ",
                     std::bit_cast<float>(group.positions[i][2]), ", ",
                     std::bit_cast<float>(group.positions[i][3]));
  }
}

GraphicsDraw::FragmentWave &GraphicsDraw::fragment_for_dispatch(uint32_t workgroup) {
  return fragments_.at(fragment_selection_active_ ? fragment_dispatch_indices_.at(workgroup)
                                                  : workgroup);
}

bool GraphicsDraw::select_fragment_waves(const GpuVmAccess &memory) {
  const uint32_t comparison = (depth_control_ >> 4) & 7;
  if (!early_depth_state_ || fragments_.empty() || (depth_control_ & 3) != 2 ||
      (depth_control_ & 8) || depth_bytes_ != 4 || depth_first_layer_ != depth_last_layer_ ||
      (comparison != 1 && comparison != 3 && comparison != 4 && comparison != 6))
    return false;
  const ColorAttachment *color = nullptr;
  for (const auto &attachment : colors_) {
    if (!attachment.write_mask)
      continue;
    if (color || attachment.volume || attachment.first_layer != attachment.last_layer)
      return false;
    color = &attachment;
  }
  // Only integer operations are used in the extra predicate. Normal finite
  // values and signed zero compare independently of host rounding/DAZ/FTZ.
  const auto supported_depth = [](uint32_t bits) {
    const uint32_t magnitude = bits & 0x7fffffffu;
    return magnitude < 0x7f800000u && (!magnitude || magnitude >= 0x00800000u);
  };
  const auto ordered_depth = [](uint32_t bits) {
    if (!(bits & 0x7fffffffu))
      bits = 0;
    return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
  };
  const bool clamp = !(context_[0x19] & 1);
  const uint32_t minimum = context_[0x115], maximum = context_[0x116];
  if (clamp && (!supported_depth(minimum) || !supported_depth(maximum) ||
                ordered_depth(minimum) > ordered_depth(maximum)))
    return false;
  RestoreErrno restore_errno;
  const bool gfx12 = arch_ == ROCJITSU_CODE_ARCH_RDNA4;
  const auto image_address = gfx12 ? gfx12_image_address : gfx11_image_address;
  struct Surface {
    uint64_t base;
    uint32_t bytes, swizzle, pitch, tail_x, tail_y, width, height;
    uint64_t begin = UINT64_MAX, end = 0;
  };
  std::array<Surface, 2> surfaces{};
  surfaces[0] = {image_layer_base(gfx12, depth_base_, depth_slice_size_, depth_first_layer_, 4,
                                  depth_swizzle_),
                 4,
                 depth_swizzle_,
                 depth_pitch_,
                 depth_tail_x_,
                 depth_tail_y_,
                 depth_width_,
                 depth_height_};
  const uint32_t surface_count = color ? 2 : 1;
  if (color)
    surfaces[1] = {image_layer_base(gfx12, color->base, color->slice_size, color->first_layer,
                                    color->bytes, color->swizzle),
                   color->bytes,
                   color->swizzle,
                   color->pitch,
                   color->tail_x,
                   color->tail_y,
                   color->width,
                   color->height};
  // Include every possible color write, before the shader has produced exports.
  // A color alias could otherwise invalidate monotone depth rejection.
  for (const auto &batch : fragments_) {
    if (batch.relative_layer)
      return false;
    for (uint32_t lane = 0; lane < fragment_wave_size_; ++lane) {
      const auto &fragment = batch.lanes[lane];
      if (!fragment.covered)
        continue;
      if (!supported_depth(std::bit_cast<uint32_t>(fragment.z)))
        return false;
      for (auto &surface : std::span{surfaces}.first(surface_count)) {
        if (fragment.x < 0 || fragment.y < 0 || uint32_t(fragment.x) >= surface.width ||
            uint32_t(fragment.y) >= surface.height)
          return false;
        const auto address =
            image_address(surface.base, fragment.x + surface.tail_x, fragment.y + surface.tail_y,
                          surface.pitch, surface.bytes, surface.swizzle);
        if (!address || *address > UINT64_MAX - surface.bytes)
          return false;
        surface.begin = std::min(surface.begin, *address);
        surface.end = std::max(surface.end, *address + surface.bytes);
      }
    }
  }
  std::array<VmRamRange, 2> ranges{};
  for (uint32_t i = 0; i < surface_count; ++i) {
    if (surfaces[i].begin >= surfaces[i].end)
      return false;
    ranges[i] = {surfaces[i].begin, surfaces[i].end - surfaces[i].begin};
  }
  std::vector<uint32_t> indices(fragments_.size());
  auto lease = memory.try_lease_ram(std::span{ranges}.first(surface_count));
  if (!lease)
    return false;
  // Admission proves strict private RAM and disjoint underlying attachment
  // bytes. It pins mappings, not values. Same-draw LESS/LEQUAL writes only lower
  // depth; GREATER/GEQUAL writes only raise it. A failed test stays failed.
  // Shader memory writes, depth exports, feedback and stencil were excluded by
  // the programmed early-Z contract. No pointers survive this inspection.
  const auto depth_bytes = lease->bytes(0);
  uint32_t survivors = 0;
  for (uint32_t wave = 0; wave < fragments_.size(); ++wave) {
    bool rejected = true;
    for (uint32_t lane = 0; lane < fragment_wave_size_; ++lane) {
      const auto &fragment = fragments_[wave].lanes[lane];
      if (!fragment.covered)
        continue;
      const auto &depth = surfaces[0];
      const auto address = image_address(depth.base, fragment.x + depth.tail_x,
                                         fragment.y + depth.tail_y, depth.pitch, 4, depth.swizzle);
      uint32_t previous;
      std::memcpy(&previous, depth_bytes.data() + (*address - depth.begin), sizeof(previous));
      if (!supported_depth(previous)) {
        rejected = false;
        break;
      }
      uint32_t incoming = std::bit_cast<uint32_t>(fragment.z);
      if (clamp) {
        if (ordered_depth(incoming) < ordered_depth(minimum))
          incoming = minimum;
        else if (ordered_depth(incoming) > ordered_depth(maximum))
          incoming = maximum;
      }
      if (depth_stencil_compare(comparison, ordered_depth(incoming), ordered_depth(previous))) {
        rejected = false;
        break;
      }
    }
    if (!rejected)
      indices[survivors++] = wave;
  }
  lease.reset();
  if (survivors == fragments_.size())
    return false;
  indices.resize(survivors);
  fragment_dispatch_indices_.swap(indices);
  fragment_selection_active_ = true;
  return true;
}

const GraphicsDraw::ColorExport &
GraphicsDraw::fragment_export(const FragmentWave &batch, uint32_t lane, uint32_t target) const {
  static constexpr ColorExport absent{};
  if (!fragment_exports_prepared_ || batch.export_offset == size_t(-1) ||
      export_slots_[target] == kColorTargets)
    return absent;
  return fragment_exports_[batch.export_offset + export_slots_[target] * 64 + lane];
}

void GraphicsDraw::prepare_fragment_exports() {
  if (fragment_exports_prepared_)
    return;
  std::array<uint8_t, kColorTargets> slots;
  slots.fill(kColorTargets);
  uint32_t count = 0;
  for (const auto &color : colors_)
    if (color.write_mask && slots[color.export_index] == kColorTargets)
      slots[color.export_index] = static_cast<uint8_t>(count++);
  const size_t stride = 64 * count;
  const size_t waves =
      fragment_selection_active_ ? fragment_dispatch_indices_.size() : fragments_.size();
  if (stride && waves > fragment_exports_.max_size() / stride)
    throw std::length_error("fragment color export storage exceeds host capacity");
  // The prior window is retired before clearing, and every new record starts
  // at zero. Allocation occurs before FS callbacks and outside any RAM lease.
  assert(fragment_exports_.empty());
  fragment_exports_.resize(waves * stride);
  export_slots_ = slots;
  for (size_t wave = 0; wave < waves && stride; ++wave) {
    const size_t index = fragment_selection_active_ ? fragment_dispatch_indices_[wave] : wave;
    fragments_[index].export_offset = wave * stride;
  }
  fragment_exports_prepared_ = true;
}

DispatchEntry GraphicsDraw::fragment_dispatch() const {
  DispatchEntry dp;
  const uint32_t rsrc1 = sh_[0xa];
  dp.kernel_wave_size = fragment_wave_size_;
  dp.kernel_entry_pc =
      addr_calc::buffer_virtual_address(((uint64_t{sh_[0x9]} << 32) | sh_[0x8]) << 8);
  dp.vgprs_per_wf = ((rsrc1 & 63) + 1) * (dp.kernel_wave_size == 32 ? 8 : 4);
  dp.initial_mode_raw = (rsrc1 >> 12) & 255;
  if (rsrc1 & (1u << 29))
    dp.initial_mode_raw |= Wavefront::FP16_OVFL_BIT;
  dp.group_segment_fixed_size = 2048;
  dp.total_wgs = dp.grid_wgs_x =
      fragment_selection_active_ ? fragment_dispatch_indices_.size() : fragments_.size();
  dp.grid_wgs_y = dp.grid_wgs_z = 1;
  dp.grid_yz_valid = true;
  dp.workgroup_size_x = dp.kernel_wave_size;
  dp.grid_size_x = dp.total_wgs * dp.kernel_wave_size;
  dp.grid_size_y = dp.grid_size_z = 1;
  dp.wait_for_predecessors = true;
  return dp;
}

void GraphicsDraw::prepare_colors() {
  const bool gfx12 = arch_ == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t color_control = context_[0x216];
  const uint32_t color_mode = (color_control >> 4) & 7;
  color_enabled_ = (context_[0x214] & context_[0x215]) != 0 && color_mode != 0;
  if (color_enabled_ && (color_mode != 1 || (color_control & 8) ||
                         ((color_control >> 16) & 15) != ((color_control >> 20) & 15)))
    throw std::runtime_error("unsupported graphics color mode, logic operation, or degamma");
  width_ = height_ = 0;
  uint32_t export_index = 0;
  for (uint32_t target = 0; target < colors_.size(); ++target) {
    auto &color = colors_[target];
    const uint32_t shader_mask = (context_[0x215] >> (4 * target)) & 15;
    // Shader exports and their format nibbles omit holes in CB_SHADER_MASK.
    // CB_TARGET_MASK can disable writes without removing a shader export slot.
    color.export_index = export_index;
    color.export_format = (context_[0x195] >> (4 * export_index)) & 15;
    export_index += shader_mask != 0;
    color.write_mask = color_enabled_ ? (context_[0x214] >> (4 * target)) & shader_mask : 0;
    if (!color.write_mask)
      continue;
    const uint32_t block = 0x318 + 9 * target;
    const uint32_t info = context_[0x3b0 + target];
    const uint32_t attrib = context_[block + 3], attrib2 = context_[block + 6],
                   attrib3 = context_[block + 7];
    util::Logger::cp("graphics target ", target, " info=", std::hex, info, " attrib=", attrib, ",",
                     attrib2, ",", attrib3, " export=", color.export_format,
                     " mask=", color.write_mask, std::dec);
    const uint32_t data_format = info & 31, number_format = (info >> 8) & 7;
    color.srgb = number_format == kNumberSrgb;
    color.memory_format = color_buffer_format(data_format, number_format);
    const auto format = decode_buffer_format(color.memory_format);
    if (format.failed())
      throw std::runtime_error("unsupported graphics color format");
    color.bytes = format.value().byte_size();
    color.components = format.value().component_count();
    color.component_widths = format.value().widths;
    const uint32_t swap = (info >> 11) & 3;
    // Map logical RGBA channels to physical components. A missing channel
    // reads as zero (RGB) or one (alpha), and has no storage or write mask.
    constexpr std::array<std::array<uint32_t, 4>, 4> swaps{
        {{0, 1, 2, 3}, {2, 1, 0, 3}, {3, 2, 1, 0}, {1, 2, 3, 0}}};
    color.component_indices = swaps[swap];
    if (color.components == 1) {
      if (swap != 0 && swap != 3)
        throw std::runtime_error("unsupported single-component graphics color swap");
      color.component_indices = swap == 0 ? std::array{0u, 4u, 4u, 4u} : std::array{4u, 4u, 4u, 0u};
    } else if (color.components == 2) {
      constexpr std::array<std::array<uint32_t, 4>, 4> pairs{
          {{0, 1, 4, 4}, {0, 4, 4, 1}, {1, 0, 4, 4}, {1, 4, 4, 0}}};
      color.component_indices = pairs[swap];
    }
    color.selectors = color.channel_mask = 0;
    for (uint32_t c = 0; c < 4; ++c) {
      const uint32_t index = color.component_indices[c];
      const bool present = index < color.components;
      color.selectors |= (present ? 4 + index : c == 3 ? 1 : 0) << (3 * c);
      color.channel_mask |= uint32_t(present) << c;
    }
    color.blend = context_[0x1e0 + target];
    uint32_t allowed_attrib = gfx12 ? 0u : 0x30u;
    // Attachments without alpha decode their destination alpha as one.
    if (color.components < 4)
      allowed_attrib |= 1u << 2;
    const uint32_t layer_bits = gfx12 ? 14 : 13, layer_mask = (1u << layer_bits) - 1;
    const uint32_t view = context_[block + 1];
    color.first_layer = view & layer_mask;
    color.last_layer = (view >> layer_bits) & layer_mask;
    // GFX11 META_LINEAR only controls enabled metadata. GFX12 reserves this bit.
    if (!color.memory_format || (color.srgb && color.export_format != kExportFp16Abgr) ||
        (attrib & ~allowed_attrib) || ((attrib3 >> 24) & 3) > 2 ||
        ((gfx12 || color.metadata) && (attrib3 & (1u << layer_bits))) ||
        (context_[block + 2] & ~31u) || color.first_layer > color.last_layer ||
        color.last_layer > (attrib3 & layer_mask) || (view & (gfx12 ? 0xf0000000u : 0xc0000000u)) ||
        !supported_export_format(color.export_format, color.components))
      throw std::runtime_error(std::format("unsupported graphics color state target={} info={:#x} "
                                           "attrib={:#x},{:#x},{:#x} export={}",
                                           target, info, attrib, attrib2, attrib3,
                                           color.export_format));
    if ((color.blend & kBlendEnable) &&
        ((color.memory_format != kBufR8Unorm && color.memory_format != kBufRgba8Unorm &&
          color.memory_format != kBufRgb10A2Unorm &&
          format.value().number != BufferNumberFormat::Float) ||
         !supported_blend(color.blend) ||
         ((color.blend & kBlendSeparateAlpha) && !supported_blend(color.blend >> 16))))
      throw std::runtime_error("unsupported graphics blend operation or integer attachment");
    color.swizzle = gfx12 ? (attrib3 >> 15) & 7 : (attrib3 >> 14) & 31;
    color.pipe_aligned = attrib3 & (1u << 30);
    if (((attrib3 >> 24) & 3) == 2)
      color.volume = image_volume_mip_layout(gfx12, color.swizzle, color.bytes, (attrib2 >> 16) + 1,
                                             (attrib2 & 0xffff) + 1, (attrib3 & layer_mask) + 1,
                                             color.max_mip + 1, color.mip);
    const auto mip = ((attrib3 >> 24) & 3) == 2
                         ? (color.volume ? std::optional{color.volume->plane} : std::nullopt)
                         : image_mip_layout(gfx12, color.swizzle, color.bytes, (attrib2 >> 16) + 1,
                                            (attrib2 & 0xffff) + 1, color.max_mip + 1, color.mip);
    if (!mip)
      throw std::runtime_error("unsupported graphics color mip layout");
    color.resource_width = (attrib2 >> 16) + 1;
    color.resource_height = (attrib2 & 0xffff) + 1;
    color.width = mip->width;
    color.height = mip->height;
    if (width_ && (width_ != color.width || height_ != color.height))
      throw std::runtime_error("graphics color attachment extents must match");
    width_ = color.width;
    height_ = color.height;
    color.pitch = mip->pitch;
    color.slice_size = mip->slice_size;
    color.tail_x = mip->tail_x;
    color.tail_y = mip->tail_y;
    color.base = addr_calc::buffer_virtual_address(
                     ((uint64_t{context_[0x390 + target] & 255} << 32) | context_[block]) << 8) +
                 mip->offset;
  }
}

void GraphicsDraw::prepare_attachments() {
  const bool gfx12 = arch_ == ROCJITSU_CODE_ARCH_RDNA4;
  prepare_colors();
  depth_control_ = context_[0x1c];
  // An invalid stencil surface disables stencil even when STENCIL_ENABLE is set.
  if (!(context_[7] & 1))
    depth_control_ &= ~1u;
  // Disabled stencil comparisons do not affect depth testing.
  if (depth_control_ & ~0x7007f7u)
    throw std::runtime_error(
        std::format("unsupported graphics depth bounds or control {:#x}", depth_control_));
  if ((context_[0x198] & ~0xef7fu) || (context_[0x197] & ~context_[0x198]))
    throw std::runtime_error(
        std::format("unsupported graphics fragment inputs addr={:#x} ena={:#x}", context_[0x198],
                    context_[0x197]));

  if (depth_control_ & 3) {
    const uint32_t zinfo = context_[6];
    const uint32_t base_width = (context_[5] & 0xffff) + 1;
    const uint32_t base_height = (context_[5] >> 16) + 1;
    depth_swizzle_ = (zinfo >> 4) & 31;
    depth_bytes_ = (zinfo & 3) == 1 ? 2 : 4;
    depth_first_layer_ = context_[1] & 0x3fff;
    depth_last_layer_ = (context_[1] >> 16) & 0x3fff;
    depth_base_ =
        addr_calc::buffer_virtual_address(((uint64_t{context_[9] & 255} << 32) | context_[8]) << 8);
    const uint64_t write_base = addr_calc::buffer_virtual_address(
        ((uint64_t{context_[11] & 255} << 32) | context_[10]) << 8);
    util::Logger::cp("graphics depth info=", std::hex, zinfo, " view=", context_[1], ",",
                     context_[2], " base=", depth_base_, " write=", write_base, std::dec);
    if (((depth_control_ & 2) && (zinfo & 15) != 3 && (zinfo & 15) != 1) ||
        depth_first_layer_ > depth_last_layer_ || (context_[1] & 0xc000c000u) ||
        ((depth_control_ & 4) && ((context_[2] & (1u << 24)) || depth_base_ != write_base)))
      throw std::runtime_error(
          std::format("unsupported graphics depth surface info={:#x} view={:#x},{:#x} size={}x{} "
                      "base={:#x} write={:#x} control={:#x}",
                      zinfo, context_[1], context_[2], base_width, base_height, depth_base_,
                      write_base, depth_control_));
    const uint32_t max_mip = (zinfo >> 15) & 31, level = (context_[2] >> 26) & 31;
    const auto mip = image_mip_layout(gfx12, depth_swizzle_, depth_bytes_, base_width, base_height,
                                      max_mip + 1, level);
    if (!mip || (depth_metadata_ && (max_mip || depth_last_layer_)))
      throw std::runtime_error("unsupported graphics depth mip or metadata layout");
    depth_width_ = mip->width;
    depth_height_ = mip->height;
    depth_pitch_ = mip->pitch;
    depth_tail_x_ = mip->tail_x;
    depth_tail_y_ = mip->tail_y;
    depth_slice_size_ = mip->slice_size;
    depth_base_ += mip->offset;
    if (depth_control_ & 1) {
      stencil_swizzle_ = (context_[7] >> 4) & 31;
      stencil_base_ = addr_calc::buffer_virtual_address(
          ((uint64_t{context_[0xd] & 255} << 32) | context_[0xc]) << 8);
      const uint64_t stencil_write = addr_calc::buffer_virtual_address(
          ((uint64_t{context_[0xf] & 255} << 32) | context_[0xe]) << 8);
      if (stencil_base_ != stencil_write || (context_[2] & (1u << 25)))
        throw std::runtime_error("unsupported graphics stencil write surface");
      const auto stencil_mip =
          image_mip_layout(gfx12, stencil_swizzle_, 1, base_width, base_height, max_mip + 1, level);
      if (!stencil_mip)
        throw std::runtime_error("unsupported graphics stencil mip layout");
      stencil_base_ += stencil_mip->offset;
      stencil_pitch_ = stencil_mip->pitch;
      stencil_tail_x_ = stencil_mip->tail_x;
      stencil_tail_y_ = stencil_mip->tail_y;
      stencil_slice_size_ = stencil_mip->slice_size;
    }
    if (!color_enabled_) {
      width_ = depth_width_;
      height_ = depth_height_;
    } else if (depth_width_ != width_ || depth_height_ != height_) {
      throw std::runtime_error("graphics depth and color extents must match");
    }
  } else if (!color_enabled_) {
    // Fragment shaders can write buffers or images without any attachment writes.
    // Use the window scissor until the viewport scissor is applied below.
    const uint32_t scissor_mask = gfx12 ? 0xffff : 0x7fff;
    width_ = (context_[0x91] & scissor_mask) + uint32_t(gfx12);
    height_ = ((context_[0x91] >> 16) & scissor_mask) + uint32_t(gfx12);
  }
  if ((color_enabled_ || (depth_control_ & 2)) && (width_ > 4096 || height_ > 4096))
    throw std::runtime_error("graphics render target exceeds supported dimensions");
}

bool GraphicsDraw::classify_attribute_ram_reads() const {
  const uint32_t attributes = (sh_[0x31] >> 11) & 63;
  if (attributes > 32 || (sh_[0x31] >> 17))
    return false;
  bool reads_ram = false;
  for (uint32_t a = 0; a < attributes; ++a) {
    const uint32_t control = context_[0x199 + a];
    if (control & ~0x100073fu)
      return false;
    reads_ram |= !(control & 32) || (control & 0x420) == 0x420;
  }
  return reads_ram;
}

bool GraphicsDraw::try_gather_attributes(const GpuVmAccess &memory, const VertexGroup &group,
                                         const std::array<uint32_t, 3> &indices,
                                         uint32_t provoking_index, std::span<uint32_t> words) {
#if defined(__GLIBC__) && defined(__x86_64__)
  if (!attribute_ram_candidate_ || !memory.supports_ram_word_reads())
    return false;
  RestoreErrno restore_errno;
  // The original parameter allocation has already happened. Reordering reads
  // ahead of coefficient math is allowed only without host exception traps.
  std::fenv_t environment;
  if (std::fegetenv(&environment) != 0 || (environment.__control_word & 0x3f) != 0x3f ||
      (environment.__mxcsr & 0x1f80) != 0x1f80)
    return false;
  const uint32_t attributes = (sh_[0x31] >> 11) & 63;
  if (attributes > 32 || words.size() != attributes * 12 || provoking_index >= group.count ||
      std::ranges::any_of(indices, [&](uint32_t index) { return index >= group.count; }) ||
      group.attribute_offset > UINT64_MAX - attribute_ring_base_)
    return false;
  const uint64_t base = attribute_ring_base_ + group.attribute_offset;
  const uint32_t stride = attribute_stride();
  std::array<VmRamWordRead, 32 * 12> reads;
  size_t count = 0;
  uint64_t begin = UINT64_MAX, end = 0;
  for (uint32_t a = 0; a < attributes; ++a) {
    const uint32_t control = context_[0x199 + a];
    if (control & ~0x100073fu)
      return false;
    const bool passthrough = (control & 0x420) == 0x420;
    const bool flat = (control & 0x400) && !passthrough;
    for (uint32_t c = 0; c < 4; ++c) {
      for (uint32_t k = 0; k < 3; ++k) {
        const uint32_t i = a * 12 + c * 3 + k;
        if ((control & 32) && !passthrough) {
          words[i] = ((control >> 8) & (c == 3 ? 1u : 2u)) ? 0x3f800000 : 0;
          continue;
        }
        const auto address = addr_calc::rdna_buffer_address(3u << 30, 2u << 21, stride, true,
                                                            flat ? provoking_index : indices[k],
                                                            (control & 31) * 16 + c * 4, 0, 0);
        if (address.offset > UINT64_MAX - base || base + address.offset > UINT64_MAX - 4)
          return false;
        reads[count++] = {base + address.offset, &words[i]};
        begin = std::min(begin, base + address.offset);
        end = std::max(end, base + address.offset + 4);
      }
    }
  }
  if (!end)
    return true; // Default parameters have no memory accesses.
  // All selected offsets of one 64-vertex group fit this interleaved ring span.
  if (end - begin > 32768)
    return false;
  // Preserve attribute/component/vertex order, including repeated flat reads.
  // Only these DWORDs are copied; neither gaps nor unused ring words are read.
  // The synchronous admission uses stack guards and releases before math/fallback.
  const bool copied =
      memory.try_read_ram_words({begin, end - begin}, std::span{reads}.first(count));
  if (!copied)
    attribute_ram_candidate_ = false;
  return copied;
#else
  (void)memory;
  (void)group;
  (void)indices;
  (void)provoking_index;
  (void)words;
  return false;
#endif
}

void GraphicsDraw::rasterize(const GpuVmAccess &memory, const VertexGroup &group,
                             CpuDispatchPool *pool, uint32_t threads,
                             bool allow_ram_read_batching) {
  const bool gfx12 = arch_ == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t clip_control = context_[0x204];
  if (clip_control & (1u << 22)) // DX_RASTERIZATION_KILL
    return;
  if (clip_control & (0x3fu | (1u << 28)))
    throw std::runtime_error("graphics user clip planes are not implemented");
  if (context_[0x1b] & (1u << 12)) // DB_SHADER_CONTROL.DEPTH_BEFORE_SHADER
    throw std::runtime_error("graphics early fragment tests are not implemented");
  const uint32_t polygon_mode = (context_[0x207] >> 3) & 3;
  if (polygon_mode && (polygon_mode != 1 || ((context_[0x207] >> 5) & 63) != (2 | (2 << 3))))
    throw std::runtime_error("graphics point or line polygon modes are not implemented");
  if ((context_[0x207] & (7u << 11)) == (1u << 13) &&
      ((context_[0x2e0] | context_[0x2e1] | context_[0x2e2] | context_[0x2e3]) & 0x7fffffffu))
    throw std::runtime_error("independent parallel graphics depth bias is not implemented");
  if (context_[0x2f8] & 7)
    throw std::runtime_error("graphics multisample rasterization is not implemented");
  if (context_[0x2f9] != 0x2d)
    throw std::runtime_error("unsupported graphics pixel center or subpixel rounding");
  prepare_attachments();
  fragment_wave_size_ = (context_[0x190] & (1u << 15)) ? 32 : 64;
  const uint32_t attributes = (sh_[0x31] >> 11) & 63;
  if (attributes > 32 || (sh_[0x31] >> 17))
    throw std::runtime_error("unsupported graphics parameter count");
  const uint32_t ring_stride = attribute_stride();
  const float sx = std::bit_cast<float>(context_[0x10f]);
  const float ox = std::bit_cast<float>(context_[0x110]);
  const float sy = std::bit_cast<float>(context_[0x111]);
  const float oy = std::bit_cast<float>(context_[0x112]);
  const float sz = std::bit_cast<float>(context_[0x113]);
  const float oz = std::bit_cast<float>(context_[0x114]);
  if (context_[0x205] != 0x43f || !std::isfinite(sx) || !std::isfinite(sy) || !std::isfinite(ox) ||
      !std::isfinite(oy))
    throw std::runtime_error("unsupported graphics viewport transform");
  const uint32_t scissor_mask = gfx12 ? 0xffff : 0x7fff;
  int left = std::max(0, int(context_[0x90] & scissor_mask));
  int top = std::max(0, int((context_[0x90] >> 16) & scissor_mask));
  // GFX12 changed the bottom-right scissor bounds from exclusive to inclusive.
  int right = std::min(int(width_), int(context_[0x91] & scissor_mask) + int(gfx12));
  int bottom = std::min(int(height_), int((context_[0x91] >> 16) & scissor_mask) + int(gfx12));
  if (context_[0x292] & 2) { // PA_SC_MODE_CNTL_0.VPORT_SCISSOR_ENABLE
    const bool window_offset = !gfx12 && !(context_[0x94] & (1u << 31));
    const int x_offset = window_offset ? int16_t(context_[0x80]) : 0;
    const int y_offset = window_offset ? int16_t(context_[0x80] >> 16) : 0;
    left = std::max(left, int(context_[0x94] & scissor_mask) + x_offset);
    top = std::max(top, int((context_[0x94] >> 16) & scissor_mask) + y_offset);
    right = std::min(right, int(context_[0x95] & scissor_mask) + x_offset + int(gfx12));
    bottom = std::min(bottom, int((context_[0x95] >> 16) & scissor_mask) + y_offset + int(gfx12));
  }
  if (left >= right || top >= bottom)
    return;
  if (right - left > 4096 || bottom - top > 4096)
    throw std::runtime_error("graphics render area exceeds supported dimensions");
  const float gx = std::bit_cast<float>(context_[gfx12 ? 0x10d : 0x2fc]);
  const float gy = std::bit_cast<float>(context_[gfx12 ? 0x10b : 0x2fa]);
  const bool valid_guard = std::isfinite(gx) && std::isfinite(gy) && gx >= 1 && gy >= 1;
  const auto guard_distance = [](double position, double w, float guard, int sign) {
    return raster::vertex_difference(raster::truncate_float(w * guard), -sign * position);
  };
  for (uint32_t p = 0; p < group.output_primitives; ++p) {
    if (group.primitives[p] & (1u << 31))
      continue;
    std::array<uint32_t, 3> indices{};
    struct Point {
      double x, y, w, z;
      std::array<double, 3> barycentric{};
      std::array<double, 3> linear{};
    };
    std::array<Point, 3> v{}, screen{}, clip_positions{};
    uint32_t outside_near = 0, outside_far = 0, nonpositive_w = 0;
    bool clip_xy = false;
    for (uint32_t k = 0; k < 3; ++k) {
      const uint32_t index_bits = gfx12 ? 9 : 10;
      indices[k] = (group.primitives[p] >> (index_bits * k)) & ((1u << index_bits) - 1);
      if (indices[k] >= group.output_vertices)
        throw std::runtime_error("graphics primitive index exceeds vertex exports");
      const auto &position = group.positions[indices[k]];
      const float w = std::bit_cast<float>(position[3]);
      const float x = std::bit_cast<float>(position[0]);
      const float y = std::bit_cast<float>(position[1]);
      if (!std::isfinite(w) || !std::isfinite(x) || !std::isfinite(y)) {
        throw std::runtime_error("graphics homogeneous clipping is not implemented");
      }
      const float z = std::bit_cast<float>(position[2]);
      if (!std::isfinite(z))
        throw std::runtime_error("graphics nonfinite depth is not implemented");
      nonpositive_w += w <= 0;
      clip_positions[k] = {x, y, w, z};
      clip_positions[k].barycentric[k] = 1;
      clip_positions[k].linear[k] = 1;
      if (!(clip_control & (1u << 16))) {
        const float near = (clip_control & (1u << 19)) ? 0.0f : -w;
        outside_near += !(clip_control & (1u << 26)) && z < near;
        outside_far += !(clip_control & (1u << 27)) && z > w;
      }
      v[k] = w == 0 ? Point{0, 0, w, 0}
                    : Point{raster::viewport_coordinate(x, w, sx, ox),
                            raster::viewport_coordinate(y, w, sy, oy), w,
                            raster::viewport_transform(z, w, sz, oz)};
      screen[k] = v[k];
      // Finite homogeneous inputs can overflow the initial FP32 projection.
      // Let guard-band clipping produce finite coordinates before validation.
      clip_xy |= w <= 0 || std::abs(v[k].x) > (1 << 20) || std::abs(v[k].y) > (1 << 20);
      if (valid_guard)
        for (int sign : {1, -1})
          clip_xy |= guard_distance(x, w, gx, sign) < 0 || guard_distance(y, w, gy, sign) < 0;
    }
    // Layer selection uses the provoking vertex before interpolation reorders vertices.
    const uint32_t provoking = (context_[0x207] & (1u << 19)) ? 2 : 0;
    const uint32_t provoking_index = indices[provoking];
    const uint32_t relative_layer =
        (context_[0x206] & (1u << 18)) ? group.layer_viewport[provoking_index] & 0xffff : 0;
    if (color_enabled_ && std::none_of(colors_.begin(), colors_.end(), [&](const auto &color) {
          return color.write_mask && relative_layer <= color.last_layer - color.first_layer;
        }))
      continue;
    if ((depth_control_ & 3) && relative_layer > depth_last_layer_ - depth_first_layer_)
      continue;
    if (nonpositive_w == 3 || outside_near == 3 || outside_far == 3)
      continue;
    // Clip coverage in homogeneous coordinates and carry the original
    // barycentrics so added vertices do not replace shader-visible parameters.
    const bool clipped = clip_xy || outside_near || outside_far;
    uint32_t clip_origin = 0;
    if (clipped) {
      // The clipper orders the polygon in homogeneous X, before projection.
      // If two X coordinates tie, start at the opposite vertex.
      for (uint32_t k = 1; k < 3; ++k)
        if (clip_positions[k].x < clip_positions[clip_origin].x ||
            (clip_positions[k].x == clip_positions[clip_origin].x &&
             clip_positions[k].w < clip_positions[clip_origin].w))
          clip_origin = k;
      for (uint32_t k = 0; k < 3; ++k)
        if (clip_positions[(k + 1) % 3].x == clip_positions[(k + 2) % 3].x &&
            clip_positions[k].x != clip_positions[(k + 1) % 3].x)
          clip_origin = k;
    }
    std::vector<Point> coverage(v.begin(), v.end());
    if (clipped) {
      if (primitive_type_ == kRectangleList)
        throw std::runtime_error("graphics clipped rectangles are not implemented");
      coverage.assign(clip_positions.begin(), clip_positions.end());
      std::rotate(coverage.begin(), coverage.begin() + clip_origin, coverage.end());
      const auto clip = [&](const auto &distance) {
        std::vector<Point> output;
        if (coverage.empty())
          return;
        Point previous = coverage.back();
        float previous_distance = raster::truncate_float(distance(previous));
        for (const Point &current : coverage) {
          const float current_distance = raster::truncate_float(distance(current));
          if ((previous_distance < 0 && current_distance > 0) ||
              (previous_distance > 0 && current_distance < 0)) {
            const auto weights = raster::clip_weights(previous_distance, current_distance);
            const auto component = [&](double a, double b) {
              return raster::clip_component(a, b, weights);
            };
            const float w = component(previous.w, current.w);
            // Screen-space noperspective clipping is only qualified with
            // DX_LINEAR_ATTR_CLIP_ENA, which RADV always sets.
            const auto linear_weights =
                (clip_control & (1u << 24))
                    ? raster::linear_clip_weights(weights, previous.w, current.w, w)
                    : weights;
            const auto linear = [&](double a, double b) {
              return raster::clip_component(a, b, linear_weights);
            };
            output.push_back({component(previous.x, current.x),
                              component(previous.y, current.y),
                              w,
                              component(previous.z, current.z),
                              {component(previous.barycentric[0], current.barycentric[0]),
                               component(previous.barycentric[1], current.barycentric[1]),
                               component(previous.barycentric[2], current.barycentric[2])},
                              {linear(previous.linear[0], current.linear[0]),
                               linear(previous.linear[1], current.linear[1]),
                               linear(previous.linear[2], current.linear[2])}});
          }
          if (current_distance >= 0)
            output.push_back(current);
          previous = current;
          previous_distance = current_distance;
        }
        coverage = std::move(output);
      };
      if (outside_near)
        clip([&](Point point) { return point.z + ((clip_control & (1u << 19)) ? 0 : point.w); });
      if (outside_far)
        clip([](Point point) { return raster::vertex_difference(point.w, point.z); });
      if (clip_xy) {
        if (!valid_guard)
          throw std::runtime_error("unsupported graphics guard band");
        clip([&](Point point) { return guard_distance(point.x, point.w, gx, 1); });
        clip([&](Point point) { return guard_distance(point.x, point.w, gx, -1); });
        clip([&](Point point) { return guard_distance(point.y, point.w, gy, 1); });
        clip([&](Point point) { return guard_distance(point.y, point.w, gy, -1); });
      }
      if (coverage.size() < 3)
        continue;
      // A clipped vertex at the homogeneous origin has no projected area.
      std::erase_if(coverage, [](Point point) { return point.w == 0; });
      if (coverage.size() < 3)
        continue;
      for (auto &point : coverage) {
        point.x = raster::viewport_coordinate(point.x, point.w, sx, ox);
        point.y = raster::viewport_coordinate(point.y, point.w, sy, oy);
        point.z = raster::viewport_transform(point.z, point.w, sz, oz);
      }
      const auto same_position = [](Point a, Point b) { return a.x == b.x && a.y == b.y; };
      coverage.erase(std::unique(coverage.begin(), coverage.end(), same_position), coverage.end());
      if (coverage.size() > 1 && same_position(coverage.front(), coverage.back()))
        coverage.pop_back();
      if (coverage.size() < 3)
        continue;
    }
    for (const auto &point : coverage)
      if (!std::isfinite(point.x) || !std::isfinite(point.y))
        throw std::runtime_error("graphics clipped vertex has nonfinite raster coordinates");
    const auto edge = [](Point a, Point b, double x, double y) {
      return (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
    };
    double area = 0;
    for (uint32_t k = 1; k + 1 < coverage.size(); ++k)
      area += edge(coverage[0], coverage[k], coverage[k + 1].x, coverage[k + 1].y);
    if (area == 0)
      continue;
    const bool front = (area < 0) != bool(context_[0x207] & 4);
    if ((front && (context_[0x207] & 1)) || (!front && (context_[0x207] & 2)))
      continue;
    const auto polygon = coverage;
    const auto original_screen = v;
    const auto original_indices = indices;
    const uint32_t triangle_count = clipped ? polygon.size() - 2 : 1;
    for (uint32_t triangle = 0; triangle < triangle_count; ++triangle) {
      indices = original_indices;
      v = original_screen;
      screen = original_screen;
      if (clipped) {
        coverage = {polygon[0], polygon[triangle + 1], polygon[triangle + 2]};
        std::copy(coverage.begin(), coverage.end(), screen.begin());
        area = edge(screen[0], screen[1], screen[2].x, screen[2].y);
        if (area == 0)
          continue;
      }
      // Choose the largest setup reciprocal W, then the leftmost vertex. Nearly
      // equal W values can share a reciprocal. Equal-W axis-aligned right
      // triangles use the corner joining the two axis-aligned edges.
      uint32_t origin = 0;
      const auto inverse_w = [&](uint32_t k) { return raster::reciprocal(screen[k].w); };
      for (uint32_t k = 1; k < 3; ++k)
        if (inverse_w(k) > inverse_w(origin) ||
            (inverse_w(k) == inverse_w(origin) &&
             (screen[k].x < screen[origin].x ||
              (screen[k].x == screen[origin].x && screen[k].y < screen[origin].y))))
          origin = k;
      for (uint32_t k = 0; k < 3; ++k) {
        const auto &a = screen[k], &b = screen[(k + 1) % 3], &c = screen[(k + 2) % 3];
        if (a.w == b.w && a.w == c.w && ((a.x == b.x && a.y == c.y) || (a.y == b.y && a.x == c.x)))
          origin = k;
      }
      const uint32_t parameter_origin = clipped ? (clip_origin + origin) % 3 : origin;
      std::rotate(indices.begin(), indices.begin() + parameter_origin, indices.end());
      std::rotate(v.begin(), v.begin() + parameter_origin, v.end());
      std::rotate(screen.begin(), screen.begin() + origin, screen.end());
      const double interpolation_area = edge(screen[0], screen[1], screen[2].x, screen[2].y);
      const double depth_inverse_area = raster::area_reciprocal(interpolation_area);
      const float inverse_area = raster::truncate_float(depth_inverse_area);
      const auto plane = [&](float a, float b, float c) {
        const float delta_b = raster::vertex_difference(b, a);
        const float delta_c = raster::vertex_difference(c, a);
        return raster::Plane{
            raster::plane_gradient(screen[2].y - screen[0].y, screen[0].y - screen[1].y, delta_b,
                                   delta_c, inverse_area),
            raster::plane_gradient(screen[0].x - screen[2].x, screen[1].x - screen[0].x, delta_b,
                                   delta_c, inverse_area),
            a};
      };
      const auto original_weight = [&](uint32_t vertex, uint32_t component) {
        return clipped ? screen[vertex].barycentric[(parameter_origin + component) % 3]
                       : double(vertex == component);
      };
      const auto perspective_plane = [&](uint32_t component) {
        return plane(
            raster::truncate_float(original_weight(0, component) * raster::reciprocal(screen[0].w)),
            raster::truncate_float(original_weight(1, component) * raster::reciprocal(screen[1].w)),
            raster::truncate_float(original_weight(2, component) *
                                   raster::reciprocal(screen[2].w)));
      };
      const raster::Plane plane_iw = perspective_plane(1);
      const raster::Plane plane_jw = perspective_plane(2);
      const raster::Plane plane_rw =
          plane(raster::reciprocal(screen[0].w), raster::reciprocal(screen[1].w),
                raster::reciprocal(screen[2].w));
      const auto linear_plane = [&](uint32_t component) {
        const uint32_t original = (parameter_origin + component) % 3;
        if (!(clip_control & (1u << 24)))
          return plane(original_weight(0, component) * v[component].w / screen[0].w,
                       original_weight(1, component) * v[component].w / screen[1].w,
                       original_weight(2, component) * v[component].w / screen[2].w);
        return plane(screen[0].linear[original], screen[1].linear[original],
                     screen[2].linear[original]);
      };
      const raster::Plane plane_i = clipped ? linear_plane(1) : plane(0, 1, 0);
      const raster::Plane plane_j = clipped ? linear_plane(2) : plane(0, 0, 1);
      const float depth_b = raster::vertex_difference(screen[1].z, screen[0].z);
      const float depth_c = raster::vertex_difference(screen[2].z, screen[0].z);
      raster::DepthPlane plane_z{
          raster::depth_gradient(raster::plane_numerator(screen[2].y - screen[0].y,
                                                         screen[0].y - screen[1].y, depth_b,
                                                         depth_c),
                                 depth_inverse_area),
          raster::depth_gradient(raster::plane_numerator(screen[0].x - screen[2].x,
                                                         screen[1].x - screen[0].x, depth_b,
                                                         depth_c),
                                 depth_inverse_area),
          screen[0].z, screen[0].x, screen[0].y};
      if (context_[0x207] & (1u << (front ? 11 : 12))) {
        const uint32_t scale_reg = front ? 0x2e0 : 0x2e2;
        const float scale = std::bit_cast<float>(context_[scale_reg]) / 16.0f;
        const float offset = std::bit_cast<float>(context_[scale_reg + 1]);
        const float clamp = std::bit_cast<float>(context_[0x2df]);
        const uint32_t format = context_[0x2de];
        if ((format & ~0x1ffu) || !std::isfinite(scale) || !std::isfinite(offset) ||
            !std::isfinite(clamp))
          throw std::runtime_error("unsupported graphics depth bias state");
        const float largest = static_cast<float>(
            std::max({std::abs(screen[0].z), std::abs(screen[1].z), std::abs(screen[2].z)}));
        plane_z.base +=
            raster::depth_bias(plane_z.dx, plane_z.dy, scale,
                               raster::depth_bias_constant(offset, format, largest), clamp);
      }
      const bool rectangle = primitive_type_ == kRectangleList;
      const auto [xmin, xmax] = std::minmax_element(coverage.begin(), coverage.end(),
                                                    [](Point a, Point b) { return a.x < b.x; });
      const auto [ymin, ymax] = std::minmax_element(coverage.begin(), coverage.end(),
                                                    [](Point a, Point b) { return a.y < b.y; });
      // Bound in floating point before narrowing, including triangles that
      // extend beyond the integer raster coordinate range.
      const int min_x = int(std::clamp(std::floor(xmin->x), double(left), double(right)));
      const int min_y = int(std::clamp(std::floor(ymin->y), double(top), double(bottom)));
      const int max_x = int(std::clamp(std::ceil(xmax->x), double(left), double(right)));
      const int max_y = int(std::clamp(std::ceil(ymax->y), double(top), double(bottom)));
      std::vector<uint32_t> parameters(attributes * 12);
      std::array<uint32_t, 32 * 12> gathered_words;
      const bool gathered =
          allow_ram_read_batching && attributes &&
          try_gather_attributes(memory, group, indices, provoking_index,
                                std::span{gathered_words}.first(parameters.size()));
      for (uint32_t a = 0; a < attributes; ++a) {
        const uint32_t control = context_[0x199 + a];
        // FLAT_SHADE with OFFSET bit 5 exposes the three raw vertex values.
        const bool passthrough = (control & 0x420) == 0x420;
        // Flat inputs retain the provoking vertex's raw bits with zero coefficient deltas.
        const bool flat = (control & 0x400) && !passthrough;
        if (control & ~0x100073fu)
          throw std::runtime_error("unsupported graphics parameter interpolation");
        for (uint32_t c = 0; c < 4; ++c) {
          std::array<float, 3> values{};
          for (uint32_t k = 0; k < 3; ++k) {
            if (gathered) {
              values[k] = std::bit_cast<float>(gathered_words[a * 12 + c * 3 + k]);
            } else if ((control & 32) && !passthrough) {
              values[k] = ((control >> 8) & (c == 3 ? 1u : 2u)) ? 1.0f : 0.0f;
            } else {
              // The hardware attribute ring uses 16-byte elements interleaved
              // across 32 vertices. Its base comes from SPI_ATTRIBUTE_RING_BASE,
              // independently of the driver's shader descriptor table.
              const auto address = addr_calc::rdna_buffer_address(
                  3u << 30, 2u << 21, ring_stride, true, flat ? provoking_index : indices[k],
                  (control & 31) * 16 + c * 4, 0, 0);
              uint32_t bits = 0;
              if (memory.read(attribute_ring_base_ + group.attribute_offset + address.offset,
                              {reinterpret_cast<std::byte *>(&bits), sizeof(bits)}) !=
                  VmAccessOutcome::Complete)
                throw std::runtime_error("graphics attribute read failed");
              values[k] = std::bit_cast<float>(bits);
            }
          }
          parameters[a * 12 + c * 3] = std::bit_cast<uint32_t>(values[0]);
          parameters[a * 12 + c * 3 + 1] =
              flat ? 0
                   : std::bit_cast<uint32_t>(
                         passthrough ? values[1]
                                     : raster::attribute_difference(values[1], values[0]));
          parameters[a * 12 + c * 3 + 2] =
              flat ? 0
                   : std::bit_cast<uint32_t>(
                         passthrough ? values[2]
                                     : raster::attribute_difference(values[2], values[0]));
        }
      }
      const auto coverage_mask = [&](int x, int y) {
        const auto inside_edges = [&](std::span<const Point> points, double signed_area, double px,
                                      double py) {
          if (signed_area == 0)
            return false;
          bool inside = true;
          for (uint32_t k = 0; k < points.size(); ++k) {
            Point a = points[k], b = points[(k + 1) % points.size()];
            if (signed_area < 0)
              std::swap(a, b);
            const double e = edge(a, b, px, py);
            const bool top_left = b.y < a.y || (b.y == a.y && b.x > a.x);
            inside &= e > 0 || (e == 0 && top_left);
          }
          return inside;
        };
        uint8_t mask = 0;
        for (int q = 0; q < 4; ++q) {
          const int fx = x + (q & 1), fy = y + (q >> 1);
          const double px = fx + 0.5, py = fy + 0.5;
          bool inside = true;
          if (rectangle) {
            inside = px >= std::min({v[0].x, v[1].x, v[2].x}) &&
                     px < std::max({v[0].x, v[1].x, v[2].x}) &&
                     py >= std::min({v[0].y, v[1].y, v[2].y}) &&
                     py < std::max({v[0].y, v[1].y, v[2].y});
          } else {
            inside = inside_edges(coverage, area, px, py);
          }
          const bool sample_enabled = (context_[0x30e + (fy & 1)] >> (16 * (fx & 1))) & 1;
          if (inside && sample_enabled && fx >= left && fx < right && fy >= top && fy < bottom)
            mask |= 1u << q;
        }
        return mask;
      };
      const bool use_sse41_planes = raster::supports_sse41_planes();
      const bool omit_unused_linear =
          allow_ram_read_batching && !(context_[0x198] & 0x70) &&
          raster::can_omit_bounded_planes(plane_i, plane_j, screen[0].x, screen[0].y);
      const auto interpolate_quad_impl = [&]<bool Sse41>(Fragment *quad, int x, int y,
                                                         uint8_t mask) {
        for (int q = 0; q < 4; ++q) {
          auto &f = quad[q];
          f.x = x + (q & 1);
          f.y = y + (q >> 1);
          f.covered = mask & (1u << q);
          f.stencil_reference = 0;
          f.stencil_exported = false;
          const double dx = x + 0.5 - screen[0].x, dy = y + 0.5 - screen[0].y;
          const auto evaluate = [&](const raster::Plane &plane) {
#if defined(__clang__) && defined(__x86_64__)
            if constexpr (Sse41)
              return plane.at_quad_sse41(dx, dy, q);
#endif
            return plane.at_quad(dx, dy, q);
          };
          if (omit_unused_linear) {
            f.linear_i = 0;
            f.linear_j = 0;
          } else {
            const double b1 = evaluate(plane_i);
            const double b2 = evaluate(plane_j);
            f.linear_i = b1;
            f.linear_j = b2;
          }
          f.pull_model = {evaluate(plane_iw), evaluate(plane_jw), evaluate(plane_rw)};
          const float w = raster::reciprocal(f.pull_model[2]);
          f.i = raster::multiply_perspective(f.pull_model[0], w);
          f.j = raster::multiply_perspective(f.pull_model[1], w);
          f.z = plane_z.at(f.x, f.y);
        }
      };
      const auto interpolate_quad = [&](Fragment *quad, int x, int y, uint8_t mask) {
        if (use_sse41_planes)
          return interpolate_quad_impl.template operator()<true>(quad, x, y, mask);
        interpolate_quad_impl.template operator()<false>(quad, x, y, mask);
      };
      const auto parallel_raster = [&]() {
#if defined(__GLIBC__) && defined(__x86_64__)
        const uint32_t columns = (max_x - (min_x & ~1) + 1) / 2;
        const uint32_t rows = (max_y - (min_y & ~1) + 1) / 2;
        const size_t quad_count = size_t{columns} * rows;
        if (!pool || threads < 2 || quad_count < 1024 || rows < 2)
          return false;
        std::fenv_t environment;
        if (std::fegetenv(&environment) != 0 || (environment.__control_word & 0x3f) != 0x3f ||
            (environment.__mxcsr & 0x1f80) != 0x1f80)
          return false;
        const int saved_errno = errno;
        const uint32_t stripe_count =
            std::min<uint32_t>({rows, std::min(threads, 32u) * 2, uint32_t(quad_count / 256)});
        std::vector<uint8_t> masks(quad_count);
        std::vector<size_t> counts(stripe_count), offsets(stripe_count + 1);
        struct TaskState {
          uint16_t x87_flags = 0;
          uint32_t sse_flags = 0;
          size_t error_index = 0;
          int error = 0;
        };
        std::vector<TaskState> results(stripe_count);
        const auto in_environment = [&](size_t stripe, auto &&work) {
          std::fenv_t previous;
          std::fegetenv(&previous);
          const int previous_errno = errno;
          struct Restore {
            std::fenv_t &environment;
            int error;
            ~Restore() {
              std::fesetenv(&environment);
              errno = error;
            }
          } restore{previous, previous_errno};
          std::fesetenv(&environment);
          work();
          std::fenv_t after;
          std::fegetenv(&after);
          results[stripe].x87_flags |= after.__status_word & 0x3f;
          results[stripe].sse_flags |= after.__mxcsr & 0x3f;
        };
        // Coverage is CPU-only. A dense byte per quad avoids copying Fragment
        // objects or padding waves at worker boundaries. Counts determine the
        // exact original compacted quad order, including partially covered quads.
        pool->run_indexed(stripe_count, threads, [&](size_t stripe) {
          in_environment(stripe, [&] {
            size_t count = 0;
            for (uint32_t row = rows * stripe / stripe_count;
                 row < rows * (stripe + 1) / stripe_count; ++row) {
              const int y = (min_y & ~1) + 2 * row;
              for (uint32_t column = 0; column < columns; ++column) {
                const uint8_t mask = coverage_mask((min_x & ~1) + 2 * column, y);
                masks[size_t{row} * columns + column] = mask;
                count += mask != 0;
              }
            }
            counts[stripe] = count;
          });
        });
        for (uint32_t stripe = 0; stripe < stripe_count; ++stripe)
          offsets[stripe + 1] = offsets[stripe] + counts[stripe];
        const uint32_t quads_per_wave = fragment_wave_size_ / 4;
        static_assert(std::is_trivially_default_constructible_v<Fragment>);
        static_assert(std::is_nothrow_move_constructible_v<FragmentWave>);
        const size_t first_wave = fragments_.size();
        const size_t wave_count = (offsets.back() + quads_per_wave - 1) / quads_per_wave;
        const size_t available = fragments_.max_size() - first_wave;
        if (wave_count > available)
          throw std::length_error("graphics fragment count exceeds capacity");
        // Grow before constructing any unfinished lanes. Keep geometric growth
        // when several primitives append to the same fragment window.
        if (first_wave + wave_count > fragments_.capacity())
          fragments_.reserve(first_wave + std::min(available, std::max(first_wave, wave_count)));
        struct InitializeOnFailure {
          std::vector<FragmentWave> &waves;
          size_t first;
          bool complete = false;
          ~InitializeOnFailure() {
            if (!complete)
              for (size_t wave = first; wave < waves.size(); ++wave)
                waves[wave].lanes.fill(Fragment{});
          }
        } initialize_on_failure{fragments_, first_wave};
        for (size_t wave = 0; wave < wave_count; ++wave) {
          const uint32_t live_lanes =
              4 * std::min<size_t>(quads_per_wave, offsets.back() - wave * quads_per_wave);
          fragments_.emplace_back(FragmentWave::InterpolatedLanes{}, live_lanes);
        }
        for (size_t wave = first_wave; wave < fragments_.size(); ++wave) {
          auto &batch = fragments_[wave];
          batch.relative_layer = relative_layer;
          batch.front = front;
          batch.parameters = parameters;
        }
        pool->run_indexed(stripe_count, threads, [&](size_t stripe) {
          in_environment(stripe, [&] {
            size_t output = offsets[stripe];
            auto &result = results[stripe];
            std::array<Fragment, 4> discarded_quad;
            for (uint32_t row = rows * stripe / stripe_count;
                 row < rows * (stripe + 1) / stripe_count; ++row) {
              const int y = (min_y & ~1) + 2 * row;
              for (uint32_t column = 0; column < columns; ++column) {
                const size_t index = size_t{row} * columns + column;
                Fragment *quad = discarded_quad.data();
                if (masks[index])
                  quad = fragments_[first_wave + output / quads_per_wave].lanes.data() +
                         (output % quads_per_wave) * 4;
                errno = 0;
                interpolate_quad(quad, (min_x & ~1) + 2 * column, y, masks[index]);
                // Keep uncovered interpolation live for FP exceptions and errno.
                if (!masks[index])
                  asm volatile("" : : "m"(discarded_quad));
                if (errno) {
                  result.error = errno;
                  result.error_index = index;
                }
                output += masks[index] != 0;
              }
            }
          });
        });
        initialize_on_failure.complete = true;
        // Join before the next primitive's attribute accesses or metadata work.
        // This preserves callback-visible caller flags and all worker FP state.
        size_t error_index = 0;
        int error = saved_errno;
        bool has_error = false;
        for (const auto &result : results) {
          environment.__status_word |= result.x87_flags;
          environment.__mxcsr |= result.sse_flags;
          if (result.error && (!has_error || result.error_index > error_index)) {
            error_index = result.error_index;
            error = result.error;
            has_error = true;
          }
        }
        std::fesetenv(&environment);
        errno = error;
        return true;
#else
        (void)pool;
        (void)threads;
        return false;
#endif
      };
      if (parallel_raster())
        continue;
      FragmentWave batch;
      batch.relative_layer = relative_layer;
      batch.front = front;
      batch.parameters = parameters;
      uint32_t used = 0;
      for (int y = min_y & ~1; y < max_y; y += 2) {
        for (int x = min_x & ~1; x < max_x; x += 2) {
          const uint8_t mask = coverage_mask(x, y);
          if (!mask) {
            std::array<Fragment, 4> discarded_quad;
            interpolate_quad(discarded_quad.data(), x, y, mask);
            // Keep uncovered interpolation live for FP exceptions and errno.
            asm volatile("" : : "m"(discarded_quad));
            continue;
          }
          interpolate_quad(batch.lanes.data() + used, x, y, mask);
          used += 4;
          if (used == fragment_wave_size_) {
            fragments_.push_back(std::move(batch));
            batch = FragmentWave{};
            batch.relative_layer = relative_layer;
            batch.front = front;
            batch.parameters = parameters;
            used = 0;
          }
        }
      }
      if (used)
        fragments_.push_back(std::move(batch));
    }
  }
  const bool clamp_depth = !(context_[0x19] & 1);
  const float depth_min = std::bit_cast<float>(context_[0x115]);
  const float depth_max = std::bit_cast<float>(context_[0x116]);
  if ((depth_control_ & 2) && clamp_depth &&
      (!std::isfinite(depth_min) || !std::isfinite(depth_max) || depth_min > depth_max))
    throw std::runtime_error("unsupported graphics viewport depth range");
  if (!attachments_prepared_) {
    for (const auto &color : colors_) {
      if (!color.write_mask || !color.metadata)
        continue;
      const uint32_t block_bits = 8 - std::countr_zero(color.bytes);
      const uint32_t block_width = 1u << ((block_bits + 1) / 2);
      const uint32_t block_height = 1u << (block_bits / 2);
      if (color.max_mip) {
        const auto mip = gfx11_dcc_mip_layout(color.swizzle, color.bytes, color.resource_width,
                                              color.resource_height, color.max_mip + 1, color.mip,
                                              color.pipe_aligned);
        if (!mip)
          throw std::runtime_error("unsupported graphics DCC mip layout");
        if (mip->enabled)
          for (uint32_t layer = color.first_layer; layer <= color.last_layer; ++layer)
            for (uint32_t y = 0; y < color.height; y += block_height)
              for (uint32_t x = 0; x < color.width; x += block_width)
                if (const char *error = image_metadata_detail::materialize_gfx11_dcc_mip(
                        memory, color.base - mip->pixels.offset, *color.metadata, x, y, color.bytes,
                        color.swizzle, color.pipe_aligned, layer, *mip))
                  throw std::runtime_error(error);
        continue;
      }
      for (uint32_t layer = color.first_layer; layer <= color.last_layer; ++layer) {
        if (allow_ram_read_batching &&
            try_gfx11_expanded_dcc(memory, *color.metadata, color.width, color.height, color.bytes,
                                   color.swizzle, color.pipe_aligned, layer))
          continue;
        if (allow_ram_read_batching &&
            try_materialize_gfx11_dcc_layer(memory, color.base, *color.metadata, color.width,
                                            color.height, color.bytes, color.swizzle,
                                            color.pipe_aligned, layer, color.slice_size))
          continue;
        for (uint32_t y = 0; y < color.height; y += block_height)
          for (uint32_t x = 0; x < color.width; x += block_width)
            materialize_gfx11_dcc(memory, color.base, *color.metadata, x, y, color.width,
                                  color.height, color.bytes, color.swizzle, color.pipe_aligned,
                                  layer, color.slice_size);
      }
    }
    if ((depth_control_ & 2) && depth_metadata_) {
      uint32_t bits = depth_clear_;
      if (depth_bytes_ == 2) {
        const std::array<uint32_t, 1> component{bits};
        if (pack_buffer_format(7, 4, component, {reinterpret_cast<uint8_t *>(&bits), 2}).failed())
          throw std::runtime_error("unsupported graphics depth format");
      }
      const bool expanded = allow_ram_read_batching && depth_bytes_ == 4 && !(depth_control_ & 1) &&
                            try_gfx11_expanded_htile(memory, *depth_metadata_, depth_width_,
                                                     depth_height_, depth_swizzle_);
      const bool materialized =
          !expanded && allow_ram_read_batching && depth_bytes_ == 4 && !(depth_control_ & 1) &&
          try_materialize_gfx11_htile_layer(memory, depth_base_, *depth_metadata_, depth_width_,
                                            depth_height_, depth_swizzle_, bits,
                                            depth_metadata_has_stencil_);
      if (!expanded && !materialized)
        for (uint32_t y = 0; y < depth_height_; y += 8)
          for (uint32_t x = 0; x < depth_width_; x += 8)
            materialize_gfx11_htile(memory, depth_base_, *depth_metadata_, x, y, depth_width_,
                                    depth_height_, depth_bytes_, depth_swizzle_, bits,
                                    depth_metadata_has_stencil_);
    }
    if ((depth_control_ & 1) && depth_metadata_ && depth_metadata_has_stencil_)
      for (uint32_t y = 0; y < depth_height_; y += 8)
        for (uint32_t x = 0; x < depth_width_; x += 8)
          materialize_gfx11_stencil_htile(memory, stencil_base_, *depth_metadata_, x, y,
                                          depth_width_, depth_height_, depth_bytes_, depth_swizzle_,
                                          stencil_swizzle_, stencil_clear_);
    attachments_prepared_ = true;
  }
}

template <typename Memory>
void GraphicsDraw::write_output_fragment(const Memory &memory, const FragmentWave &batch,
                                         uint32_t lane) {
  const auto image_address =
      arch_ == ROCJITSU_CODE_ARCH_RDNA4 ? gfx12_image_address : gfx11_image_address;
  const auto decode_export = [](uint32_t format, const ColorExport &exported) {
    auto components = exported.values;
    uint32_t component_mask = exported.mask;
    if (format == kExportFp16Abgr || format == kExportUnorm16Abgr || format == kExportSnorm16Abgr ||
        format == kExportUint16Abgr || format == kExportSint16Abgr) {
      if (exported.mask & ~3u)
        throw std::runtime_error("unsupported packed graphics color export mask");
      component_mask = ((exported.mask & 1) ? 3u : 0u) | ((exported.mask & 2) ? 12u : 0u);
      for (uint32_t c = 0; c < 4; ++c) {
        const uint16_t half = exported.values[c / 2] >> (16 * (c % 2));
        if (format == kExportUint16Abgr)
          components[c] = half;
        else if (format == kExportUnorm16Abgr)
          components[c] = std::bit_cast<uint32_t>(half / 65535.0f);
        else if (format == kExportSnorm16Abgr)
          components[c] =
              std::bit_cast<uint32_t>(std::max(static_cast<int16_t>(half) / 32767.0f, -1.0f));
        else if (format == kExportSint16Abgr)
          components[c] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(half)));
        else
          components[c] = std::bit_cast<uint32_t>(util::f16_to_f32(half));
      }
    } else {
      const uint32_t export_mask = format == kExport32R                               ? 1u
                                   : (format == kExport32Gr || format == kExport32Ar) ? 3u
                                                                                      : 15u;
      if (exported.mask & ~export_mask)
        throw std::runtime_error("unsupported graphics color export mask");
      if (format == kExport32Ar) {
        // GFX10+ carries alpha in the second export operand.
        components[3] = components[1];
        components[1] = 0;
        component_mask = (component_mask & 1) | ((component_mask & 2) << 2);
      }
    }
    return ColorExport{.mask = component_mask, .values = components};
  };
  const bool clamp_depth = !(context_[0x19] & 1);
  const float depth_min = std::bit_cast<float>(context_[0x115]);
  const float depth_max = std::bit_cast<float>(context_[0x116]);
  const auto &f = batch.lanes[lane];
  if (!f.covered)
    return;
  const bool back = !batch.front && (depth_control_ & (1u << 7));
  const uint32_t face_shift = back ? 8 : 0;
  const uint8_t reference = f.stencil_exported && (context_[0x1b] & 2)
                                ? f.stencil_reference
                                : context_[0x22] >> face_shift;
  uint8_t previous_stencil = 0;
  std::optional<uint64_t> stencil_address;
  bool stencil_pass = true, depth_pass = true;
  if (depth_control_ & 1) {
    const uint64_t base =
        image_layer_base(arch_ == ROCJITSU_CODE_ARCH_RDNA4, stencil_base_, stencil_slice_size_,
                         depth_first_layer_ + batch.relative_layer, 1, stencil_swizzle_);
    stencil_address = image_address(base, f.x + stencil_tail_x_, f.y + stencil_tail_y_,
                                    stencil_pitch_, 1, stencil_swizzle_);
    if (!stencil_address ||
        memory.read(*stencil_address, std::as_writable_bytes(std::span{&previous_stencil, 1})) !=
            VmAccessOutcome::Complete)
      throw std::runtime_error("graphics stencil read failed");
    const uint8_t mask = context_[0x24] >> face_shift;
    stencil_pass = depth_stencil_compare((depth_control_ >> (back ? 20 : 8)) & 7, reference & mask,
                                         previous_stencil & mask);
  }
  if (depth_control_ & 2) {
    const uint64_t layer_base =
        image_layer_base(arch_ == ROCJITSU_CODE_ARCH_RDNA4, depth_base_, depth_slice_size_,
                         depth_first_layer_ + batch.relative_layer, depth_bytes_, depth_swizzle_);
    const auto address = image_address(layer_base, f.x + depth_tail_x_, f.y + depth_tail_y_,
                                       depth_pitch_, depth_bytes_, depth_swizzle_);
    uint32_t previous_bits = 0;
    if (!address || memory.read(*address, {reinterpret_cast<std::byte *>(&previous_bits),
                                           depth_bytes_}) != VmAccessOutcome::Complete)
      throw std::runtime_error("graphics depth read failed");
    const float clamped_depth =
        clamp_depth ? raster::clamp_viewport_depth(f.z, depth_min, depth_max) : f.z;
    uint32_t next_bits = std::bit_cast<uint32_t>(clamped_depth);
    if (depth_bytes_ == 2) {
      const std::array<uint32_t, 1> component{next_bits};
      if (pack_buffer_format(7, 4, component, {reinterpret_cast<uint8_t *>(&next_bits), 2})
              .failed())
        throw std::runtime_error("unsupported graphics depth format");
      next_bits &= 0xffff;
    }
    const float previous =
        depth_bytes_ == 2 ? previous_bits / 65535.0f : std::bit_cast<float>(previous_bits);
    const float depth = depth_bytes_ == 2 ? next_bits / 65535.0f : clamped_depth;
    depth_pass = depth_stencil_compare((depth_control_ >> 4) & 7, depth, previous);
    if (stencil_pass && depth_pass && (depth_control_ & 4) &&
        memory.write(*address, {reinterpret_cast<const std::byte *>(&next_bits), depth_bytes_}) !=
            VmAccessOutcome::Complete)
      throw std::runtime_error("graphics depth write failed");
  }
  if (stencil_address) {
    const uint32_t operation = (context_[0x1d] >> ((back ? 12 : 0) + (!stencil_pass ? 0
                                                                      : depth_pass  ? 4
                                                                                    : 8))) &
                               15;
    const uint8_t mask = context_[0x25] >> face_shift;
    const uint8_t next =
        stencil_operation(operation, previous_stencil, reference, context_[0x23] >> face_shift);
    const uint8_t result = (previous_stencil & ~mask) | (next & mask);
    if (mask && operation &&
        memory.write(*stencil_address, std::as_bytes(std::span{&result, 1})) !=
            VmAccessOutcome::Complete)
      throw std::runtime_error("graphics stencil write failed");
  }
  if (!stencil_pass || !depth_pass)
    return;
  if (count_occlusion_)
    ++occlusion_samples_;
  for (uint32_t target = 0; target < colors_.size(); ++target) {
    const auto &color = colors_[target];
    if (!color.write_mask)
      continue;
    const auto &exported = fragment_export(batch, lane, color.export_index);
    if (!exported.mask || batch.relative_layer > color.last_layer - color.first_layer)
      continue;
    auto [component_mask, components] = decode_export(color.export_format, exported);
    const uint64_t layer_base =
        image_layer_base(arch_ == ROCJITSU_CODE_ARCH_RDNA4, color.base, color.slice_size,
                         color.first_layer + batch.relative_layer, color.bytes, color.swizzle);
    const auto address =
        color.volume ? image_volume_address(arch_ == ROCJITSU_CODE_ARCH_RDNA4,
                                            color.base - color.volume->plane.offset, *color.volume,
                                            f.x, f.y, color.first_layer + batch.relative_layer,
                                            color.bytes, color.swizzle)
                     : image_address(layer_base, f.x + color.tail_x, f.y + color.tail_y,
                                     color.pitch, color.bytes, color.swizzle);
    if (!address)
      throw std::runtime_error("graphics color address is unsupported");
    std::array<uint8_t, 16> previous{}, bytes{};
    const uint32_t blend = color.blend,
                   write_mask = color.write_mask & component_mask & color.channel_mask;
    // Hardware ignores ROP3 when blending is enabled, matching DISABLE_ROP3.
    const uint32_t rop =
        (blend & (kDisableRop3 | kBlendEnable)) ? kRopCopy : (context_[0x216] >> 16) & 15;
    const uint32_t component_bytes = color.bytes / color.components;
    // The blend gate accepts only UNORM and floating attachments.
    const bool fp16_blend = color.component_widths[0] == 16;
    const bool fp32_blend = color.component_widths[0] == 32;
    const bool unorm_blend = color.memory_format == kBufR8Unorm ||
                             color.memory_format == kBufRgba8Unorm ||
                             color.memory_format == kBufRgb10A2Unorm;
    const auto store_unorm = [&](uint32_t c, double value) {
      const uint32_t physical = color.component_indices[c];
      if (physical >= color.components)
        return;
      if (color.memory_format != kBufRgb10A2Unorm) {
        bytes[physical] = color.srgb && c < 3 ? srgb_color_byte(value) : unorm_color_bits(value);
      } else {
        uint32_t packed = 0;
        std::memcpy(&packed, bytes.data(), 4);
        packed |= unorm_color_bits(value, color.component_widths[physical]) << (physical * 10);
        std::memcpy(bytes.data(), &packed, 4);
      }
    };
    if (!write_mask)
      continue;
    const uint32_t full_mask = color.channel_mask;
    if ((rop != kRopCopy || (blend & kBlendEnable) || (write_mask & full_mask) != full_mask) &&
        memory.read(*address, std::as_writable_bytes(std::span{previous}.first(color.bytes))) !=
            VmAccessOutcome::Complete)
      throw std::runtime_error("graphics color read failed");
    if (blend & kBlendEnable) {
      std::array<float, 4> source{}, destination{}, constant{};
      const auto decoded = unpack_buffer_format(color.memory_format, color.selectors,
                                                std::span{previous}.first(color.bytes));
      if (decoded.failed())
        throw std::runtime_error("unsupported graphics blend format");
      for (uint32_t c = 0; c < 4; ++c) {
        source[c] = std::bit_cast<float>(components[c]);
        destination[c] = std::bit_cast<float>(decoded.value()[c]);
        constant[c] = std::bit_cast<float>(context_[0x105 + c]);
        if (unorm_blend) {
          source[c] = std::isnan(source[c]) ? 0 : std::clamp(source[c], 0.0f, 1.0f);
          // UNORM destinations round to twelve significant bits before blending.
          const uint32_t normalized = decoded.value()[c];
          destination[c] =
              std::bit_cast<float>((normalized + 0x7ffu + ((normalized >> 12) & 1)) & ~0xfffu);
          if (color.srgb && c < 3)
            destination[c] = srgb_color_linear(previous[color.component_indices[c]]);
          constant[c] = std::isnan(constant[c]) ? 0 : std::clamp(constant[c], 0.0f, 1.0f);
          // Color blending truncates UNORM constants to twelve significant bits.
          constant[c] = std::bit_cast<float>(std::bit_cast<uint32_t>(constant[c]) & ~0xfffu);
        }
      }
      const auto widen = [](const std::array<float, 4> &values) {
        std::array<double, 4> wide{};
        std::copy(values.begin(), values.end(), wide.begin());
        return wide;
      };
      // FP16 attachments also truncate blend constants to twelve significant bits.
      if (fp16_blend)
        for (auto &value : constant)
          value = std::bit_cast<float>(std::bit_cast<uint32_t>(value) & ~0xfffu);
      // SX classifies the raw export before arithmetic saturation. NaNs do
      // not set zero/one flags, while negative values set the zero flag.
      auto flag_source = source;
      const uint32_t epsilon = (context_[0x1d6] >> (4 * target)) & 15;
      const float threshold =
          epsilon ? std::ldexp(epsilon & 1 ? 0.75f : 0.5f, int(epsilon / 2) - 11) : 0;
      const auto source_flag = [&](float value) {
        return value < threshold || value == 0       ? 0.0f
               : value > 1 - threshold || value == 1 ? 1.0f
                                                     : value;
      };
      if (unorm_blend)
        for (uint32_t c = 0; c < 4; ++c)
          flag_source[c] = source_flag(std::bit_cast<float>(components[c]));
      const uint32_t opt_disable = context_[0x1d7] >> (4 * target);
      const uint32_t opt = context_[0x1d8 + target];
      const auto preserves_group = [&](bool alpha) {
        return !(write_mask & (alpha ? 8u : 7u)) ||
               (!(opt_disable & (alpha ? 2u : 1u)) &&
                blend_can_copy(BlendCopy::Destination, opt >> (alpha ? 16 : 0), alpha ? 3 : 0,
                               component_mask, flag_source));
      };
      // A destination bypass retains the whole pixel. A written group that
      // needs arithmetic also prevents the other group from bypassing.
      const bool preserve_destination = preserves_group(false) && preserves_group(true);
      if (preserve_destination)
        continue;
      bool copy_source = false;
      if (fp32_blend) {
        auto copy = fixed_blend_copy(blend, write_mask, constant);
        // GFX11 exposes overrides for automatic source/destination bypass.
        const uint32_t info = context_[0x3b0 + target];
        if (arch_ != ROCJITSU_CODE_ARCH_RDNA4 &&
            ((copy == BlendCopy::Source && ((info >> 20) & 7)) ||
             (copy == BlendCopy::Destination && ((info >> 23) & 7))))
          copy = BlendCopy::None;
        // Test bypass before input flushing or widening, preserving the raw
        // destination bits even for subnormals and signaling NaNs.
        if (copy == BlendCopy::Destination)
          continue;
        copy_source = copy == BlendCopy::Source;
        if (!copy_source)
          for (uint32_t c = 0; c < 4; ++c) {
            source[c] = raster::blend_input(source[c]);
            destination[c] = raster::blend_input(destination[c]);
            constant[c] = raster::blend_input(constant[c]);
          }
      }
      if (unorm_blend) {
        const auto copy_group = [&](bool alpha, const std::array<float, 4> &flags, uint32_t mask,
                                    BlendCopy from = BlendCopy::Source) {
          return !(color.write_mask & mask & (alpha ? 8u : 7u)) ||
                 (!(opt_disable & (alpha ? 2u : 1u)) &&
                  blend_can_copy(from, opt >> (alpha ? 16 : 0), alpha ? 3 : 0, mask, flags));
        };
        // Destination-preserving pixels leave the quad first. The remaining
        // exports, including helper lanes, must all permit a source copy to
        // bypass arithmetic. Coverage only controls attachment writes.
        copy_source = true;
        for (uint32_t neighbor = lane & ~3u; neighbor < (lane & ~3u) + 4; ++neighbor) {
          const auto &other = fragment_export(batch, neighbor, color.export_index);
          if (!other.mask)
            continue;
          const auto decoded_neighbor = decode_export(color.export_format, other);
          const uint32_t mask = decoded_neighbor.mask;
          std::array<float, 4> flags;
          for (uint32_t c = 0; c < 4; ++c)
            flags[c] = source_flag(std::bit_cast<float>(decoded_neighbor.values[c]));
          if (copy_group(false, flags, mask, BlendCopy::Destination) &&
              copy_group(true, flags, mask, BlendCopy::Destination))
            continue;
          copy_source &= copy_group(false, flags, mask) && copy_group(true, flags, mask);
        }
        if (copy_source)
          for (uint32_t c = 0; c < 4; ++c)
            store_unorm(c, source[c]);
      }
      if (!copy_source) {
        const auto precision = unorm_blend  ? raster::BlendPrecision::Unorm
                               : fp16_blend ? raster::BlendPrecision::Float16
                               : fp32_blend ? raster::BlendPrecision::Float32
                                            : raster::BlendPrecision::Unaligned;
        for (uint32_t c = 0; c < 4; ++c) {
          const uint32_t control = c == 3 && (blend & kBlendSeparateAlpha) ? blend >> 16 : blend;
          const double blended = blend_component(control, c, widen(source), widen(destination),
                                                 widen(constant), precision);
          if (unorm_blend) {
            store_unorm(c, blended);
            continue;
          }
          float result = static_cast<float>(blended);
          // Floating attachments preserve signed and out-of-range values.
          // The FP16 blend result truncates on its way back to the attachment.
          if (fp16_blend)
            result = util::f16_to_f32(util::f32_to_f16_rtz(result));
          components[c] = std::bit_cast<uint32_t>(result);
        }
      }
    }
    if ((!(blend & kBlendEnable) || !unorm_blend) &&
        pack_buffer_format(color.memory_format, color.selectors, components,
                           std::span{bytes}.first(color.bytes))
            .failed())
      throw std::runtime_error("unsupported graphics color format");
    if (color.srgb && !(blend & kBlendEnable))
      for (uint32_t c = 0; c < 3; ++c)
        bytes[color.component_indices[c]] = srgb_color_byte(std::bit_cast<float>(components[c]));
    std::array<uint8_t, 16> output = previous;
    for (uint32_t c = 0; c < 4; ++c) {
      if (!(write_mask & (1u << c)))
        continue;
      const uint32_t physical = color.component_indices[c];
      if (color.component_widths[physical] % 8) {
        uint32_t source = 0, destination = 0, result = 0;
        std::memcpy(&source, bytes.data(), 4);
        std::memcpy(&destination, previous.data(), 4);
        std::memcpy(&result, output.data(), 4);
        uint32_t destination_shift = 0;
        for (uint32_t i = 0; i < physical; ++i)
          destination_shift += color.component_widths[i];
        const uint32_t mask = ((1u << color.component_widths[physical]) - 1) << destination_shift;
        const uint32_t value =
            ((rop & 1) ? ~source & ~destination : 0) | ((rop & 2) ? ~source & destination : 0) |
            ((rop & 4) ? source & ~destination : 0) | ((rop & 8) ? source & destination : 0);
        result = (result & ~mask) | (value & mask);
        std::memcpy(output.data(), &result, 4);
        continue;
      }
      for (uint32_t b = 0; b < component_bytes; ++b) {
        const uint32_t index = physical * component_bytes + b;
        const uint8_t source = bytes[index], destination = previous[index];
        output[index] =
            ((rop & 1) ? ~source & ~destination : 0) | ((rop & 2) ? ~source & destination : 0) |
            ((rop & 4) ? source & ~destination : 0) | ((rop & 8) ? source & destination : 0);
      }
    }
    if (memory.write(*address, std::as_bytes(std::span{output}.first(color.bytes))) !=
        VmAccessOutcome::Complete)
      throw std::runtime_error("graphics color write failed");
  }
}

void GraphicsDraw::write_outputs(const GpuVmAccess &memory, CpuDispatchPool *pool,
                                 uint32_t threads) {
  if (pool && threads > 1 && try_parallel_outputs(memory, *pool, threads))
    return;
  for (const auto &batch : fragments_)
    for (uint32_t lane = 0; lane < fragment_wave_size_; ++lane)
      write_output_fragment(memory, batch, lane);
}

bool GraphicsDraw::try_parallel_outputs(const GpuVmAccess &memory, CpuDispatchPool &pool,
                                        uint32_t threads) {
#if defined(__GLIBC__) && defined(__x86_64__)
  // One color and one D32 surface have independent tiles after a joint lease
  // proves they do not alias. Stencil, MRT and layered output remain serial.
  if (count_occlusion_ || (depth_control_ & 1) || fragments_.size() < 128)
    return false;
  const ColorAttachment *color = nullptr;
  for (const auto &candidate : colors_) {
    if (!candidate.write_mask)
      continue;
    if (color || candidate.first_layer != candidate.last_layer)
      return false;
    color = &candidate;
  }
  // D16 normalization can raise different host flags when the compiler packs
  // scalar divisions into vector lanes in the two memory specializations.
  const bool depth = depth_control_ & 2;
  if ((!color && !depth) ||
      (depth && (depth_bytes_ != 4 || depth_first_layer_ != depth_last_layer_)))
    return false;
  struct Surface {
    uint64_t base, slice_size;
    uint32_t first_layer, bytes, swizzle, pitch, tail_x, tail_y;
    uint64_t begin = UINT64_MAX, end = 0;
  };
  std::array<Surface, VmRamLease::kMaxRanges> surfaces{};
  size_t surface_count = 0;
  if (color)
    surfaces[surface_count++] = {color->base,   color->slice_size, color->first_layer,
                                 color->bytes,  color->swizzle,    color->pitch,
                                 color->tail_x, color->tail_y};
  if (depth)
    surfaces[surface_count++] = {depth_base_,   depth_slice_size_, depth_first_layer_,
                                 depth_bytes_,  depth_swizzle_,    depth_pitch_,
                                 depth_tail_x_, depth_tail_y_};
  std::fenv_t environment;
  if (std::fegetenv(&environment) != 0 || (environment.__control_word & 0x3f) != 0x3f ||
      (environment.__mxcsr & 0x1f80) != 0x1f80)
    return false;
  const int saved_errno = errno;
  const bool gfx12 = arch_ == ROCJITSU_CODE_ARCH_RDNA4;
  const auto image_address = gfx12 ? gfx12_image_address : gfx11_image_address;
  for (auto &surface : std::span{surfaces}.first(surface_count))
    surface.base = image_layer_base(gfx12, surface.base, surface.slice_size, surface.first_layer,
                                    surface.bytes, surface.swizzle);
  constexpr uint32_t tile_size = 32;
  const uint32_t columns = (width_ + tile_size - 1) / tile_size;
  const uint32_t rows = (height_ + tile_size - 1) / tile_size;
  std::vector<std::vector<size_t>> tiles(size_t{columns} * rows);
  const uint32_t format = color ? color->export_format : 0;
  const uint32_t export_mask = format >= kExportFp16Abgr && format <= kExportSint16Abgr ? 3u
                               : format == kExport32R                                   ? 1u
                               : format == kExport32Gr || format == kExport32Ar         ? 3u
                                                                                        : 15u;
  for (size_t batch_index = 0; batch_index < fragments_.size(); ++batch_index) {
    const auto &batch = fragments_[batch_index];
    if (batch.relative_layer)
      return false;
    for (uint32_t lane = 0; lane < fragment_wave_size_; ++lane) {
      const auto &fragment = batch.lanes[lane];
      if (!fragment.covered)
        continue;
      bool color_export = false;
      if (color) {
        const auto &exported = fragment_export(batch, lane, color->export_index);
        // Refuse before any effect; the serial path reports an invalid export at
        // its original position, after any earlier pixels have committed.
        if (exported.mask & ~export_mask)
          return false;
        color_export = exported.mask != 0;
      }
      // Depth precedes color handling, including lanes with no color export.
      if (!depth && !color_export)
        continue;
      if (fragment.x < 0 || fragment.y < 0 || uint32_t(fragment.x) >= width_ ||
          uint32_t(fragment.y) >= height_)
        return false;
      for (size_t i = 0; i < surface_count; ++i) {
        if (color && i == 0 && !color_export)
          continue;
        auto &surface = surfaces[i];
        const auto address =
            image_address(surface.base, fragment.x + surface.tail_x, fragment.y + surface.tail_y,
                          surface.pitch, surface.bytes, surface.swizzle);
        if (!address || surface.bytes > UINT64_MAX - *address)
          return false;
        surface.begin = std::min(surface.begin, *address);
        surface.end = std::max(surface.end, *address + surface.bytes);
      }
      tiles[(fragment.y / tile_size) * columns + fragment.x / tile_size].push_back(
          batch_index * fragment_wave_size_ + lane);
    }
  }
  std::array<VmRamRange, VmRamLease::kMaxRanges> ranges{};
  size_t range_count = 0;
  for (const auto &surface : std::span{surfaces}.first(surface_count))
    if (surface.begin < surface.end)
      ranges[range_count++] = {surface.begin, surface.end - surface.begin};
  if (!range_count)
    return false;
  std::erase_if(tiles, [](const auto &tile) { return tile.empty(); });
  if (tiles.size() < 2)
    return false;
  struct TaskState {
    std::fenv_t environment{};
    size_t error_index = 0;
    int error = 0;
  };
  std::vector<TaskState> results(tiles.size());
  auto lease = memory.try_lease_ram(std::span{ranges}.first(range_count));
  if (!lease) {
    errno = saved_errno;
    return false;
  }
  // The joint lease rejects noncontiguous or overlapping physical ranges.
  // Supported image equations are bijections within each padded surface, so
  // disjoint XY tiles cannot write overlapping bytes in either attachment.
  // Stable ordering inside a tile retains every same-pixel
  // blend, logic and depth dependency across primitives. No shader execution is reordered.
  class RamAccess {
  public:
    RamAccess(std::span<const VmRamRange> ranges, const VmRamLease &lease) : ranges_(ranges) {
      for (size_t i = 0; i < ranges.size(); ++i)
        bytes_[i] = lease.bytes(i);
    }
    VmAccessOutcome read(uint64_t address, std::span<std::byte> bytes) const {
      std::memcpy(bytes.data(), pointer(address, bytes.size()), bytes.size());
      return VmAccessOutcome::Complete;
    }
    VmAccessOutcome write(uint64_t address, std::span<const std::byte> bytes) const {
      std::memcpy(pointer(address, bytes.size()), bytes.data(), bytes.size());
      return VmAccessOutcome::Complete;
    }

  private:
    std::byte *pointer(uint64_t address, size_t size) const {
      for (size_t i = 0; i < ranges_.size(); ++i)
        if (address >= ranges_[i].address && address - ranges_[i].address <= bytes_[i].size() &&
            size <= bytes_[i].size() - (address - ranges_[i].address))
          return bytes_[i].data() + address - ranges_[i].address;
      assert(false && "output address was not prequalified");
      return nullptr;
    }
    std::span<const VmRamRange> ranges_;
    std::array<std::span<std::byte>, VmRamLease::kMaxRanges> bytes_{};
  };
  const RamAccess ram(std::span{ranges}.first(range_count), *lease);
  pool.run_indexed(tiles.size(), threads, [&](size_t tile) {
    std::fenv_t previous;
    std::fegetenv(&previous);
    const int previous_errno = errno;
    struct Restore {
      std::fenv_t &environment;
      int error;
      ~Restore() {
        std::fesetenv(&environment);
        errno = error;
      }
    } restore{previous, previous_errno};
    std::fesetenv(&environment);
    auto &result = results[tile];
    for (const size_t index : tiles[tile]) {
      errno = 0;
      write_output_fragment(ram, fragments_[index / fragment_wave_size_],
                            index % fragment_wave_size_);
      if (errno) {
        result.error = errno;
        result.error_index = index;
      }
    }
    std::fegetenv(&result.environment);
  });
  lease.reset();
  // Preserve all x87/SSE sticky flags, including the denormal-operand flag which
  // FE_ALL_EXCEPT omits. Enabled traps take the serial path above. Workers retain
  // their own modes/flags/errno; the caller receives the serial-order final errno.
  size_t error_index = 0;
  int error = saved_errno;
  bool has_error = false;
  for (const auto &result : results) {
    environment.__status_word |= result.environment.__status_word & 0x3f;
    environment.__mxcsr |= result.environment.__mxcsr & 0x3f;
    if (result.error && (!has_error || result.error_index > error_index)) {
      error_index = result.error_index;
      error = result.error;
      has_error = true;
    }
  }
  std::fesetenv(&environment);
  errno = error;
  return true;
#else
  (void)memory;
  (void)pool;
  (void)threads;
  return false;
#endif
}

std::optional<DispatchEntry> GraphicsDraw::advance(const GpuVmAccess &memory, CpuDispatchPool *pool,
                                                   uint32_t threads, bool allow_ram_read_batching,
                                                   bool allow_early_depth) {
  for (;;) {
    if (fragment_stage_) {
      write_outputs(memory, pool, threads);
      fragments_.clear();
      fragment_exports_.clear();
      fragment_exports_prepared_ = false;
      fragment_selection_active_ = false;
      fragment_dispatch_indices_.clear();
      if (next_raster_group_ == vertex_groups_.size())
        return next_vertex_group();
    }
    // Bound retained fragments to one original group's output plus this window.
    // All pending groups keep their own ring slots until their FS work retires.
    constexpr size_t kFragmentWindowWaves = 4096;
    while (next_raster_group_ < vertex_groups_.size()) {
      const auto &group = vertex_groups_[next_raster_group_++];
      finish_vertices(group);
      rasterize(memory, group, pool, threads, allow_ram_read_batching);
      if (fragments_.size() >= kFragmentWindowWaves)
        break;
    }
    fragment_stage_ = true;
    if (fragments_.empty())
      return next_vertex_group();
    if (allow_early_depth)
      select_fragment_waves(memory);
    if (!fragment_selection_active_ || !fragment_dispatch_indices_.empty()) {
      prepare_fragment_exports();
      return fragment_dispatch();
    }
    // No shader was invoked, but retain every original late attachment visit
    // and its ordinary failure/retirement path before advancing the draw.
  }
}

} // namespace rocjitsu::amdgpu
