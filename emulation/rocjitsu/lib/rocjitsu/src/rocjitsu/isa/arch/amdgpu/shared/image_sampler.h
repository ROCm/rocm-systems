// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/isa_traits.h"

#include <array>
#include <cstdint>

namespace rocjitsu::amdgpu {

enum class ImageClamp : uint32_t {
  Repeat = 0,
  Mirror = 1,
  Edge = 2,
  MirrorOnce = 3,
  HalfBorder = 4,
  MirrorHalfBorder = 5,
  Border = 6,
  MirrorBorder = 7,
};

enum class ImageXyFilter : uint32_t { Point = 0, Linear = 1 };
enum class ImageMipFilter : uint32_t { None = 0, Point = 1, Linear = 2 };

struct ImageSampler {
  std::array<ImageClamp, 3> clamp;
  uint32_t min_lod_fixed8, max_lod_fixed8; ///< Unsigned 1/256-LOD units.
  int32_t lod_bias_fixed8;                 ///< Signed 1/256-LOD units.
  ImageXyFilter mag_filter, min_filter;
  ImageMipFilter mip_filter;
  uint32_t z_filter;
  uint32_t perf_mip, perf_z;
  uint32_t border_color_type, filter_mode, max_aniso_ratio, aniso_threshold, aniso_bias;
  uint32_t compare_function;
  int32_t secondary_lod_bias; ///< Signed 1/16-LOD units.
  bool force_unnormalized, truncate_coordinates, force_degamma, mc_coord_trunc;
  bool skip_degamma, filter_precision_fix, derivative_adjust, mip_point_preclamp;
  bool compat_mode, blend_prt, aniso_override, disable_cube_wrap;
};

/// @brief Decode the sampler words used by the supported image families.
/// @details SQ_IMG_SAMP_WORD layouts follow ROCr resource_ai/gfx11/gfx12 and
/// PAL gfx10_sq_ko_reg.h. GFX12 reuses the high word-2 bits for PERF_MIP.
inline ImageSampler decode_image_sampler(const std::array<uint32_t, 4> &words,
                                         rj_code_arch_t arch) {
  const bool gfx9 = arch_is_cdna_4_or_lower(arch);
  const bool gfx10 = arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
  const bool gfx12 = arch == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t lod_bits = gfx12 ? 13 : 12;
  const uint32_t lod_mask = (1u << lod_bits) - 1;
  return {
      .clamp = {ImageClamp(words[0] & 7), ImageClamp((words[0] >> 3) & 7),
                ImageClamp((words[0] >> 6) & 7)},
      .min_lod_fixed8 = words[1] & lod_mask,
      .max_lod_fixed8 = (words[1] >> lod_bits) & lod_mask,
      .lod_bias_fixed8 = int32_t((words[2] & 0x3fffu) ^ 0x2000u) - 0x2000,
      .mag_filter = static_cast<ImageXyFilter>((words[2] >> 20) & 3),
      .min_filter = static_cast<ImageXyFilter>((words[2] >> 22) & 3),
      .mip_filter = static_cast<ImageMipFilter>((words[2] >> 26) & 3),
      .z_filter = (words[2] >> 24) & 3,
      .perf_mip = gfx12 ? (words[2] >> 30) | ((words[3] & 3) << 2) : (words[1] >> 24) & 15,
      .perf_z = words[1] >> 28,
      .border_color_type = words[3] >> 30,
      .filter_mode = (words[0] >> 29) & 3,
      .max_aniso_ratio = (words[0] >> 9) & 7,
      .aniso_threshold = (words[0] >> 16) & 7,
      .aniso_bias = (words[0] >> 21) & 63,
      .compare_function = (words[0] >> 12) & 7,
      .secondary_lod_bias = int32_t(((words[2] >> 14) & 63) ^ 32) - 32,
      .force_unnormalized = bool(words[0] & (1u << 15)),
      .truncate_coordinates = bool(words[0] & (1u << 27)),
      .force_degamma = bool(words[0] & (1u << 20)),
      .mc_coord_trunc = bool(words[0] & (1u << 19)),
      .skip_degamma = gfx9 ? bool(words[3] & (1u << 12)) : bool(words[0] & (1u << 31)),
      .filter_precision_fix = gfx9 && (words[2] & (1u << 30)),
      .derivative_adjust = !gfx9 && !gfx12 && (words[2] & (1u << 31)),
      .mip_point_preclamp = (gfx9 || gfx10) && (words[2] & (1u << 28)),
      .compat_mode = gfx9 && (words[0] & (1u << 31)),
      .blend_prt = !gfx12 && (words[2] & (1u << (gfx9 ? 29 : 30))),
      .aniso_override = bool(words[2] & (1u << (gfx9 ? 31 : 29))),
      .disable_cube_wrap = bool(words[0] & (1u << 28)),
  };
}

} // namespace rocjitsu::amdgpu
