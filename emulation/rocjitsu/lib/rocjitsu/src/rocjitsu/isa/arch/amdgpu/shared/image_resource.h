// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/image_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include "util/bit.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// Translate the R8, RG8 and RGBA8 sRGB texture storage to its UNORM carrier.
constexpr uint32_t image_srgb_storage_format(uint32_t format, bool gfx10) {
  struct SrgbFormat {
    uint32_t srgb, unorm;
  };
  constexpr std::array<SrgbFormat, 3> gfx10_formats{{{128, 1}, {129, 14}, {130, 56}}};
  constexpr std::array<SrgbFormat, 3> gfx11_formats{{{64, 1}, {65, 14}, {66, 42}}};
  for (const auto &entry : (gfx10 ? gfx10_formats : gfx11_formats))
    if (format == entry.srgb)
      return entry.unorm;
  return format;
}

/// Native interpolation of 8/10-bit UNORM and 16-bit floats retains format-specific precision.
constexpr bool native_low_precision_filter(uint32_t format) {
  format = image_srgb_storage_format(format, false);
  return format == 1 || format == 14 || format == 36 || format == 42 || format == 13 ||
         format == 29 || format == 57;
}

constexpr bool native_block_filter(uint32_t format) { return format >= 109 && format <= 118; }

/// Formats with a qualified linear/mip filter in the native texture backend.
constexpr bool native_filterable_format(uint32_t format) {
  return native_low_precision_filter(format) || format == 22 || format == 50 || format == 63 ||
         native_block_filter(format);
}

/// Use the ordinary compute filter for a feedback sample on a format outside
/// the native texture filter. Point samples retain their existing route.
inline bool compute_filtered_image_feedback(rj_code_arch_t arch, uint32_t dim, uint32_t type,
                                            uint32_t resource_word1, uint32_t sampler_word2) {
  const uint32_t format = (resource_word1 >> (arch == ROCJITSU_CODE_ARCH_RDNA4 ? 17 : 20)) & 255u;
  const bool filtered_xy = (sampler_word2 & (15u << 20)) != 0;
  const bool filtered_mip = ((sampler_word2 >> 26) & 3u) == 2;
  return dim == kImageDim2d && type == kImage2d && !native_filterable_format(format) &&
         (filtered_xy || filtered_mip);
}

/// Select the 2D linear filter by storage format and effective coordinate mode.
/// Filter arithmetic for low-precision storage is independent of tiling and pitch.
template <typename Read, typename Backed>
bool uses_native_2d_sampler(rj_code_arch_t arch, uint32_t resource, uint32_t sampler, uint32_t dim,
                            bool unorm, Read read, Backed backed) {
  const bool gfx12 = arch == ROCJITSU_CODE_ARCH_RDNA4;
  if ((!gfx12 && arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA3_5) ||
      dim != kImageDim2d || !backed(resource, 8))
    return false;
  if (!backed(sampler, 4))
    return false;
  const bool normalized = !unorm && !(read(sampler) & (1u << 15));
  const uint32_t word3 = read(resource + 3);
  if ((word3 >> 28) != kImage2d || (read(resource + 5) & (1u << 24)))
    return false;
  const uint32_t word1 = read(resource + 1);
  const uint32_t format = (word1 >> (gfx12 ? 17 : 20)) & 255u;
  if ((word3 >> 20) & 31u) {
    const uint32_t filter = read(sampler + 2);
    const bool linear_xy = (filter & ((1u << 20) | (1u << 22))) != 0;
    const bool linear_mip = ((filter >> 26) & 3u) == 2;
    return native_low_precision_filter(format) && (linear_xy || linear_mip);
  }
  return native_low_precision_filter(format) ||
         (native_filterable_format(format) && (!native_block_filter(format) || normalized) &&
          !(read(resource + 4) & (gfx12 ? 0xffffu : 0x3fffu)));
}

/// Choose the compute sampler for supported ordinary descriptors. Value probes
/// supply a non-observing reader when planning waits before execution.
template <typename Read, typename Backed>
bool use_linear_image_path(uint32_t resource, uint32_t sampler, uint32_t dim, bool unorm,
                           bool sample, Read read, Backed backed) {
  if (!backed(resource, 8))
    return true;
  const auto word = [&](uint32_t n) { return read(resource + n); };
  const uint32_t format = (word(1) >> 20) & 255, type = word(3) >> 28;
  if (type < 8)
    return true;
  if (word(5) & (1u << 24))
    return true;
  if ((word(5) >> 8) & 0xfff)
    return false;
  if ((word(6) & (1u << 21)) && ((word(6) >> 24) || word(7)))
    return false;
  if ((format >= 109 && format <= 118) || (type == 11 && dim == 3) || (type >= 14 && dim >= 6))
    return false;
  if (!sample || unorm || type == 10 || !backed(sampler, 4) || (read(sampler) & (1u << 15)))
    return true;
  const uint32_t filter = read(sampler + 2);
  const uint32_t mag = (filter >> 20) & 3, min = (filter >> 22) & 3;
  return mag == min && min <= 1 && (read(sampler) & (7u << 9)) == 0 &&
         (read(sampler + 3) >> 30) == 0;
}

inline uint32_t rdna_image_format(rj_code_arch_t arch, uint32_t word1) {
  const bool gfx10 = arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
  return (word1 >> (arch == ROCJITSU_CODE_ARCH_RDNA4 ? 17 : 20)) & (gfx10 ? 511u : 255u);
}

inline bool image_descriptor_is_unbound(rj_code_arch_t arch, uint32_t word1, uint32_t word3) {
  return (word3 >> 28) < kImage1d || rdna_image_format(arch, word1) == 0;
}

/// @brief Decoded AMDGPU image descriptor (T#) for queries and image accesses.
struct ImageResource {
  uint64_t base = 0; ///< Byte address of mip 0, slice 0.
  uint32_t data_format = 0;
  uint32_t num_format = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t depth = 0; ///< Slices of a 3D image or layers of an array.
  uint32_t pitch = 0; ///< Row pitch in elements.
  uint32_t selectors = 0;
  uint32_t base_level = 0;
  uint32_t last_level = 0;
  uint32_t max_level = 0;      ///< Last physical mip in the allocation.
  uint32_t min_lod = 0;        ///< Raw resource LOD clamp; nonzero values are unqualified.
  uint32_t min_lod_warn = 0;   ///< PRT feedback threshold in units of 1/256 mip.
  uint32_t perf_mod = 0;       ///< Resource filter-performance modifier.
  bool corner_samples = false; ///< Sample coordinates refer to corners rather than centers.
  bool compressed = false;     ///< Compression metadata or compressed writes are enabled.
  /// Selects the GFX10+ residency or sampling-status map descriptor layout.
  bool linked_resource = false;
  uint32_t swizzle_mode = 0;
  uint32_t type = 0;
  uint32_t base_array = 0;
  uint32_t array_pitch = 0;
  bool uav3d = false;       ///< GFX12 UAV3D view control, separate from DEPTH_SCALE.
  uint32_t depth_scale = 0; ///< GFX11/12 descriptor scaling controls.
  uint32_t height_scale = 0;
  uint32_t width_scale = 0;
};

/// @brief GFX10.3+ non-array types whose word-4 field encodes pitch.
inline bool image_resource_encodes_pitch(uint32_t type) {
  return type == kImage1d || type == kImage2d;
}

/// @brief Decode the fields shared by GFX9 image queries and linear accesses.
inline ImageResource decode_gfx9_image_resource(const std::array<uint32_t, 8> &descriptor) {
  const auto field = [&](uint32_t bit, uint32_t width) {
    return util::extract_bits_from_words(descriptor.data(), bit, width);
  };
  ImageResource resource;
  resource.base = ((uint64_t{field(32, 8)} << 32) | descriptor[0]) << 8;
  resource.min_lod = field(40, 12);
  resource.perf_mod = field(92, 3);
  resource.data_format = field(52, 6);
  resource.num_format = field(58, 4);
  resource.width = field(64, 14) + 1;
  resource.height = field(78, 14) + 1;
  resource.selectors = field(96, 12);
  resource.base_level = field(108, 4);
  resource.last_level = field(112, 4);
  resource.max_level = field(188, 4);
  resource.compressed = field(213, 1);
  resource.swizzle_mode = field(116, 5);
  resource.type = field(124, 4);
  resource.depth = field(128, 13) + 1;
  resource.pitch = field(141, 16) + 1;
  resource.base_array = field(160, 13);
  resource.array_pitch = field(173, 4);
  return resource;
}

struct ImageMipRange {
  uint32_t first;
  uint32_t last;
};

inline ImageMipRange rdna_image_mip_range(rj_code_arch_t arch, uint32_t word1, uint32_t word3) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA4)
    return {(word1 >> 25) & 31u, (word3 >> 15) & 31u};
  return {(word3 >> 12) & 15u, (word3 >> 16) & 15u};
}

inline uint32_t rdna_image_lod_warning_threshold(rj_code_arch_t arch, uint32_t word5) {
  return arch == ROCJITSU_CODE_ARCH_RDNA4 ? (word5 >> 5) & 0x1fffu : (word5 >> 8) & 0xfffu;
}

/// @brief Decode a T# using the requesting generation's resource layout.
inline ImageResource decode_image_resource(const std::array<uint32_t, 8> &descriptor,
                                           rj_code_arch_t arch, bool r128 = false) {
  if (arch_is_cdna_4_or_lower(arch))
    return decode_gfx9_image_resource(descriptor);
  const auto field = [&](uint32_t bit, uint32_t width) {
    return util::extract_bits_from_words(descriptor.data(), bit, width);
  };
  const bool rdna4 = arch == ROCJITSU_CODE_ARCH_RDNA4;
  const bool rdna1 = arch == ROCJITSU_CODE_ARCH_RDNA1;
  ImageResource resource;
  resource.min_lod = rdna1 || arch == ROCJITSU_CODE_ARCH_RDNA2
                         ? field(40, 12)
                         : field(rdna4 ? 186 : 187, rdna4 ? 13 : 12);
  resource.base = ((uint64_t{field(32, 8)} << 32) | descriptor[0]) << 8;
  resource.perf_mod = field(180, 3);
  resource.corner_samples = field(183, 1);
  resource.linked_resource = field(184, 1);
  if (!resource.linked_resource)
    resource.min_lod_warn = rdna_image_lod_warning_threshold(arch, descriptor[5]);
  resource.type = field(124, 4);
  resource.width = field(62, rdna4 || rdna1 ? 16 : 14) + 1;
  resource.height = field(78, rdna4 || rdna1 ? 16 : 14) + 1;
  resource.data_format = rdna_image_format(arch, descriptor[1]);
  resource.selectors = field(96, 12);
  const auto mip = rdna_image_mip_range(arch, descriptor[1], descriptor[3]);
  resource.base_level = mip.first;
  resource.last_level = mip.last;
  resource.max_level = field(rdna4                                       ? 44
                             : rdna1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ? 164
                                                                         : 48,
                             rdna4 ? 5 : 4);
  resource.compressed = field(213, 1) || field(212, 1);
  resource.swizzle_mode = field(116, 5);
  // RDNA1 follows the manual's 16-bit DEPTH/BASE_ARRAY fields; ROCr's
  // resource_nv.h defines 13 bits. Reserved high bits in ROCr descriptors
  // are zero; the wider form remains unqualified on RDNA1 hardware.
  // On gfx1201, array loads and RESINFO ignore bit 13 of both array endpoints.
  resource.depth = field(128, rdna1 ? 16 : rdna4 && resource.type == kImage3d ? 14 : 13) + 1;
  resource.base_array = field(144, rdna1 ? 16 : 13);
  if (rdna4) {
    resource.uav3d = field(164, 1);
    if (resource.linked_resource) {
      resource.depth_scale = field(165, 5);
      resource.height_scale = field(170, 5);
      resource.width_scale = field(175, 5);
    }
  } else {
    resource.array_pitch = field(160, 4);
    // An ordinary GFX11.5 descriptor holds MIN_LOD_WARN here. Linked
    // resources overlay the same bits with three scales.
    if ((arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5) &&
        resource.linked_resource) {
      resource.depth_scale = field(168, 4);
      resource.height_scale = field(172, 4);
      resource.width_scale = field(176, 4);
    }
  }
  if (image_resource_encodes_pitch(resource.type)) {
    resource.depth = 1;
    // GFX10.3+ repurpose DEPTH as PITCH for non-array images.
    if (!rdna1)
      resource.pitch = field(128, rdna4 ? 16 : 14) + 1;
  }
  if (r128 && (rdna1 || arch == ROCJITSU_CODE_ARCH_RDNA2)) {
    // Four-word resources have no extended pitch, depth or allocation mip field.
    resource.pitch = 0;
    resource.depth = 1;
    resource.max_level = resource.last_level;
  }
  return resource;
}

/// Descriptor-only IMAGE_GET_RESINFO; no image allocation or texture addressing is needed.
inline std::array<uint32_t, 4> image_resource_info(const std::array<uint32_t, 8> &descriptor,
                                                   rj_code_arch_t arch, uint32_t lod) {
  const ImageResource resource = decode_image_resource(descriptor, arch);
  const uint32_t type = resource.type;
  if (type < kImage1d)
    return {};
  const bool gfx9 = arch_is_cdna_4_or_lower(arch);
  const bool msaa = type >= kImage2dMsaa;
  const uint32_t base = msaa ? 0 : resource.base_level;
  const uint32_t last = msaa ? 0 : resource.last_level;
  if (last < base)
    return {};
  const uint32_t levels = last - base + 1;
  if (lod >= levels)
    return {0, 0, 0, levels};
  const uint32_t level = base + lod;
  const auto mip_size = [level](uint32_t size) { return std::max(1u, size >> level); };
  const uint32_t width = mip_size(resource.width);
  const uint32_t height = mip_size(resource.height);
  const uint32_t depth = resource.depth;
  const uint32_t base_array = resource.base_array;
  const uint32_t layers = depth > base_array ? depth - base_array : 0;
  // GFX10/11 encode the 3D view flag in ARRAY_PITCH's low bit. GFX12 gives
  // UAV3D its own bit; DEPTH_SCALE does not change this descriptor-only query.
  const bool uav3d = resource.uav3d || (!gfx9 && (resource.array_pitch & 1));
  switch (type) {
  case kImage1d:
    return {width, 0, 0, levels};
  case kImage2d:
  case kImage2dMsaa:
    return {width, height, 0, levels};
  case kImage3d:
    return {width, height, uav3d ? layers : mip_size(depth), levels};
  case kImage1dArray:
    return {width, layers, 0, levels};
  default:
    return {width, height, layers, levels}; // Cube depth is in faces, not cubes.
  }
}

inline void execute_image_resource_info(Wavefront &wf, uint32_t resource, uint32_t lod_vgpr,
                                        uint32_t dst_reg_base, uint32_t mask, bool r128, bool a16,
                                        bool result_in_range = true) {
  const auto arch = wf.cu().arch();
  const bool gfx10 = arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
  const bool modern_sources = has_gfx11_image_operands(arch);
  if (gfx10 || modern_sources) {
    const unsigned data_words = std::max(1, std::popcount(mask));
    if (!wf.vgpr_alloc().contains(dst_reg_base, data_words))
      return;
    if (mask == 0) {
      RegisterAccess regs(wf);
      for (uint64_t lanes = wf.exec(); lanes; lanes &= lanes - 1)
        regs.write_vgpr(dst_reg_base, std::countr_zero(lanes), 0);
      return;
    }
  }
  if (modern_sources)
    lod_vgpr = gfx115_source_vgpr(lod_vgpr, wf.num_vgprs());
  const uint32_t words = r128 ? 4 : 8;
  std::array<uint32_t, 8> descriptor{};
  // Validate each selector before reading, including descriptors ending in VCC.
  for (uint32_t i = 0; i < words; ++i)
    if (!scalar_selector_range_is_backed(wf, resource + i, 1))
      return;
  if (!result_in_range)
    return;
  for (uint32_t i = 0; i < words; ++i)
    descriptor[i] = read_scalar_selector(wf, resource + i);
  RegisterAccess regs(wf);
  const uint32_t base = wf.vgpr_alloc().base;
  const auto lods = regs.read_vgpr_region(
      base + lod_vgpr, 1, wf.exec(),
      image_coordinate_component_bytes(0, !arch_is_cdna_4_or_lower(arch) && a16));
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(wf.exec() & (uint64_t{1} << lane)))
      continue;
    // Read LOD before writing any result, since VADDR may alias VDATA.
    const auto result = image_resource_info(descriptor, wf.cu().arch(),
                                            lods.lane(0, lane) & (a16 ? 0xffffu : 0xffffffffu));
    // RESINFO leaves the extra TFE status register unchanged on gfx11/gfx12.
    uint32_t dst = dst_reg_base;
    for (uint32_t component = 0; component < 4; ++component)
      if (mask & (1u << component))
        regs.write_vgpr(dst++, lane, result[component]);
  }
}

} // namespace rocjitsu::amdgpu
