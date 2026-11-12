// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/image_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include "util/bit.h"

#include <algorithm>
#include <array>
#include <cstdint>

namespace rocjitsu::amdgpu {

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

/// @brief Decode a T# using the requesting generation's resource layout.
inline ImageResource decode_image_resource(const std::array<uint32_t, 8> &descriptor,
                                           rj_code_arch_t arch) {
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
  resource.type = field(124, 4);
  resource.width = field(62, rdna4 || rdna1 ? 16 : 14) + 1;
  resource.height = field(78, rdna4 || rdna1 ? 16 : 14) + 1;
  resource.data_format = field(rdna4 ? 49 : 52, rdna1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ? 9 : 8);
  resource.selectors = field(96, 12);
  resource.base_level = field(rdna4 ? 57 : 108, rdna4 ? 5 : 4);
  resource.last_level = field(rdna4 ? 111 : 112, rdna4 ? 5 : 4);
  resource.max_level = field(rdna4                                       ? 44
                             : rdna1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ? 164
                                                                         : 48,
                             rdna4 ? 5 : 4);
  resource.compressed = field(213, 1) || field(212, 1);
  resource.swizzle_mode = field(116, 5);
  // RDNA1 follows the manual's 16-bit DEPTH/BASE_ARRAY fields; ROCr's
  // resource_nv.h defines 13 bits. Reserved high bits in ROCr descriptors
  // are zero; the wider form remains unqualified on RDNA1 hardware.
  resource.depth = field(128, rdna1 ? 16 : rdna4 ? 14 : 13) + 1;
  resource.base_array = field(144, rdna1 ? 16 : 13);
  if (rdna4) {
    resource.uav3d = field(164, 1);
    resource.depth_scale = field(165, 5);
    resource.height_scale = field(170, 5);
    resource.width_scale = field(175, 5);
  } else {
    resource.array_pitch = field(160, 4);
    if (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5) {
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
                                        uint32_t dst_reg_base, uint32_t mask, bool r128, bool a16) {
  const uint32_t words = r128 ? 4 : 8;
  std::array<uint32_t, 8> descriptor{};
  // Validate each selector before reading, including descriptors ending in VCC.
  for (uint32_t i = 0; i < words; ++i)
    if (!scalar_selector_range_is_backed(wf, resource + i, 1))
      return;
  for (uint32_t i = 0; i < words; ++i)
    descriptor[i] = read_scalar_selector(wf, resource + i);
  RegisterAccess regs(wf);
  const uint32_t base = wf.vgpr_alloc().base;
  const auto lods = regs.read_vgpr_region(base + lod_vgpr, 1, wf.exec(), a16 ? 0x3 : 0xf);
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
