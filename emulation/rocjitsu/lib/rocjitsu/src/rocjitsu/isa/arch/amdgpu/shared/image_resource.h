// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_IMAGE_RESOURCE_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_IMAGE_RESOURCE_H_

#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include "util/bit.h"

#include <algorithm>
#include <array>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief Decoded GFX9 image descriptor (T#) for GET_RESINFO and IMAGE_LOAD/STORE.
struct Gfx9ImageResource {
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
  uint32_t swizzle_mode = 0;
  uint32_t type = 0;
  uint32_t base_array = 0;
  uint32_t array_pitch = 0;
};

/// SQ_RSRC_IMG_* resource types.
inline constexpr uint32_t kGfx9Image1d = 8;
inline constexpr uint32_t kGfx9Image2d = 9;
inline constexpr uint32_t kGfx9Image3d = 10;
inline constexpr uint32_t kGfx9ImageCube = 11;
inline constexpr uint32_t kGfx9Image1dArray = 12;
inline constexpr uint32_t kGfx9Image2dArray = 13;
inline constexpr uint32_t kGfx9Image2dMsaa = 14;
inline constexpr uint32_t kGfx9Image2dMsaaArray = 15;

/// @brief Decode the fields shared by GFX9 image queries and linear accesses.
inline Gfx9ImageResource decode_gfx9_image_resource(const std::array<uint32_t, 8> &r) {
  const auto field = [&](uint32_t bit, uint32_t width) {
    return util::extract_bits_from_words(r.data(), bit, width);
  };
  Gfx9ImageResource resource;
  resource.base = ((uint64_t{field(32, 8)} << 32) | r[0]) << 8;
  resource.data_format = field(52, 6);
  resource.num_format = field(58, 4);
  resource.width = field(64, 14) + 1;
  resource.height = field(78, 14) + 1;
  resource.selectors = field(96, 12);
  resource.base_level = field(108, 4);
  resource.last_level = field(112, 4);
  resource.swizzle_mode = field(116, 5);
  resource.type = field(124, 4);
  resource.depth = field(128, 13) + 1;
  resource.pitch = field(141, 16) + 1;
  resource.base_array = field(160, 13);
  resource.array_pitch = field(173, 4);
  return resource;
}

/// Descriptor-only IMAGE_GET_RESINFO; no image allocation or texture addressing is needed.
inline std::array<uint32_t, 4> image_resource_info(const std::array<uint32_t, 8> &r,
                                                   rj_code_arch_t arch, uint32_t lod) {
  const auto field = [&](uint32_t bit, uint32_t width) {
    return util::extract_bits_from_words(r.data(), bit, width);
  };
  const uint32_t type = field(124, 4);
  if (type < kGfx9Image1d)
    return {};
  const bool gfx9 = arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2;
  const Gfx9ImageResource gfx9_resource =
      gfx9 ? decode_gfx9_image_resource(r) : Gfx9ImageResource{};
  const bool gfx12 = arch == ROCJITSU_CODE_ARCH_RDNA4;
  const bool gfx10 = arch == ROCJITSU_CODE_ARCH_RDNA1;
  const bool msaa = type >= kGfx9Image2dMsaa;
  const uint32_t base =
      msaa ? 0 : (gfx9 ? gfx9_resource.base_level : field(gfx12 ? 57 : 108, gfx12 ? 5 : 4));
  const uint32_t last =
      msaa ? 0 : (gfx9 ? gfx9_resource.last_level : field(gfx12 ? 111 : 112, gfx12 ? 5 : 4));
  if (last < base)
    return {};
  const uint32_t levels = last - base + 1;
  if (lod >= levels)
    return {0, 0, 0, levels};
  const uint32_t level = base + lod;
  const auto mip_size = [level](uint32_t size) { return std::max(1u, size >> level); };
  const uint32_t width =
      mip_size(gfx9 ? gfx9_resource.width : field(62, gfx12 || gfx10 ? 16 : 14) + 1);
  const uint32_t height =
      mip_size(gfx9 ? gfx9_resource.height : field(78, gfx12 || gfx10 ? 16 : 14) + 1);
  const uint32_t depth = gfx9 ? gfx9_resource.depth : field(128, gfx12 ? 14 : gfx10 ? 16 : 13) + 1;
  const uint32_t base_array = gfx9 ? gfx9_resource.base_array : field(144, gfx10 ? 16 : 13);
  const uint32_t layers = depth > base_array ? depth - base_array : 0;
  const bool uav3d = !gfx9 && field(gfx12 ? 164 : 160, 1);
  switch (type) {
  case kGfx9Image1d:
    return {width, 0, 0, levels};
  case kGfx9Image2d:
  case kGfx9Image2dMsaa:
    return {width, height, 0, levels};
  case kGfx9Image3d:
    return {width, height, uav3d ? layers : mip_size(depth), levels};
  case kGfx9Image1dArray:
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
  const auto lods = regs.read_vgpr_region(base + lod_vgpr, 1, wf.exec());
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
#endif
