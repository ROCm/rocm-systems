// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_IMAGE_RESOURCE_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_IMAGE_RESOURCE_H_

#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <algorithm>
#include <array>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// Descriptor-only IMAGE_GET_RESINFO; no image allocation or texture addressing is needed.
inline std::array<uint32_t, 4> image_resource_info(const std::array<uint32_t, 8> &r,
                                                   rj_code_arch_t arch, uint32_t lod) {
  const auto field = [&](uint32_t bit, uint32_t width) {
    uint64_t word = r[bit / 32];
    if (bit % 32 + width > 32)
      word |= uint64_t{r[bit / 32 + 1]} << 32;
    return static_cast<uint32_t>((word >> (bit % 32)) & ((uint64_t{1} << width) - 1));
  };
  const uint32_t type = field(124, 4);
  if (type < 8)
    return {};
  const bool gfx9 = arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2;
  const bool gfx12 = arch == ROCJITSU_CODE_ARCH_RDNA4;
  const bool gfx10 = arch == ROCJITSU_CODE_ARCH_RDNA1;
  const bool msaa = type >= 14;
  const uint32_t base = msaa ? 0 : field(gfx12 ? 57 : 108, gfx12 ? 5 : 4);
  const uint32_t last = msaa ? 0 : field(gfx12 ? 111 : 112, gfx12 ? 5 : 4);
  if (last < base)
    return {};
  const uint32_t levels = last - base + 1;
  if (lod >= levels)
    return {0, 0, 0, levels};
  const uint32_t level = base + lod;
  const auto mip_size = [level](uint32_t size) { return std::max(1u, size >> level); };
  const uint32_t width = mip_size(field(gfx9 ? 64 : 62, gfx12 || gfx10 ? 16 : 14) + 1);
  const uint32_t height = mip_size(field(78, gfx12 || gfx10 ? 16 : 14) + 1);
  const uint32_t depth = field(128, gfx12 ? 14 : gfx10 ? 16 : 13) + 1;
  const uint32_t base_array = field(gfx9 ? 160 : 144, gfx10 ? 16 : 13);
  const uint32_t layers = depth > base_array ? depth - base_array : 0;
  const bool uav3d = !gfx9 && field(gfx12 ? 164 : 160, 1);
  switch (type) {
  case 8:
    return {width, 0, 0, levels};
  case 9:
  case 14:
    return {width, height, 0, levels};
  case 10:
    return {width, height, uav3d ? layers : mip_size(depth), levels};
  case 12:
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
