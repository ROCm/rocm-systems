// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_GFX9_IMAGE_ACCESS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_GFX9_IMAGE_ACCESS_H_

#include "rocjitsu/isa/arch/amdgpu/shared/buffer_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/vm/amdgpu/buffer_format.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/result.h"

#include <array>
#include <bit>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief The GFX9 image resource (T#) fields IMAGE_LOAD and IMAGE_STORE read.
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
enum : uint32_t {
  kGfx9Image1d = 8,
  kGfx9Image2d = 9,
  kGfx9Image3d = 10,
  kGfx9ImageCube = 11,
  kGfx9Image1dArray = 12,
  kGfx9Image2dArray = 13,
};

inline Gfx9ImageResource decode_gfx9_image_resource(const std::array<uint32_t, 8> &r) {
  const auto field = [&](uint32_t bit, uint32_t width) {
    uint64_t word = r[bit / 32];
    if (bit % 32 + width > 32)
      word |= uint64_t{r[bit / 32 + 1]} << 32;
    return static_cast<uint32_t>((word >> (bit % 32)) & ((uint64_t{1} << width) - 1));
  };
  Gfx9ImageResource t;
  t.base = ((uint64_t{field(32, 8)} << 32) | r[0]) << 8;
  t.data_format = field(52, 6);
  t.num_format = field(58, 4);
  t.width = field(64, 14) + 1;
  t.height = field(78, 14) + 1;
  t.selectors = field(96, 12);
  t.base_level = field(108, 4);
  t.last_level = field(112, 4);
  t.swizzle_mode = field(116, 5);
  t.type = field(124, 4);
  t.depth = field(128, 13) + 1;
  t.pitch = field(141, 16) + 1;
  t.base_array = field(160, 13);
  t.array_pitch = field(173, 4);
  return t;
}

/// @brief Set up a GFX9 IMAGE_LOAD or IMAGE_STORE as a formatted vector access.
/// @details Each active lane addresses one texel of a linear, single-level
/// image with unnormalized coordinates read from @p vaddr. Lanes outside the
/// image are left out of the lane mask, so a load returns zero and a store is
/// dropped. A null resource drops every lane. The texel layout is the one
/// addrlib gives a linear surface: rows of PITCH elements, one row per 1D slice
/// and HEIGHT rows per 2D slice. DMASK picks the shader components after the
/// descriptor's channel selects, packed into consecutive VGPRs; a store
/// inverts those selects the way formatted buffer stores do.
/// @returns Failure for a resource this does not model yet: a tiled swizzle
/// mode, mip levels, a cube, MSAA, a view that starts past layer 0, a format
/// outside the buffer format table, or the LWE status register.
inline util::Result prepare_gfx9_image_access(Wavefront &wf, VectorMemState &d, uint32_t resource,
                                              uint32_t vaddr, uint32_t dmask, bool a16, bool d16,
                                              bool lwe) {
  const uint64_t exec = wf.exec();
  d.wf_size = wf.wf_size();
  d.wg_id = wf.wg_id();
  d.wf_id = wf.wf_id();
  d.exec_mask = exec;
  d.lane_mask = 0;
  d.per_lane_addr.fill(0);
  d.num_elems = 1;
  d.elem_size = 1;
  d.buffer_d16 = d16;
  d.buffer_components = static_cast<uint32_t>(std::popcount(dmask & 15u));
  d.buffer_format_encoding = BufferFormatEncoding::Gfx9;
  if (lwe || d.buffer_components == 0)
    return util::Result::failure();
  if (!scalar_selector_range_is_backed(wf, resource, 8))
    return util::Result::success();

  std::array<uint32_t, 8> words{};
  for (uint32_t i = 0; i < words.size(); ++i)
    words[i] = read_scalar_selector(wf, resource + i);
  const Gfx9ImageResource t = decode_gfx9_image_resource(words);

  // Shader component k is the k-th component DMASK enables.
  uint32_t selectors = 0;
  for (uint32_t component = 0, k = 0; component < 4; ++component)
    if (dmask & (1u << component))
      selectors |= ((t.selectors >> (3 * component)) & 7) << (3 * k++);
  d.buffer_selectors = selectors;

  if (t.type < kGfx9Image1d || t.data_format == 0)
    return util::Result::success();
  const bool layered = t.type == kGfx9Image1dArray || t.type == kGfx9Image2dArray;
  if (t.type == kGfx9ImageCube || t.type > kGfx9Image2dArray || t.swizzle_mode != 0 ||
      t.base_level != 0 || t.last_level != 0 || t.base_array != 0 || t.array_pitch != 0 ||
      t.data_format > 14 || t.num_format == 6 || t.num_format > 7)
    return util::Result::failure();
  d.buffer_format = t.data_format | (t.num_format << 4);
  const auto format = decode_buffer_format(d.buffer_format, BufferFormatEncoding::Gfx9);
  if (format.failed())
    return util::Result::failure();
  d.decoded_buffer_format = format.value();
  uint32_t bits = 0;
  for (uint32_t width : d.decoded_buffer_format.widths)
    bits += width;
  d.elem_size = bits / 8;

  const bool has_y = t.type != kGfx9Image1d && t.type != kGfx9Image1dArray;
  const uint32_t coordinates = 1 + (has_y ? 1 : 0) + (t.type == kGfx9Image3d || layered ? 1 : 0);
  const uint32_t slice_rows = has_y ? t.height : 1;
  RegisterAccess regs(wf);
  const auto addresses = regs.read_vgpr_region(wf.vgpr_alloc().base + vaddr,
                                               a16 ? (coordinates + 1) / 2 : coordinates, exec);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (uint64_t{1} << lane)))
      continue;
    std::array<uint32_t, 3> c{};
    for (uint32_t i = 0; i < coordinates; ++i)
      c[i] =
          a16 ? (addresses.lane(i / 2, lane) >> (16 * (i % 2))) & 0xffffu : addresses.lane(i, lane);
    const uint32_t x = c[0];
    const uint32_t y = has_y ? c[1] : 0;
    const uint32_t slice = coordinates == 3 ? c[2] : coordinates == 2 && !has_y ? c[1] : 0;
    if (x >= t.width || y >= t.height || slice >= t.depth)
      continue;
    const uint64_t element = (uint64_t{slice} * slice_rows + y) * t.pitch + x;
    d.per_lane_addr[lane] = addr_calc::buffer_virtual_address(t.base + element * d.elem_size);
    d.lane_mask |= uint64_t{1} << lane;
  }
  return util::Result::success();
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_GFX9_IMAGE_ACCESS_H_
