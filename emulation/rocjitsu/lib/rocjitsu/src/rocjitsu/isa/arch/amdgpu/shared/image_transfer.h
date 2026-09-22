// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_IMAGE_TRANSFER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_IMAGE_TRANSFER_H_

#include "rocjitsu/isa/arch/amdgpu/shared/buffer_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/image_address.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <bit>
#include <cmath>

namespace rocjitsu::amdgpu {

/// Prepare a bounded subset of uncompressed GFX11/12 integer-coordinate image transfers.
inline bool prepare_image_transfer(Wavefront &wf, VectorMemState &d, uint32_t resource,
                                   uint32_t data, std::array<uint32_t, 3> coords, uint32_t dim,
                                   uint32_t mask, bool d16, bool unsupported_flags,
                                   uint32_t sampler = ~0u) {
  const auto unsupported = [&] {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return false;
  };
  const auto arch = wf.cu().arch();
  const bool gfx12 = arch == ROCJITSU_CODE_ARCH_RDNA4;
  if ((!gfx12 && arch != ROCJITSU_CODE_ARCH_RDNA3 && arch != ROCJITSU_CODE_ARCH_RDNA3_5) ||
      unsupported_flags || (dim != 1 && dim != 5) || !mask)
    return unsupported();
  std::array<uint32_t, 8> r{};
  for (uint32_t i = 0; i < r.size(); ++i)
    if (!scalar_selector_range_is_backed(wf, resource + i, 1))
      return false;
  for (uint32_t i = 0; i < r.size(); ++i)
    r[i] = read_scalar_selector(wf, resource + i);
  const uint32_t type = r[3] >> 28;
  const uint32_t swizzle = (r[3] >> 20) & 31;
  const uint32_t width = ((r[1] >> 30) | ((r[2] & (gfx12 ? 0x3fff : 0xfff)) << 2)) + 1;
  const uint32_t height = ((r[2] >> 14) & (gfx12 ? 0xffff : 0x3fff)) + 1;
  const uint32_t image_format = (r[1] >> (gfx12 ? 17 : 20)) & 255;
  const uint32_t format = image_format == 66 ? 42 : image_format;
  d.image_srgb = image_format == 66;
  if (!gfx12 && (r[6] & (1u << 21)))
    return unsupported(); // GFX11 DCC decoding is not implemented.
  const uint32_t bytes = buffer_format_bytes(format);
  const bool mipmapped = gfx12 ? ((r[1] >> 12) & 31) || ((r[1] >> 25) & 31) || ((r[3] >> 15) & 31)
                               : ((r[1] >> 16) & 15) || ((r[3] >> 12) & 255);
  if ((type != 9 && type != 13) || mipmapped || (r[4] >> 16) || !bytes ||
      (!d.is_load &&
       (d.image_srgb || (mask != 15 && !(mask == 1 && format >= 20 && format <= 22)))))
    return unsupported();
  const bool sample = sampler != ~0u;
  bool normalized = true;
  if (sample) {
    if (dim != 1 || !d.is_load)
      return unsupported();
    std::array<uint32_t, 4> s{};
    for (uint32_t i = 0; i < s.size(); ++i) {
      if (!scalar_selector_range_is_backed(wf, sampler + i, 1))
        return false;
      s[i] = read_scalar_selector(wf, sampler + i);
    }
    // Nearest filtering and clamp-to-edge at the sole mip level.
    if ((s[0] & 7) != 2 || ((s[0] >> 3) & 7) != 2 ||
        (s[0] & ((7u << 9) | (7u << 12) | (3u << 29))) || (s[1] & (gfx12 ? 0x1fff : 0xfff)) ||
        (s[2] & 0x03f00000u) || ((s[2] >> 26) & 3) > 1)
      return unsupported();
    normalized = !(s[0] & (1u << 15));
    d.image_srgb &= !(s[0] & (1u << 31));
  }
  // The functional model retains uncompressed backing for image operations.
  // GFX11 metadata-only clears still need separate command processing.
  d.buffer_components = std::popcount(mask);
  d.buffer_d16 = d16;
  d.buffer_format = format;
  d.buffer_format_encoding = BufferFormatEncoding::Gfx11;
  uint32_t component = 0;
  for (uint32_t i = 0; i < 4; ++i)
    if (mask & (1u << i))
      d.buffer_selectors |= ((r[3] >> (3 * i)) & 7) << (3 * component++);
  d.wf_size = wf.wf_size();
  d.exec_mask = wf.exec();
  d.elem_size = bytes;
  d.num_elems = 1;
  d.dst_reg_base = wf.vgpr_alloc().base + data;
  const uint32_t registers = d16 ? (d.buffer_components + 1) / 2 : d.buffer_components;
  if (data >= wf.num_vgprs() || registers > wf.num_vgprs() - data || coords[0] >= wf.num_vgprs() ||
      coords[1] >= wf.num_vgprs() || (dim == 5 && coords[2] >= wf.num_vgprs()))
    return unsupported();
  const uint64_t base =
      addr_calc::buffer_virtual_address(((uint64_t{r[1] & 255} << 32) | r[0]) << 8);
  const uint32_t pitch_field = r[4] & (gfx12 ? 0xffff : 0x3fff);
  const uint32_t pitch = swizzle == 0 && pitch_field ? pitch_field + 1 : width;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(wf.exec() & (uint64_t{1} << lane)))
      continue;
    uint32_t x = wf.debug_read_vgpr(coords[0], lane);
    uint32_t y = wf.debug_read_vgpr(coords[1], lane);
    if (sample) {
      const double u = std::bit_cast<float>(x), v = std::bit_cast<float>(y);
      if (!std::isfinite(u) || !std::isfinite(v))
        return unsupported();
      x = std::clamp(std::floor(u * (normalized ? width : 1)), 0.0, double(width - 1));
      y = std::clamp(std::floor(v * (normalized ? height : 1)), 0.0, double(height - 1));
    }
    const uint32_t z = dim == 5 ? wf.debug_read_vgpr(coords[2], lane) : 0;
    if (z)
      return unsupported();
    if (x >= width || y >= height)
      continue;
    const auto address = gfx12 ? gfx12_image_address(base, x, y, pitch, bytes, swizzle)
                               : gfx11_image_address(base, x, y, pitch, bytes, swizzle);
    if (!address)
      return unsupported();
    d.per_lane_addr[lane] = *address;
    d.lane_mask |= uint64_t{1} << lane;
  }
  if (!d.is_load)
    capture_buffer_format_store(wf, d, d.dst_reg_base);
  return true;
}

} // namespace rocjitsu::amdgpu

#endif
