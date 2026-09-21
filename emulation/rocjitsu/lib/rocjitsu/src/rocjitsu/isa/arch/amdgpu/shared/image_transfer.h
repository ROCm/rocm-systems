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

namespace rocjitsu::amdgpu {

/// Prepare a bounded subset of uncompressed GFX12 integer-coordinate image transfers.
inline bool prepare_image_transfer(Wavefront &wf, VectorMemState &d, uint32_t resource,
                                   uint32_t data, std::array<uint32_t, 3> coords, uint32_t dim,
                                   uint32_t mask, bool d16, bool unsupported_flags) {
  const auto unsupported = [&] {
    wf.report_instruction_execution_error(InstructionExecutionError::UnsupportedOperandValue);
    return false;
  };
  if (wf.cu().arch() != ROCJITSU_CODE_ARCH_RDNA4 || unsupported_flags || (dim != 1 && dim != 5) ||
      !mask)
    return unsupported();
  std::array<uint32_t, 8> r{};
  for (uint32_t i = 0; i < r.size(); ++i)
    if (!scalar_selector_range_is_backed(wf, resource + i, 1))
      return false;
  for (uint32_t i = 0; i < r.size(); ++i)
    r[i] = read_scalar_selector(wf, resource + i);
  const uint32_t type = r[3] >> 28;
  const uint32_t swizzle = (r[3] >> 20) & 31;
  const uint32_t width = ((r[1] >> 30) | ((r[2] & 0x3fff) << 2)) + 1;
  const uint32_t height = ((r[2] >> 14) & 0xffff) + 1;
  const uint32_t format = (r[1] >> 17) & 255;
  const uint32_t bytes = buffer_format_bytes(format);
  if ((type != 9 && type != 13) || ((r[1] >> 12) & 31) || ((r[1] >> 25) & 31) ||
      ((r[3] >> 15) & 31) || (r[4] >> 16) || !bytes ||
      (!d.is_load && mask != 15 && !(mask == 1 && format >= 20 && format <= 22)))
    return unsupported();
  // GFX12 compression is gated by the allocation's PTE.D bit. The functional
  // memory model keeps uncompressed backing, so descriptor compression enables
  // do not change addressing. GFX11's separate metadata layout is not covered.
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
  const uint32_t pitch = swizzle == 0 && (r[4] & 0xffff) ? (r[4] & 0xffff) + 1 : width;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(wf.exec() & (uint64_t{1} << lane)))
      continue;
    const uint32_t x = wf.debug_read_vgpr(coords[0], lane);
    const uint32_t y = wf.debug_read_vgpr(coords[1], lane);
    const uint32_t z = dim == 5 ? wf.debug_read_vgpr(coords[2], lane) : 0;
    if (z)
      return unsupported();
    if (x >= width || y >= height)
      continue;
    const auto address = gfx12_image_address(base, x, y, pitch, bytes, swizzle);
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
