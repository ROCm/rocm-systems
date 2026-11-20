// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/cdna1/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

namespace rocjitsu {
namespace cdna1 {

std::optional<uint64_t> smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf,
                                               amdgpu::ScalarMemState *state) {
  return amdgpu::addr_calc::smem_calculate_address(inst, wf, state, true);
}

void flat_calculate_addresses(const FlatMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::flat_calculate_addresses<amdgpu::addr_calc::ScratchAddressMode::LegacyCdna>(
      inst, wf, d);
}

void mubuf_calculate_addresses(const MubufMachineInst &inst, amdgpu::Wavefront &wf,
                               amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::mubuf_calculate_addresses(inst, wf, d);
}

void mtbuf_calculate_addresses(const MtbufMachineInst &inst, amdgpu::Wavefront &wf,
                               amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::mtbuf_calculate_addresses(inst, wf, d);
}

void ds_calculate_addresses(const DsMachineInst &inst, amdgpu::Wavefront &wf,
                            amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::ds_calculate_addresses(inst, wf, d);
}

void ds_calculate_src2_addresses(const DsMachineInst &inst, amdgpu::Wavefront &wf,
                                 amdgpu::VectorMemState &d) {
  // MI100 11.3.3. Bit 15 selects a signed 15-bit displacement from the
  // immediate or the high bits of ADDR. Both forms scale by four, even for B64.
  d.ds_src2 = true;
  d.wg_id = wf.wg_id();
  d.wf_id = wf.wf_id();
  d.exec_mask = d.lane_mask = wf.exec();
  d.wf_size = wf.wf_size();
  const uint32_t size = std::min(wf.lds_size(), wf.m0());
  const uint32_t immediate = (inst.offset1 << 8) | inst.offset0;
  const uint32_t addr_reg = ds_src2_address_register(inst.addr, wf.vgpr_alloc().count);
  const auto addresses =
      amdgpu::RegisterAccess(wf).read_vgpr_region(wf.vgpr_alloc().base + addr_reg, 1, d.exec_mask);
  for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
    if (!(d.exec_mask & (uint64_t{1} << lane)))
      continue;
    const uint32_t addr = addresses.lane(0, lane);
    const bool indirect = (immediate & 0x8000u) != 0;
    const uint32_t offset = indirect ? addr >> 17 : immediate;
    const int32_t displacement = static_cast<int32_t>(offset ^ 0x4000u) - 0x4000;
    const uint32_t dst = indirect ? addr & 0x1ffffu : addr;
    const int64_t src = int64_t{dst} + displacement * 4;
    if (uint64_t{dst} + d.elem_size > size) {
      d.lane_mask &= ~(uint64_t{1} << lane);
      continue;
    }
    d.per_lane_addr[lane] = uint64_t{wf.lds_base()} + dst;
    if (src < 0 || static_cast<uint64_t>(src) + d.elem_size > size)
      d.ds2_oob_lane_mask |= uint64_t{1} << lane;
    else
      d.ds2_per_lane_addr[lane] = uint64_t{wf.lds_base()} + src;
  }
}

} // namespace cdna1
} // namespace rocjitsu
