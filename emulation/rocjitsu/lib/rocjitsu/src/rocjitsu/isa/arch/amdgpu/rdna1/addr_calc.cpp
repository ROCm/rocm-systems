// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/rdna1/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cassert>
#include <cstdint>

namespace rocjitsu {
namespace rdna1 {

namespace {

std::optional<uint32_t> read_smem_offset(uint32_t soffset, amdgpu::Wavefront &wf) {
  if (soffset == OPR_SMEM_OFFSET_NULL || soffset == 0x7F)
    return 0;
  if (soffset == OPR_SMEM_OFFSET_M0)
    return wf.m0();
  return amdgpu::try_read_scalar_selector(wf, soffset);
}

} // namespace

std::optional<uint64_t> smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf) {
  const uint32_t sbase_sel = inst.sbase * 2;
  auto base = amdgpu::try_read_scalar_selector64(wf, sbase_sel);
  if (!base)
    return std::nullopt;
  int64_t off = static_cast<int64_t>(static_cast<int32_t>(inst.offset << 11) >> 11);
  auto soffset = read_smem_offset(inst.soffset, wf);
  if (!soffset)
    return std::nullopt;
  off += *soffset;
  return (*base + off) & ~0x3ULL;
}

void flat_calculate_addresses(const FlatMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d) {
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d.lane_mask = exec;
  d.exec_mask = exec;
  int64_t offset = static_cast<int64_t>(static_cast<int32_t>(inst.offset << 20) >> 20);
  amdgpu::RegisterAccess regs(cu);
  uint64_t saddr_val = 0;
  if (inst.saddr != 0x7F) {
    const uint32_t sb_sel = inst.saddr;
    auto saddr = amdgpu::try_read_scalar_selector64(wf, sb_sel);
    if (!saddr) {
      amdgpu::reject_vector_memory_access(d);
      return;
    }
    saddr_val = *saddr;
  }
  uint32_t vbase = wf.vgpr_alloc().base + inst.addr;
  auto vaddr_region = regs.read_vgpr_region(vbase, inst.saddr != 0x7F ? 1 : 2, exec);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t vaddr;
    if (inst.saddr != 0x7F) {
      vaddr = vaddr_region.lane(0, lane);
    } else {
      vaddr = vaddr_region.lane64(0, lane);
    }
    d.per_lane_addr[lane] = saddr_val + vaddr + offset;
  }
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

} // namespace rdna1
} // namespace rocjitsu
