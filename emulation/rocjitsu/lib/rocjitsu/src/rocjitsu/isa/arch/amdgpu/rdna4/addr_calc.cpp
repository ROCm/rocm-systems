// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <optional>

namespace rocjitsu {
namespace rdna4 {

namespace {

bool has_saddr(uint32_t saddr) {
  // LLVM prints saddr=0x7c as "off" for GFX12 global/flat memory ops.
  // Keep 0x7f as a no-base sentinel for older generated configs/tests.
  return saddr != 0x7C && saddr != 0x7F;
}

std::optional<uint32_t> read_optional_sreg_m0(uint32_t reg, amdgpu::Wavefront &wf) {
  if (reg == OPR_SREG_M0_NULL)
    return 0;
  if (reg == OPR_SREG_M0_M0)
    return wf.m0();
  return amdgpu::try_read_scalar_selector(wf, reg);
}

std::optional<uint32_t> read_smem_offset(uint32_t soffset, amdgpu::Wavefront &wf) {
  if (soffset == OPR_SMEM_OFFSET_NULL || soffset == 0x7F)
    return 0;
  if (soffset == OPR_SMEM_OFFSET_M0)
    return wf.m0();
  return amdgpu::try_read_scalar_selector(wf, soffset);
}

void init_vector_mem_state(amdgpu::Wavefront &wf, amdgpu::VectorMemState &d) {
  uint64_t exec = wf.exec();
  d.lane_mask = exec;
  d.exec_mask = exec;
  d.wf_size = wf.wf_size();
  d.wg_id = wf.wg_id();
  d.wf_id = wf.wf_id();
}

bool buffer_range_exceeds(uint64_t offset, uint32_t payload, uint64_t bound) {
  return offset > bound || payload > bound - offset;
}

bool vbuffer_is_format_op(uint32_t op) { return op <= 15 || op == 38 || op == 39; }

} // namespace

std::optional<uint64_t> smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf) {
  // GFX12 SMEM: sbase is an aligned SGPR pair, ioffset is a 24-bit signed immediate,
  // soffset is an SGPR/M0/null selector from rdna4/operand_types.h.
  const uint32_t sbase_sel = inst.sbase * 2;
  auto base = amdgpu::try_read_scalar_selector64(wf, sbase_sel);
  if (!base)
    return std::nullopt;
  int64_t off = static_cast<int64_t>(static_cast<int32_t>(inst.ioffset << 8) >> 8);
  auto soffset = read_smem_offset(inst.soffset, wf);
  if (!soffset)
    return std::nullopt;
  off += *soffset;
  return (*base + off) & ~0x3ULL;
}

void flat_calculate_addresses(const VflatMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d) {
  // GFX12 VFLAT: 24-bit signed offset, optional SGPR base via saddr.
  auto &cu = wf.cu();
  init_vector_mem_state(wf, d);
  uint64_t exec = d.exec_mask;
  int64_t offset = static_cast<int64_t>(static_cast<int32_t>(inst.ioffset << 8) >> 8);
  amdgpu::RegisterAccess regs(cu);
  uint64_t saddr_val = 0;
  if (has_saddr(inst.saddr)) {
    const uint32_t sb_sel = inst.saddr;
    auto saddr = amdgpu::try_read_scalar_selector64(wf, sb_sel);
    if (!saddr) {
      amdgpu::reject_vector_memory_access(d);
      return;
    }
    saddr_val = *saddr;
  }
  uint32_t priv_hi = static_cast<uint32_t>(wf.private_aperture_base() >> 32);
  uint64_t scratch_base = wf.scratch_base();
  uint32_t lane_stride = wf.scratch_lane_size();
  uint32_t vbase = wf.vgpr_alloc().base + inst.vaddr;
  auto vaddr_region = regs.read_vgpr_region(vbase, has_saddr(inst.saddr) ? 1 : 2, exec);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t vaddr;
    if (has_saddr(inst.saddr)) {
      vaddr = vaddr_region.lane(0, lane);
    } else {
      vaddr = vaddr_region.lane64(0, lane);
    }
    uint64_t addr = saddr_val + vaddr + offset;
    if (priv_hi != 0 && static_cast<uint32_t>(addr >> 32) == priv_hi)
      addr = scratch_base + static_cast<uint64_t>(lane) * lane_stride + (addr & 0xFFFFFFFFULL);
    d.per_lane_addr[lane] = addr;
  }
}

void flat_calculate_addresses(const VglobalMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d) {
  // GFX12 VGLOBAL: 24-bit signed offset, optional SGPR base via saddr.
  auto &cu = wf.cu();
  init_vector_mem_state(wf, d);
  uint64_t exec = d.exec_mask;
  int64_t offset = static_cast<int64_t>(static_cast<int32_t>(inst.ioffset << 8) >> 8);
  amdgpu::RegisterAccess regs(cu);
  uint64_t saddr_val = 0;
  if (has_saddr(inst.saddr)) {
    const uint32_t sb_sel = inst.saddr;
    auto saddr = amdgpu::try_read_scalar_selector64(wf, sb_sel);
    if (!saddr) {
      amdgpu::reject_vector_memory_access(d);
      return;
    }
    saddr_val = *saddr;
  }
  uint32_t vbase = wf.vgpr_alloc().base + inst.vaddr;
  auto vaddr_region = regs.read_vgpr_region(vbase, has_saddr(inst.saddr) ? 1 : 2, exec);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t vaddr;
    if (has_saddr(inst.saddr)) {
      vaddr = vaddr_region.lane(0, lane);
    } else {
      vaddr = vaddr_region.lane64(0, lane);
    }
    d.per_lane_addr[lane] = saddr_val + vaddr + offset;
  }
}

void flat_calculate_addresses(const VscratchMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d) {
  // GFX12 VSCRATCH: scratch_base + lane scratch slice + optional VGPR
  // (32-bit) + optional saddr + signed 24-bit offset.
  auto &cu = wf.cu();
  init_vector_mem_state(wf, d);
  uint64_t exec = d.exec_mask;
  int64_t offset = static_cast<int64_t>(static_cast<int32_t>(inst.ioffset << 8) >> 8);
  uint64_t scratch_base = wf.scratch_base();
  uint32_t lane_stride = wf.scratch_lane_size();
  uint32_t saddr_val = 0;
  amdgpu::RegisterAccess regs(cu);
  if (has_saddr(inst.saddr)) {
    const uint32_t sb_sel = inst.saddr;
    auto saddr = amdgpu::try_read_scalar_selector(wf, sb_sel);
    if (!saddr) {
      amdgpu::reject_vector_memory_access(d);
      return;
    }
    saddr_val = *saddr;
  }
  uint32_t vbase = wf.vgpr_alloc().base + inst.vaddr;
  std::optional<amdgpu::RegisterAccess::VgprReadRegion> vaddr_region;
  if (inst.sve)
    vaddr_region.emplace(regs.read_vgpr_region(vbase, 1, exec));
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t vaddr = inst.sve ? vaddr_region->lane(0, lane) : 0;
    d.per_lane_addr[lane] =
        scratch_base + static_cast<uint64_t>(lane) * lane_stride + vaddr + saddr_val + offset;
  }
}

void mubuf_calculate_addresses(const VbufferMachineInst &inst, amdgpu::Wavefront &wf,
                               amdgpu::VectorMemState &d) {
  // GFX12 VBUFFER: rsrc is the first SGPR in the 4-dword resource descriptor,
  // soffset is a 7-bit SGPR/null selector, and ioffset is a signed immediate.
  auto &cu = wf.cu();
  init_vector_mem_state(wf, d);
  uint64_t exec = d.exec_mask;
  const uint32_t sb_sel = inst.rsrc;
  if (!amdgpu::scalar_selector_range_is_backed(wf, sb_sel, 4)) {
    amdgpu::reject_vector_memory_access(d);
    return;
  }
  uint32_t srd0 = amdgpu::read_scalar_selector(wf, sb_sel);
  uint32_t srd1 = amdgpu::read_scalar_selector(wf, sb_sel + 1);
  uint32_t num_records = amdgpu::read_scalar_selector(wf, sb_sel + 2);
  uint32_t srd3 = amdgpu::read_scalar_selector(wf, sb_sel + 3);
  uint64_t base_addr = (static_cast<uint64_t>(srd1 & 0xFFFF) << 32) | srd0;
  constexpr std::array<uint32_t, 4> kStrideMultipliers = {1, 4, 8, 32};
  uint32_t raw_stride = (srd1 >> 16) & 0x3FFF;
  uint32_t stride = raw_stride * kStrideMultipliers[(srd3 >> 18) & 0x3];
  uint32_t swizzle_enable = srd1 >> 30;
  uint32_t oob_select = (srd3 >> 28) & 0x3;
  amdgpu::RegisterAccess regs(cu);
  auto soffset = read_optional_sreg_m0(inst.soffset, wf);
  if (!soffset) {
    amdgpu::reject_vector_memory_access(d);
    return;
  }
  uint32_t soffset_val = *soffset;
  int64_t ioff = static_cast<int64_t>(static_cast<int32_t>(inst.ioffset << 8) >> 8);
  assert(!inst.idxen && "Vbuffer idxen not yet supported");
  assert(d.elem_size != 0 && d.num_elems != 0);
  const bool per_component =
      d.num_elems > 1 && d.atomic_op == amdgpu::AtomicOp::NONE && !vbuffer_is_format_op(inst.op);
  d.element_lane_masks.clear();
  if (per_component)
    d.element_lane_masks.assign(d.num_elems, exec);
  d.lane_mask = 0;
  std::optional<amdgpu::RegisterAccess::VgprReadRegion> voffset_region;
  if (inst.offen)
    voffset_region.emplace(regs.read_vgpr_region(wf.vgpr_alloc().base + inst.vaddr, 1, exec));
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t voffset = 0;
    if (inst.offen) {
      voffset = voffset_region->lane(0, lane);
    }
    uint32_t offset_part = amdgpu::addr_calc::buffer_offset_part(voffset, ioff);
    bool any_in_bounds = true;
    if (oob_select == 3) {
      any_in_bounds = false;
      const uint32_t components = per_component ? d.num_elems : 1;
      const uint32_t payload = per_component ? d.elem_size : d.elem_size * d.num_elems;
      const uint64_t reduced_num_records =
          soffset_val < num_records ? num_records - soffset_val : 0;
      for (uint32_t elem = 0; elem < components; ++elem) {
        const uint64_t component_offset = static_cast<uint64_t>(offset_part) + elem * d.elem_size;
        const bool oob = swizzle_enable != 0 && stride != 0
                             ? reduced_num_records == 0 ||
                                   buffer_range_exceeds(component_offset, payload, stride)
                             : buffer_range_exceeds(component_offset, payload, reduced_num_records);
        if (oob && per_component)
          d.element_lane_masks[elem] &= ~(uint64_t{1} << lane);
        any_in_bounds |= !oob;
      }
    }
    if (!any_in_bounds) {
      d.per_lane_addr[lane] = 0;
      continue;
    }
    d.per_lane_addr[lane] = base_addr + offset_part + soffset_val;
    d.lane_mask |= uint64_t{1} << lane;
  }
}

void ds_calculate_addresses(const VdsMachineInst &inst, amdgpu::Wavefront &wf,
                            amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::ds_calculate_addresses(inst, wf, d);
}

} // namespace rdna4
} // namespace rocjitsu
