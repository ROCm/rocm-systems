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

} // namespace

std::optional<uint64_t> smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf,
                                               amdgpu::ScalarMemState *state) {
  // GFX12 SMEM: sbase is an aligned SGPR pair, ioffset is a 24-bit signed immediate,
  // soffset is an SGPR/M0/null selector from rdna4/operand_types.h.
  const uint32_t sbase_sel = inst.sbase * 2;
  // Byte and halfword scalar loads retain their sub-dword address bits.
  const uint64_t align_mask = (state && state->elem_size < 4 ? state->elem_size : 4) - 1;
  auto base = amdgpu::try_read_scalar_selector64(wf, sbase_sel);
  if (!base)
    return std::nullopt;
  int64_t off = static_cast<int64_t>(static_cast<int32_t>(inst.ioffset << 8) >> 8);
  auto soffset = read_smem_offset(inst.soffset, wf);
  if (!soffset)
    return std::nullopt;
  *base &= ~align_mask;
  off = (off & ~static_cast<int64_t>(align_mask)) + (*soffset & ~align_mask);
  if (amdgpu::addr_calc::gfx12_smem_is_buffer_load_op(inst.op)) {
    return amdgpu::addr_calc::scalar_buffer_address(wf, sbase_sel, *base, off, state, align_mask,
                                                    align_mask);
  }
  return (*base + off) & ~align_mask;
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
  // GFX12 VBUFFER uses an unscaled SGPR selector and signed 24-bit IOFFSET.
  init_vector_mem_state(wf, d);
  const uint32_t sb_sel = inst.rsrc;
  if (!amdgpu::addr_calc::buffer_resource_range_is_backed(wf, sb_sel)) {
    amdgpu::reject_vector_memory_access(d);
    return;
  }
  const uint32_t srd1 = amdgpu::read_scalar_selector(wf, sb_sel + 1);
  const uint32_t records = amdgpu::read_scalar_selector(wf, sb_sel + 2);
  const uint32_t srd3 = amdgpu::read_scalar_selector(wf, sb_sel + 3);
  const uint64_t base_addr =
      (uint64_t{srd1 & 0xffff} << 32) | amdgpu::read_scalar_selector(wf, sb_sel);
  constexpr uint32_t scales[] = {1, 4, 8, 32};
  const uint32_t stride = ((srd1 >> 16) & 0x3fff) * scales[(srd3 >> 18) & 3];
  auto soffset = read_optional_sreg_m0(inst.soffset, wf);
  if (!soffset) {
    amdgpu::reject_vector_memory_access(d);
    return;
  }
  const int64_t ioff = static_cast<int32_t>(inst.ioffset << 8) >> 8;
  amdgpu::RegisterAccess regs(wf);
  std::optional<amdgpu::RegisterAccess::VgprReadRegion> vaddr;
  if (inst.idxen || inst.offen)
    vaddr.emplace(regs.read_vgpr_region(wf.vgpr_alloc().base + inst.vaddr,
                                        inst.idxen && inst.offen ? 2 : 1, d.exec_mask));
  amdgpu::addr_calc::rdna_buffer_apply_swizzle(d, srd1, srd3, inst.idxen != 0, d.exec_mask,
                                               base_addr, *soffset);
  const bool per_element =
      d.num_elems > 1 && !d.buffer_components && d.atomic_op == amdgpu::AtomicOp::NONE;
  d.element_lane_masks.clear();
  if (per_element)
    d.element_lane_masks.assign(d.num_elems, d.exec_mask);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(d.exec_mask & (1ULL << lane)))
      continue;
    const uint32_t index = inst.idxen ? vaddr->lane(0, lane) : 0;
    const uint32_t voffset = inst.offen ? vaddr->lane(inst.idxen ? 1 : 0, lane) : 0;
    const uint32_t offset = amdgpu::addr_calc::buffer_offset_part(voffset, ioff);
    const auto address = amdgpu::addr_calc::rdna_buffer_address(srd1, srd3, stride, inst.idxen != 0,
                                                                index, offset, *soffset, lane);
    if (!amdgpu::addr_calc::rdna_buffer_range_check_lane(d, lane, (srd3 >> 28) & 3, stride, records,
                                                         address.index, offset, *soffset,
                                                         address.swizzled, per_element)) {
      d.lane_mask &= ~(uint64_t{1} << lane);
      d.per_lane_addr[lane] = 0;
    } else {
      d.per_lane_addr[lane] = amdgpu::addr_calc::buffer_virtual_address(base_addr + address.offset);
    }
  }
}

void ds_calculate_addresses(const VdsMachineInst &inst, amdgpu::Wavefront &wf,
                            amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::ds_calculate_addresses(inst, wf, d);
}

} // namespace rdna4
} // namespace rocjitsu
