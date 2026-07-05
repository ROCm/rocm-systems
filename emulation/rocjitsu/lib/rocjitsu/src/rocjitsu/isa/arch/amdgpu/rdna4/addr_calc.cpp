// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cassert>
#include <cstdint>

namespace rocjitsu {
namespace rdna4 {

namespace {

bool has_saddr(uint32_t saddr) {
  // LLVM prints saddr=0x7c as "off" for GFX12 global/flat memory ops.
  // Keep 0x7f as a no-base sentinel for older generated configs/tests.
  return saddr != 0x7C && saddr != 0x7F;
}

uint32_t read_optional_sreg_m0(uint32_t reg, amdgpu::Wavefront &wf) {
  if (reg == OPR_SREG_M0_NULL)
    return 0;
  if (reg == OPR_SREG_M0_M0)
    return wf.m0();
  return wf.cu().read_sgpr(wf.sgpr_alloc().base + reg);
}

uint32_t read_smem_offset(uint32_t soffset, amdgpu::Wavefront &wf) {
  if (soffset == OPR_SMEM_OFFSET_NULL || soffset == 0x7F)
    return 0;
  if (soffset == OPR_SMEM_OFFSET_M0)
    return wf.m0();
  return wf.cu().read_sgpr(wf.sgpr_alloc().base + soffset);
}

void init_vector_mem_state(amdgpu::Wavefront &wf, amdgpu::VectorMemState &d) {
  uint64_t exec = wf.exec();
  d.lane_mask = exec;
  d.exec_mask = exec;
  d.wf_size = wf.wf_size();
  d.wg_id = wf.wg_id();
  d.wf_id = wf.wf_id();
  d.cu_path = wf.cu().full_path();
}

} // namespace

uint64_t smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf) {
  // GFX12 SMEM: sbase is an aligned SGPR pair, ioffset is a 24-bit signed immediate,
  // soffset is an SGPR/M0/null selector from rdna4/operand_types.h.
  auto &cu = wf.cu();
  uint32_t sbase = wf.sgpr_alloc().base + inst.sbase * 2;
  uint64_t base = (static_cast<uint64_t>(cu.read_sgpr(sbase + 1)) << 32) | cu.read_sgpr(sbase);
  int64_t off = static_cast<int64_t>(static_cast<int32_t>(inst.ioffset << 8) >> 8);
  off += read_smem_offset(inst.soffset, wf);
  return (base + off) & ~0x3ULL;
}

void flat_calculate_addresses(const VflatMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d) {
  // GFX12 VFLAT: 24-bit signed offset, optional SGPR base via saddr.
  auto &cu = wf.cu();
  init_vector_mem_state(wf, d);
  uint64_t exec = d.exec_mask;
  int64_t offset = static_cast<int64_t>(static_cast<int32_t>(inst.ioffset << 8) >> 8);
  uint64_t saddr_val = 0;
  if (has_saddr(inst.saddr)) {
    uint32_t sb = wf.sgpr_alloc().base + inst.saddr;
    saddr_val = (static_cast<uint64_t>(cu.read_sgpr(sb + 1)) << 32) | cu.read_sgpr(sb);
  }
  uint32_t priv_hi = static_cast<uint32_t>(wf.private_aperture_base() >> 32);
  uint64_t scratch_base = wf.scratch_base();
  uint32_t lane_stride = wf.scratch_lane_size();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t vbase = wf.vgpr_alloc().base + inst.vaddr;
    uint64_t vaddr;
    if (has_saddr(inst.saddr)) {
      vaddr = cu.read_vgpr(vbase, lane);
    } else {
      vaddr =
          (static_cast<uint64_t>(cu.read_vgpr(vbase + 1, lane)) << 32) | cu.read_vgpr(vbase, lane);
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
  uint64_t saddr_val = 0;
  if (has_saddr(inst.saddr)) {
    uint32_t sb = wf.sgpr_alloc().base + inst.saddr;
    saddr_val = (static_cast<uint64_t>(cu.read_sgpr(sb + 1)) << 32) | cu.read_sgpr(sb);
  }
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t vbase = wf.vgpr_alloc().base + inst.vaddr;
    uint64_t vaddr;
    if (has_saddr(inst.saddr)) {
      vaddr = cu.read_vgpr(vbase, lane);
    } else {
      vaddr =
          (static_cast<uint64_t>(cu.read_vgpr(vbase + 1, lane)) << 32) | cu.read_vgpr(vbase, lane);
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
  if (has_saddr(inst.saddr)) {
    uint32_t sb = wf.sgpr_alloc().base + inst.saddr;
    saddr_val = cu.read_sgpr(sb);
  }
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t vbase = wf.vgpr_alloc().base + inst.vaddr;
    uint32_t vaddr = inst.sve ? cu.read_vgpr(vbase, lane) : 0;
    d.per_lane_addr[lane] =
        scratch_base + static_cast<uint64_t>(lane) * lane_stride + vaddr + saddr_val + offset;
  }
}

void mubuf_calculate_addresses(const VbufferMachineInst &inst, amdgpu::Wavefront &wf,
                               amdgpu::VectorMemState &d) {
  // GFX12 VBUFFER: rsrc is the first SGPR in the 4-dword resource descriptor,
  // soffset is a 7-bit SGPR/null selector, and ioffset is a signed immediate.
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d.lane_mask = exec;
  d.exec_mask = exec;
  uint32_t sb = wf.sgpr_alloc().base + inst.rsrc;
  uint64_t base_addr =
      (static_cast<uint64_t>(cu.read_sgpr(sb + 1) & 0xFFFF) << 32) | cu.read_sgpr(sb);
  uint32_t soffset_val = read_optional_sreg_m0(inst.soffset, wf);
  int64_t ioff = static_cast<int64_t>(static_cast<int32_t>(inst.ioffset << 8) >> 8);
  assert(!inst.idxen && "Vbuffer idxen not yet supported");
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t voffset = 0;
    if (inst.offen) {
      voffset = cu.read_vgpr(wf.vgpr_alloc().base + inst.vaddr, lane);
    }
    uint32_t offset_part = amdgpu::addr_calc::buffer_offset_part(voffset, ioff);
    d.per_lane_addr[lane] = base_addr + offset_part + soffset_val;
  }
}

void ds_calculate_addresses(const VdsMachineInst &inst, amdgpu::Wavefront &wf,
                            amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::ds_calculate_addresses(inst, wf, d);
}

} // namespace rdna4
} // namespace rocjitsu
