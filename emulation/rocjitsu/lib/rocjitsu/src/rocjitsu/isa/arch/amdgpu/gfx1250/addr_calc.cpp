// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/gfx1250/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/operand.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/operand_types.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/bit.h"
#include "util/except.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

namespace rocjitsu::gfx1250 {
namespace {

uint32_t scaled_vaddr_factor(const amdgpu::VectorMemState &d) {
  // LLVM folds gfx1250 scale_offset when the scale matches the full memory
  // access size. The encoded immediate offset remains a byte offset.
  assert(d.elem_size != 0 && d.num_elems != 0);
  return d.elem_size * d.num_elems;
}

bool has_saddr(uint32_t saddr) { return saddr != OPR_SREG_NULL; }

bool has_smem_offset(uint32_t soffset) { return soffset != OPR_SMEM_OFFSET_NULL; }

uint32_t read_sreg_m0_operand(amdgpu::Wavefront &wf, uint32_t operand) {
  auto &cu = wf.cu();
  uint32_t base = wf.sgpr_alloc().base;
  if (operand <= 105)
    return cu.read_sgpr(base + operand);
  if (operand == 106)
    return static_cast<uint32_t>(wf.vcc());
  if (operand == 107)
    return static_cast<uint32_t>(wf.vcc() >> 32);
  if (operand >= 108 && operand <= 123) {
    // CommandProcessor aliases TTMP selectors into the wavefront SGPR slice.
    assert(operand < cu.config().sgprs_per_wf);
    return cu.read_sgpr(base + operand);
  }
  if (operand == 124)
    return 0;
  if (operand == 125)
    return wf.m0();
  throw util::UnimplementedInst("unsupported gfx1250 scalar memory offset operand");
}

uint64_t read_sreg64_operand(amdgpu::Wavefront &wf, uint32_t operand) {
  return (static_cast<uint64_t>(read_sreg_m0_operand(wf, operand + 1)) << 32) |
         read_sreg_m0_operand(wf, operand);
}

uint32_t resolved_vgpr_base(const amdgpu::Wavefront &wf, uint32_t operand,
                            amdgpu::VgprMsbRole role) {
  return wf.vgpr_alloc().base +
         *Isa::resolved_vgpr_offset(wf, OperandType::OPR_VGPR, operand, role);
}

uint64_t buffer_base_addr(uint32_t srd0, uint32_t srd1) {
  return (static_cast<uint64_t>(srd1 & 0x01FF'FFFFu) << 32) | srd0;
}

uint64_t buffer_num_records(uint32_t srd1, uint32_t srd2, uint32_t srd3) {
  return ((static_cast<uint64_t>(srd3 & 0x3Fu) << 32) | srd2) << 7 | ((srd1 >> 25) & 0x7Fu);
}

uint32_t buffer_stride(uint32_t srd3) { return (srd3 >> 12) & 0xFFFFu; }

bool buffer_oob_raw(uint32_t srd3) { return (srd3 & 0x8000'0000u) != 0; }

void init_vector_mem_state(amdgpu::Wavefront &wf, amdgpu::VectorMemState &d) {
  uint64_t exec = wf.exec();
  d.lane_mask = exec;
  d.exec_mask = exec;
  d.wf_size = wf.wf_size();
  d.wg_id = wf.wg_id();
  d.wf_id = wf.wf_id();
  d.cu_path = wf.cu().full_path();
}

template <typename Inst>
void flat_global_calculate_addresses(const Inst &inst, amdgpu::Wavefront &wf,
                                     amdgpu::VectorMemState &d) {
  auto &cu = wf.cu();
  init_vector_mem_state(wf, d);
  uint64_t exec = d.exec_mask;
  int64_t offset = static_cast<int64_t>(static_cast<int32_t>(inst.ioffset << 8) >> 8);
  bool saddr_present = has_saddr(inst.saddr);
  uint64_t saddr_val = saddr_present ? read_sreg64_operand(wf, inst.saddr) : 0;
  uint32_t scale = saddr_present && inst.scale_offset ? scaled_vaddr_factor(d) : 1;
  uint32_t vbase = resolved_vgpr_base(wf, inst.vaddr, amdgpu::VgprMsbRole::Src0);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t vaddr;
    if (saddr_present) {
      vaddr = cu.read_vgpr(vbase, lane);
      vaddr *= scale;
    } else {
      vaddr =
          (static_cast<uint64_t>(cu.read_vgpr(vbase + 1, lane)) << 32) | cu.read_vgpr(vbase, lane);
    }
    d.per_lane_addr[lane] = saddr_val + vaddr + offset;
  }
}

} // namespace

uint64_t smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf,
                                uint32_t access_size_bytes) {
  auto &cu = wf.cu();
  assert(access_size_bytes != 0);
  uint32_t sbase = wf.sgpr_alloc().base + inst.sbase * 2;
  uint64_t base = (static_cast<uint64_t>(cu.read_sgpr(sbase + 1)) << 32) | cu.read_sgpr(sbase);
  int64_t off = static_cast<int64_t>(static_cast<int32_t>(inst.ioffset << 8) >> 8);
  uint32_t scale = inst.scale_offset ? access_size_bytes : 1;
  off *= scale;
  if (has_smem_offset(inst.soffset))
    off += static_cast<int64_t>(read_sreg_m0_operand(wf, inst.soffset)) * scale;
  uint64_t addr = base + off;
  assert(util::is_aligned(addr, std::min<uint64_t>(access_size_bytes, 4u)) &&
         "gfx1250 scalar memory address must satisfy access alignment");
  return addr;
}

void flat_calculate_addresses(const VflatMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d) {
  flat_global_calculate_addresses(inst, wf, d);
}

void flat_calculate_addresses(const VglobalMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d) {
  flat_global_calculate_addresses(inst, wf, d);
}

void flat_calculate_addresses(const VscratchMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d) {
  auto &cu = wf.cu();
  init_vector_mem_state(wf, d);
  uint64_t exec = d.exec_mask;
  int64_t offset = static_cast<int64_t>(static_cast<int32_t>(inst.ioffset << 8) >> 8);
  uint64_t scratch_base = wf.scratch_base();
  uint32_t saddr_val = 0;
  if (has_saddr(inst.saddr))
    saddr_val = read_sreg_m0_operand(wf, inst.saddr);
  uint32_t vbase = 0;
  uint32_t scale = 1;
  if (inst.sve) {
    vbase = resolved_vgpr_base(wf, inst.vaddr, amdgpu::VgprMsbRole::Src0);
    if (inst.scale_offset)
      scale = scaled_vaddr_factor(d);
  }
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t lane_base = scratch_base + static_cast<uint64_t>(lane) * wf.scratch_lane_size();
    uint32_t vaddr = 0;
    if (inst.sve) {
      vaddr = cu.read_vgpr(vbase, lane);
      vaddr *= scale;
    }
    d.per_lane_addr[lane] = lane_base + vaddr + saddr_val + offset;
  }
}

void mubuf_calculate_addresses(const VbufferMachineInst &inst, amdgpu::Wavefront &wf,
                               amdgpu::VectorMemState &d) {
  auto &cu = wf.cu();
  init_vector_mem_state(wf, d);
  uint64_t exec = d.exec_mask;
  uint32_t sb = wf.sgpr_alloc().base + inst.rsrc;
  uint32_t srd0 = cu.read_sgpr(sb);
  uint32_t srd1 = cu.read_sgpr(sb + 1);
  uint32_t srd2 = cu.read_sgpr(sb + 2);
  uint32_t srd3 = cu.read_sgpr(sb + 3);
  uint64_t base_addr = buffer_base_addr(srd0, srd1);
  uint64_t num_records = buffer_num_records(srd1, srd2, srd3);
  uint32_t stride = buffer_stride(srd3);
  bool oob_raw = buffer_oob_raw(srd3);
  uint32_t soffset_val = has_smem_offset(inst.soffset) ? read_sreg_m0_operand(wf, inst.soffset) : 0;
  int64_t ioff = static_cast<int64_t>(static_cast<int32_t>(inst.ioffset << 8) >> 8);
  uint32_t vbase = 0;
  if (inst.idxen || inst.offen)
    vbase = resolved_vgpr_base(wf, inst.vaddr, amdgpu::VgprMsbRole::Src0);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t index = 0;
    uint32_t voffset = 0;
    if (inst.idxen && inst.offen) {
      index = cu.read_vgpr(vbase, lane);
      voffset = cu.read_vgpr(vbase + 1, lane);
    } else if (inst.idxen) {
      index = cu.read_vgpr(vbase, lane);
    } else if (inst.offen) {
      voffset = cu.read_vgpr(vbase, lane);
    }
    int64_t total_offset = static_cast<int64_t>(static_cast<uint64_t>(index) * stride) +
                           static_cast<int64_t>(voffset) + ioff + soffset_val;
    const int64_t offset_part = static_cast<int64_t>(voffset) + ioff;
    bool oob = total_offset < 0 || offset_part < 0;
    if (!oob && num_records != 0) {
      if (oob_raw) {
        oob = static_cast<uint64_t>(offset_part) >= num_records;
      } else if (stride > 0) {
        oob = index >= num_records;
      } else {
        oob = static_cast<uint64_t>(offset_part) >= num_records;
      }
    }
    if (oob) {
      d.lane_mask &= ~(1ULL << lane);
      d.per_lane_addr[lane] = 0;
    } else {
      d.per_lane_addr[lane] = (base_addr + static_cast<uint64_t>(total_offset)) & 0xFFFFFFFFFFFFULL;
    }
  }
}

void ds_calculate_addresses(const VdsMachineInst &inst, amdgpu::Wavefront &wf,
                            amdgpu::VectorMemState &d) {
  auto &cu = wf.cu();
  init_vector_mem_state(wf, d);
  uint64_t exec = d.exec_mask;
  uint32_t addr_base = resolved_vgpr_base(wf, inst.addr, amdgpu::VgprMsbRole::Src0);
  uint32_t offset = (static_cast<uint32_t>(inst.offset1) << 8) | inst.offset0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    d.per_lane_addr[lane] = cu.read_vgpr(addr_base, lane) + offset + wf.lds_base();
  }
}

} // namespace rocjitsu::gfx1250
