// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/cdna5/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/bit.h"
#include "util/except.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>

namespace rocjitsu::cdna5 {
namespace {

constexpr uint64_t kBufferOffsetMask = (uint64_t{1} << 45) - 1;

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
    return amdgpu::RegisterAccess(cu).read_sgpr(base + operand);
  if (operand == 106)
    return static_cast<uint32_t>(wf.vcc());
  if (operand == 107)
    return static_cast<uint32_t>(wf.vcc() >> 32);
  if (operand >= 108 && operand <= 123) {
    // CommandProcessor aliases TTMP selectors into the wavefront SGPR slice.
    assert(operand < cu.sgprs_per_wf());
    return amdgpu::RegisterAccess(cu).read_sgpr(base + operand);
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

void init_vector_mem_state(amdgpu::Wavefront &wf, amdgpu::VectorMemState &d) {
  uint64_t exec = wf.exec();
  d.lane_mask = exec;
  d.exec_mask = exec;
  d.wf_size = wf.wf_size();
  d.wg_id = wf.wg_id();
  d.wf_id = wf.wf_id();
  d.cu_path = wf.cu().full_path();
}

int64_t logical_buffer_offset(uint32_t index, uint32_t stride, uint32_t voffset, int32_t ioffset) {
  return static_cast<int64_t>(index) * stride + voffset + ioffset;
}

bool byte_range_exceeds(int64_t offset, uint32_t size, uint64_t bound) {
  if (offset < 0)
    return true;
  uint64_t unsigned_offset = static_cast<uint64_t>(offset);
  return unsigned_offset > bound || size > bound - unsigned_offset;
}

bool decode_flat_private_address(amdgpu::Wavefront &wf, uint64_t addr, uint64_t *translated) {
  uint32_t lane_stride = wf.scratch_lane_size();
  if (lane_stride == 0)
    return false;

  uint32_t wf_size = wf.wf_size();
  assert(wf_size == 32 || wf_size == 64);
  uint32_t lane_shift = wf_size == 64 ? 51 : 52;
  uint64_t lane_mask = static_cast<uint64_t>(wf_size - 1) << lane_shift;
  uint64_t scratch_base = wf.scratch_base();
  uint64_t base_without_lane = scratch_base & ~lane_mask;
  uint64_t addr_without_lane = addr & ~lane_mask;
  if (addr_without_lane < base_without_lane)
    return false;

  uint64_t private_offset = addr_without_lane - base_without_lane;
  if (private_offset > 0xFFFF'FFFFULL)
    return false;

  if (translated != nullptr) {
    constexpr uint32_t kScratchInterleave = sizeof(uint32_t);
    uint32_t encoded_lane = static_cast<uint32_t>((addr & lane_mask) >> lane_shift);
    *translated = scratch_base +
                  (private_offset / kScratchInterleave) * wf_size * kScratchInterleave +
                  static_cast<uint64_t>(encoded_lane) * kScratchInterleave +
                  (private_offset % kScratchInterleave);
  }
  return true;
}

template <typename Inst>
void flat_global_calculate_addresses(const Inst &inst, amdgpu::Wavefront &wf,
                                     amdgpu::VectorMemState &d, bool decode_flat_private) {
  auto &cu = wf.cu();
  init_vector_mem_state(wf, d);
  uint64_t exec = d.exec_mask;
  int64_t offset = static_cast<int64_t>(signed_ioffset(inst.ioffset));
  bool saddr_present = has_saddr(inst.saddr);
  uint64_t saddr_val = saddr_present ? read_sreg64_operand(wf, inst.saddr) : 0;
  uint32_t scale = saddr_present && inst.scale_offset ? scaled_vaddr_factor(d) : 1;
  uint32_t vbase = resolved_vgpr_base(wf, inst.vaddr, amdgpu::VgprMsbRole::Src0);
  amdgpu::RegisterAccess regs(cu);
  auto vaddr_region = regs.read_vgpr_region(vbase, saddr_present ? 1 : 2, exec);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t vaddr;
    if (saddr_present) {
      vaddr = vaddr_region.lane(0, lane);
      vaddr *= scale;
    } else {
      vaddr = vaddr_region.lane64(0, lane);
    }
    uint64_t addr = saddr_val + vaddr + offset;
    if (decode_flat_private) {
      uint64_t translated = 0;
      if (decode_flat_private_address(wf, addr, &translated)) {
        addr = translated;
        d.scratch_swizzle = true;
        d.scratch_addr_stride = wf.wf_size() * sizeof(uint32_t);
        d.scratch_lane_mask |= uint64_t{1} << lane;
      }
    } else {
      assert(!decode_flat_private_address(wf, addr, nullptr) &&
             "gfx1250 global memory address must not use flat private scratch encoding");
    }
    d.per_lane_addr[lane] = addr;
  }
}

} // namespace

BufferResource decode_buffer_resource(uint32_t srd0, uint32_t srd1, uint32_t srd2, uint32_t srd3) {
  static constexpr uint32_t stride_multipliers[] = {1, 4, 8, 32};
  BufferResource resource{};
  resource.base_address = (static_cast<uint64_t>(srd1 & 0x01FF'FFFFu) << 32) | srd0;
  resource.num_records =
      ((static_cast<uint64_t>(srd3 & 0x3Fu) << 32) | srd2) << 7 | ((srd1 >> 25) & 0x7Fu);
  resource.raw_stride = (srd3 >> 12) & 0x3FFFu;
  resource.stride_scale = static_cast<uint8_t>((srd3 >> 26) & 0x3u);
  resource.stride = resource.raw_stride * stride_multipliers[resource.stride_scale];
  resource.swizzle_enabled = ((srd3 >> 28) & 0x1u) != 0;
  resource.oob_select = ((srd3 >> 29) & 0x1u) != 0;
  resource.type = static_cast<uint8_t>((srd3 >> 30) & 0x3u);
  return resource;
}

uint64_t smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf,
                                uint32_t access_size_bytes) {
  assert(access_size_bytes != 0);
  const uint32_t sbase_sel = inst.sbase * 2;
  uint64_t base = amdgpu::read_scalar_selector64(wf, sbase_sel);
  int64_t off = static_cast<int64_t>(signed_ioffset(inst.ioffset));
  uint32_t scale = inst.scale_offset ? access_size_bytes : 1;
  if (has_smem_offset(inst.soffset))
    off += static_cast<int64_t>(read_sreg_m0_operand(wf, inst.soffset)) * scale;
  uint64_t addr = base + off;
  assert(util::is_aligned(addr, std::min<uint64_t>(access_size_bytes, 4u)) &&
         "gfx1250 scalar memory address must satisfy access alignment");
  return addr;
}

void flat_calculate_addresses(const VflatMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d) {
  flat_global_calculate_addresses(inst, wf, d, true);
}

void flat_calculate_addresses(const VglobalMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d) {
  flat_global_calculate_addresses(inst, wf, d, false);
}

uint32_t async_lds_lane_address(const VglobalMachineInst &inst, const amdgpu::Wavefront &wf,
                                uint32_t lds_operand, uint32_t access_size_bytes) {
  int64_t relative_addr = static_cast<int64_t>(lds_operand) + signed_ioffset(inst.ioffset);
  if (relative_addr < 0 || static_cast<uint64_t>(relative_addr) + access_size_bytes > wf.lds_size())
    return amdgpu::kInvalidLdsAddress;

  uint64_t absolute_addr = static_cast<uint64_t>(wf.lds_base()) + relative_addr;
  if (absolute_addr >= UINT32_MAX)
    return amdgpu::kInvalidLdsAddress;
  return static_cast<uint32_t>(absolute_addr);
}

void flat_calculate_addresses(const VscratchMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d) {
  auto &cu = wf.cu();
  init_vector_mem_state(wf, d);
  uint64_t exec = d.exec_mask;
  int64_t offset = static_cast<int64_t>(signed_ioffset(inst.ioffset));
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
  amdgpu::RegisterAccess regs(cu);
  std::optional<amdgpu::RegisterAccess::VgprReadRegion> vaddr_region;
  if (inst.sve)
    vaddr_region.emplace(regs.read_vgpr_region(vbase, 1, exec));
  constexpr uint32_t kScratchInterleave = sizeof(uint32_t);
  const uint32_t lane_count = wf.wf_size();
  d.scratch_swizzle = true;
  d.scratch_lane_mask = exec;
  d.scratch_addr_stride = lane_count * kScratchInterleave;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t vaddr = 0;
    if (inst.sve) {
      vaddr = vaddr_region->lane(0, lane);
      vaddr *= scale;
    }
    const uint64_t private_offset =
        static_cast<uint64_t>(static_cast<int64_t>(vaddr) + saddr_val + offset);
    d.per_lane_addr[lane] =
        scratch_base + (private_offset / kScratchInterleave) * lane_count * kScratchInterleave +
        static_cast<uint64_t>(lane) * kScratchInterleave + (private_offset % kScratchInterleave);
  }
}

void mubuf_calculate_addresses(const VbufferMachineInst &inst, amdgpu::Wavefront &wf,
                               amdgpu::VectorMemState &d) {
  auto &cu = wf.cu();
  init_vector_mem_state(wf, d);
  uint64_t exec = d.exec_mask;
  // Read through the scalar selector rather than the SGPR allocation: a
  // descriptor sourced from TTMPs lives in the trap-temporary file, and
  // read_sgpr() would fetch whatever the allocation holds at that index.
  const uint32_t sb_sel = inst.rsrc;
  uint32_t srd0 = amdgpu::read_scalar_selector(wf, sb_sel);
  uint32_t srd1 = amdgpu::read_scalar_selector(wf, sb_sel + 1);
  uint32_t srd2 = amdgpu::read_scalar_selector(wf, sb_sel + 2);
  uint32_t srd3 = amdgpu::read_scalar_selector(wf, sb_sel + 3);
  BufferResource resource = decode_buffer_resource(srd0, srd1, srd2, srd3);
  uint32_t soffset_val = has_smem_offset(inst.soffset) ? read_sreg_m0_operand(wf, inst.soffset) : 0;
  int32_t ioff = signed_ioffset(inst.ioffset);
  uint32_t vbase = 0;
  if (inst.idxen || inst.offen)
    vbase = resolved_vgpr_base(wf, inst.vaddr, amdgpu::VgprMsbRole::Src0);
  amdgpu::RegisterAccess regs(cu);
  std::optional<amdgpu::RegisterAccess::VgprReadRegion> vaddr_region;
  if (inst.idxen || inst.offen) {
    uint32_t reg_count = (inst.idxen && inst.offen) ? 2 : 1;
    vaddr_region.emplace(regs.read_vgpr_region(vbase, reg_count, exec));
  }
  assert(d.elem_size != 0 && d.num_elems != 0);
  d.element_lane_masks.clear();
  if (ioff < 0) {
    // CDNA5 requires VBUFFER IOFFSET to be non-negative. Suppress illegal
    // encodings defensively so their wrapped 45-bit address cannot reach L1.
    d.element_lane_masks.assign(d.num_elems, 0);
    d.lane_mask = 0;
    return;
  }
  if (!resource.swizzle_enabled)
    d.element_lane_masks.assign(d.num_elems, exec);
  d.lane_mask = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t index = 0;
    uint32_t voffset = 0;
    if (inst.idxen && inst.offen) {
      index = vaddr_region->lane(0, lane);
      voffset = vaddr_region->lane(1, lane);
    } else if (inst.idxen) {
      index = vaddr_region->lane(0, lane);
    } else if (inst.offen) {
      voffset = vaddr_region->lane(0, lane);
    }
    int64_t logical_offset = logical_buffer_offset(index, resource.stride, voffset, ioff);
    uint64_t buffer_offset = static_cast<uint64_t>(logical_offset) & kBufferOffsetMask;
    d.per_lane_addr[lane] = resource.base_address + soffset_val + buffer_offset;

    if (resource.swizzle_enabled) {
      // Swizzled address generation is not modeled yet. Preserve the existing
      // linear address behavior, but do not apply unswizzled bounds rules.
      d.lane_mask |= uint64_t{1} << lane;
      continue;
    }

    int64_t record_offset = static_cast<int64_t>(soffset_val) + voffset + ioff;
    int64_t total_offset = static_cast<int64_t>(soffset_val) + logical_offset;
    for (uint32_t elem = 0; elem < d.num_elems; ++elem) {
      int64_t component_offset = static_cast<int64_t>(elem) * d.elem_size;
      bool oob =
          byte_range_exceeds(total_offset + component_offset, d.elem_size, resource.num_records);
      if (resource.oob_select) {
        oob = oob ||
              byte_range_exceeds(record_offset + component_offset, d.elem_size, resource.stride);
      }
      if (oob)
        d.element_lane_masks[elem] &= ~(uint64_t{1} << lane);
      else
        d.lane_mask |= uint64_t{1} << lane;
    }
    if (!(d.lane_mask & (uint64_t{1} << lane)))
      d.per_lane_addr[lane] = 0;
  }
}

void ds_calculate_addresses(const VdsMachineInst &inst, amdgpu::Wavefront &wf,
                            amdgpu::VectorMemState &d) {
  auto &cu = wf.cu();
  init_vector_mem_state(wf, d);
  uint64_t exec = d.exec_mask;
  uint32_t addr_base = resolved_vgpr_base(wf, inst.addr, amdgpu::VgprMsbRole::Src0);
  uint32_t offset = (static_cast<uint32_t>(inst.offset1) << 8) | inst.offset0;
  amdgpu::RegisterAccess regs(cu);
  auto addr_region = regs.read_vgpr_region(addr_base, 1, exec);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    d.per_lane_addr[lane] = addr_region.lane(0, lane) + offset + wf.lds_base();
  }
}

} // namespace rocjitsu::cdna5
