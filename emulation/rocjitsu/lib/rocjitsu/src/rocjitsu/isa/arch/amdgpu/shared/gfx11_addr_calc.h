// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu::gfx11 {

inline bool has_saddr(uint32_t saddr) {
  // LLVM prints saddr=0x7c as "off" for GFX11 global/flat memory ops.
  // Keep 0x7f as a no-base sentinel for older generated configs/tests.
  return saddr != 0x7C && saddr != 0x7F;
}

inline int64_t sign_extend(uint32_t value, uint32_t bits) {
  const uint32_t shift = 32u - bits;
  return static_cast<int64_t>(static_cast<int32_t>(value << shift) >> shift);
}

inline std::optional<uint32_t> read_smem_offset(uint32_t soffset, Wavefront &wf) {
  if (soffset == kModernNullSelector || soffset == 0x7F)
    return 0;
  if (soffset == kModernM0Selector)
    return wf.m0();
  return try_read_scalar_selector(wf, soffset);
}

template <typename Encoding>
std::optional<uint64_t> smem_calculate_address(const Encoding &inst, Wavefront &wf,
                                               ScalarMemState *state) {
  const uint32_t sbase_sel = inst.sbase * 2;
  auto base = try_read_scalar_selector64(wf, sbase_sel);
  if (!base)
    return std::nullopt;
  const int64_t immediate = sign_extend(inst.offset, 21);
  auto soffset = read_smem_offset(inst.soffset, wf);
  if (!soffset)
    return std::nullopt;
  const int64_t offset = immediate + *soffset;
  const uint64_t aligned_base = *base & ~uint64_t{3};
  if (addr_calc::smem_is_buffer_load_op(inst.op) || inst.op == 35 /* S_ATC_PROBE_BUFFER */) {
    // ISA 8.1.2 checks the immediate separately even if SOFFSET compensates for it.
    if (state)
      state->memviol = immediate < 0;
    return addr_calc::scalar_buffer_address(wf, sbase_sel, aligned_base, offset, state,
                                            /*align_mask=*/3, /*size_align_mask=*/0);
  }
  if (offset < 0)
    wf.report_undefined_behavior("negative combined scalar-memory offset");
  return aligned_base + (immediate & ~int64_t{3}) + (*soffset & ~uint32_t{3});
}

template <typename Encoding>
void flat_calculate_addresses(const Encoding &inst, Wavefront &wf, VectorMemState &d) {
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d.lane_mask = exec;
  d.exec_mask = exec;
  d.wf_size = wf.wf_size();
  const int64_t offset = inst.seg == 0 ? (inst.offset & 0xfffu) : sign_extend(inst.offset, 13);
  if (inst.seg == 0)
    d.flat_aperture_offset = static_cast<int32_t>(offset);

  if (inst.seg == 1) {
    RegisterAccess regs(cu);
    uint32_t saddr_val = 0;
    if (has_saddr(inst.saddr)) {
      const uint32_t sb_sel = inst.saddr;
      auto saddr = try_read_scalar_selector(wf, sb_sel);
      if (!saddr) {
        reject_vector_memory_access(d);
        return;
      }
      saddr_val = *saddr;
    }
    uint64_t scratch_base = wf.scratch_base();
    addr_calc::add_private_scratch_lanes(d, wf.wf_size(), exec);
    std::optional<RegisterAccess::VgprReadRegion> vaddr_region;
    if (inst.sve) {
      uint32_t vbase = wf.vgpr_alloc().base + inst.addr;
      vaddr_region.emplace(regs.read_vgpr_region(vbase, 1, exec));
    }
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      if (!(exec & (1ULL << lane)))
        continue;
      uint32_t vaddr = 0;
      if (inst.sve) {
        vaddr = vaddr_region->lane(0, lane);
      }
      const uint64_t private_offset = uint64_t{vaddr} + saddr_val + offset;
      d.per_lane_addr[lane] =
          scratch_lane_address(scratch_base, private_offset, lane, wf.wf_size());
    }
    return;
  }

  uint64_t saddr_val = 0;
  RegisterAccess regs(cu);
  if (has_saddr(inst.saddr)) {
    const uint32_t sb_sel = inst.saddr;
    auto saddr = try_read_scalar_selector64(wf, sb_sel);
    if (!saddr) {
      reject_vector_memory_access(d);
      return;
    }
    saddr_val = *saddr;
  }
  uint32_t vbase = wf.vgpr_alloc().base + inst.addr;
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
    if (inst.seg == 0)
      addr = addr_calc::capture_flat_private_address(wf, d, saddr_val + vaddr, lane,
                                                     d.flat_aperture_offset);
    d.per_lane_addr[lane] = addr;
  }
}

} // namespace rocjitsu::amdgpu::gfx11
