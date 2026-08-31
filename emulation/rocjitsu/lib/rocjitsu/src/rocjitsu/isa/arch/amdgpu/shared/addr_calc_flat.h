// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_FLAT_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_FLAT_H_

/// @file Shared FLAT/GLOBAL/SCRATCH address calculation.
///
/// Templated on the FlatMachineInst type to work across ISA generations that
/// share the same field names. CDNA3/4 FlatMachineInst fields (seg, saddr,
/// addr, offset, pad_12) are confirmed identical.
///
/// Segment encoding: seg==0 → FLAT, seg==1 → SCRATCH, seg==2 → GLOBAL.

#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cstdint>
#include <optional>

namespace rocjitsu {
namespace amdgpu {
namespace addr_calc {

/// @brief Compute per-lane addresses for FLAT/GLOBAL/SCRATCH encoding.
///
/// @details Populates d.per_lane_addr, d.lane_mask, and d.exec_mask.
/// Handles all three segments:
/// - FLAT (seg==0): 64-bit VGPR pair + unsigned 12-bit offset.
/// - SCRATCH (seg==1): scratch_base + 32-bit VGPR + saddr + signed 13-bit offset.
/// - GLOBAL (seg==2): 64-bit saddr + 32-bit VGPR + signed 13-bit offset,
///   or 64-bit VGPR pair when saddr==0x7F.
///
/// Requires: inst.seg, inst.saddr, inst.addr, inst.offset, inst.pad_12.
template <typename FlatInst>
void flat_calculate_addresses(const FlatInst &inst, amdgpu::Wavefront &wf, VectorMemState &d) {
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  RegisterAccess regs(cu);
  d.lane_mask = exec;
  d.exec_mask = exec;
  d.wf_size = wf.wf_size();

  // Compute signed 13-bit offset for GLOBAL/SCRATCH, unsigned 12-bit for FLAT.
  int64_t offset;
  if (inst.seg != 0) {
    uint32_t raw;
    if constexpr (requires { inst.pad_12; })
      raw = inst.offset | (inst.pad_12 << 12);
    else
      raw = inst.offset;
    offset = static_cast<int64_t>(static_cast<int32_t>(raw << 19) >> 19);
  } else {
    offset = inst.offset & 0xFFF;
  }

  if (inst.seg == 1) {
    // SCRATCH: architected flat scratch (GFX940/CDNA4).
    // The hardware stores scratch in a dword-interleaved ("swizzled") layout:
    // a lane's private byte offset `off` maps to
    //   scratch_base + (off/4)*lane_count*4 + lane*4 + off%4
    // so consecutive dwords of one lane are lane_count*4 bytes apart. We
    // reproduce that exact layout (rather than lane-major) so it matches how
    // rocm-dbgapi reads private_swizzled memory (rocdbgapi memory.cpp), letting
    // ROCgdb resolve scratch-resident variables. The interleave granule is a
    // dword (rocdbgapi architecture.cpp: private_lane interleave_size =
    // sizeof(uint32_t)). FLAT_SCRATCH is a dedicated register held in the
    // wavefront's scratch_base_ member.
    constexpr uint32_t kScratchInterleave = sizeof(uint32_t);
    const uint32_t lane_count = wf.wf_size();
    uint64_t scratch_base = wf.scratch_base();
    int64_t saddr_val = 0;
    if (inst.saddr != 0x7F) {
      const uint32_t sb_sel = inst.saddr;
      auto saddr = amdgpu::try_read_scalar_selector(wf, sb_sel);
      if (!saddr) {
        reject_vector_memory_access(d);
        return;
      }
      saddr_val = static_cast<int32_t>(*saddr);
    }
    bool has_vaddr = true;
    if constexpr (requires { inst.sve; })
      has_vaddr = (inst.sve == 1);
    else if (inst.seg == 1)
      has_vaddr = (inst.lds == 1);
    uint32_t vbase = wf.vgpr_alloc().base + inst.addr;
    std::optional<RegisterAccess::VgprReadRegion> vaddr_region;
    if (has_vaddr)
      vaddr_region.emplace(regs.read_vgpr_region(vbase, 1, exec));
    d.scratch_swizzle = true;
    d.scratch_addr_stride = lane_count * kScratchInterleave;
    d.scratch_lane_mask = exec;
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      if (!(exec & (1ULL << lane)))
        continue;
      uint32_t vaddr = 0;
      if (has_vaddr)
        vaddr = vaddr_region->lane(0, lane);
      uint64_t priv_off = static_cast<uint64_t>(static_cast<int64_t>(vaddr) + saddr_val + offset);
      d.per_lane_addr[lane] =
          scratch_base + (priv_off / kScratchInterleave) * lane_count * kScratchInterleave +
          static_cast<uint64_t>(lane) * kScratchInterleave + (priv_off % kScratchInterleave);
    }
  } else if (inst.seg == 2) {
    // GLOBAL: saddr (64-bit SGPR pair) + VGPR (32-bit) + offset,
    //         or VGPR pair (64-bit) + offset when saddr==0x7F.
    uint64_t saddr_val = 0;
    if (inst.saddr != 0x7F) {
      const uint32_t sb_sel = inst.saddr;
      auto saddr = amdgpu::try_read_scalar_selector64(wf, sb_sel);
      if (!saddr) {
        reject_vector_memory_access(d);
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
        vaddr = static_cast<uint64_t>(static_cast<int64_t>(
            static_cast<int32_t>(vaddr_region.lane(0, lane)))); // sign-extended 32-bit offset
      } else {
        vaddr = vaddr_region.lane64(0, lane); // 64-bit VGPR pair
      }
      d.per_lane_addr[lane] = saddr_val + vaddr + offset;
    }
  } else {
    // FLAT: 64-bit VGPR pair + unsigned 12-bit offset.
    // Real hardware checks the address against private/shared apertures and
    // routes accordingly. We perform the same conversion here so that private
    // (scratch) accesses reach the mapped scratch buffer instead of the
    // unmapped aperture VA range, using the same dword-interleaved swizzle as
    // dedicated SCRATCH ops so the private layout is uniform (and readable by
    // rocm-dbgapi).
    constexpr uint32_t kScratchInterleave = sizeof(uint32_t);
    const uint32_t lane_count = wf.wf_size();
    uint32_t priv_hi = static_cast<uint32_t>(wf.private_aperture_base() >> 32);
    uint64_t scratch_base = wf.scratch_base();
    uint32_t vbase = wf.vgpr_alloc().base + inst.addr;
    auto vaddr_region = regs.read_vgpr_region(vbase, 2, exec);
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      if (!(exec & (1ULL << lane)))
        continue;
      uint64_t vaddr = vaddr_region.lane64(0, lane);
      uint64_t addr = vaddr + offset;
      if (priv_hi != 0 && static_cast<uint32_t>(addr >> 32) == priv_hi) {
        uint64_t priv_off = addr & 0xFFFFFFFFULL;
        addr = scratch_base + (priv_off / kScratchInterleave) * lane_count * kScratchInterleave +
               static_cast<uint64_t>(lane) * kScratchInterleave + (priv_off % kScratchInterleave);
        d.scratch_swizzle = true;
        d.scratch_addr_stride = lane_count * kScratchInterleave;
        d.scratch_lane_mask |= 1ULL << lane;
      }
      d.per_lane_addr[lane] = addr;
    }
  }
}

} // namespace addr_calc
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_FLAT_H_
