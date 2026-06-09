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

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <bit>
#include <cstdint>

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
  const uint32_t wf_size = wf.wf_size();
  d.lane_mask = exec;
  d.exec_mask = exec;
  d.wf_size = wf_size;

  const uint64_t full_exec_mask = wf_size == 64 ? ~0ULL : ((1ULL << wf_size) - 1);
  const bool full_exec = exec == full_exec_mask;
  auto for_each_active_lane = [&](auto &&body) {
    if (full_exec) {
      for (uint32_t lane = 0; lane < wf_size; ++lane)
        body(lane);
      return;
    }

    uint64_t remaining = exec;
    while (remaining) {
      const uint32_t lane = std::countr_zero(remaining);
      remaining &= remaining - 1;
      body(lane);
    }
  };

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
    // addr = FLAT_SCRATCH + lane * scratch_lane_size + VGPR[lane] + saddr + offset.
    // On real hardware FLAT_SCRATCH is a dedicated register, not part of the
    // SGPR file. We store it in the wavefront's scratch_base_ member and also
    // mirror it to the flat_scratch_init user SGPRs for legacy compatibility.
    uint64_t scratch_base = wf.scratch_base();
    uint32_t saddr_val = 0;
    if (inst.saddr != 0x7F) {
      uint32_t sb = wf.sgpr_alloc().base + inst.saddr;
      saddr_val = cu.read_sgpr(sb);
    }
    uint32_t lane_stride = wf.scratch_lane_size();
    bool has_vaddr = true;
    if constexpr (requires { inst.sve; })
      has_vaddr = (inst.sve == 1);
    else if (inst.seg == 1)
      has_vaddr = (inst.lds == 1);
    uint32_t vbase = wf.vgpr_alloc().base + inst.addr;
    const uint32_t *vaddr_lanes =
        has_vaddr && !cu.plugin_hooks_enabled() ? cu.vgpr_lanes32(vbase) : nullptr;
    for_each_active_lane([&](uint32_t lane) {
      uint32_t vaddr = 0;
      if (has_vaddr) {
        vaddr = vaddr_lanes ? vaddr_lanes[lane] : cu.read_vgpr(vbase, lane);
      }
      d.per_lane_addr[lane] =
          scratch_base + static_cast<uint64_t>(lane) * lane_stride + vaddr + saddr_val + offset;
    });
  } else if (inst.seg == 2) {
    // GLOBAL: saddr (64-bit SGPR pair) + VGPR (32-bit) + offset,
    //         or VGPR pair (64-bit) + offset when saddr==0x7F.
    uint64_t saddr_val = 0;
    if (inst.saddr != 0x7F) {
      uint32_t sb = wf.sgpr_alloc().base + inst.saddr;
      saddr_val = (static_cast<uint64_t>(cu.read_sgpr(sb + 1)) << 32) | cu.read_sgpr(sb);
    }
    uint32_t vbase = wf.vgpr_alloc().base + inst.addr;
    const bool notify_vgpr_reads = cu.plugin_hooks_enabled();
    const uint32_t *lo_lanes = notify_vgpr_reads ? nullptr : cu.vgpr_lanes32(vbase);
    const uint32_t *hi_lanes =
        (!notify_vgpr_reads && inst.saddr == 0x7F) ? cu.vgpr_lanes32(vbase + 1) : nullptr;
    if (inst.saddr != 0x7F) {
      for_each_active_lane([&](uint32_t lane) {
        uint32_t raw = lo_lanes ? lo_lanes[lane] : cu.read_vgpr(vbase, lane);
        uint64_t vaddr = static_cast<uint64_t>(
            static_cast<int64_t>(static_cast<int32_t>(raw))); // sign-extended 32-bit offset
        d.per_lane_addr[lane] = saddr_val + vaddr + offset;
      });
    } else {
      for_each_active_lane([&](uint32_t lane) {
        uint32_t lo = lo_lanes ? lo_lanes[lane] : cu.read_vgpr(vbase, lane);
        uint32_t hi = hi_lanes ? hi_lanes[lane] : cu.read_vgpr(vbase + 1, lane);
        uint64_t vaddr = (static_cast<uint64_t>(hi) << 32) | lo; // 64-bit VGPR pair
        d.per_lane_addr[lane] = saddr_val + vaddr + offset;
      });
    }
  } else {
    // FLAT: 64-bit VGPR pair + unsigned 12-bit offset.
    // Real hardware checks the address against private/shared apertures and
    // routes accordingly. We perform the same conversion here so that private
    // (scratch) accesses reach the mapped scratch buffer instead of the
    // unmapped aperture VA range.
    uint32_t priv_hi = static_cast<uint32_t>(wf.private_aperture_base() >> 32);
    uint64_t scratch_base = wf.scratch_base();
    uint32_t lane_stride = wf.scratch_lane_size();
    uint32_t vbase = wf.vgpr_alloc().base + inst.addr;
    const bool notify_vgpr_reads = cu.plugin_hooks_enabled();
    const uint32_t *lo_lanes = notify_vgpr_reads ? nullptr : cu.vgpr_lanes32(vbase);
    const uint32_t *hi_lanes = notify_vgpr_reads ? nullptr : cu.vgpr_lanes32(vbase + 1);
    for_each_active_lane([&](uint32_t lane) {
      uint32_t lo = lo_lanes ? lo_lanes[lane] : cu.read_vgpr(vbase, lane);
      uint32_t hi = hi_lanes ? hi_lanes[lane] : cu.read_vgpr(vbase + 1, lane);
      uint64_t vaddr = (static_cast<uint64_t>(hi) << 32) | lo;
      uint64_t addr = vaddr + offset;
      if (priv_hi != 0 && static_cast<uint32_t>(addr >> 32) == priv_hi)
        addr = scratch_base + static_cast<uint64_t>(lane) * lane_stride + (addr & 0xFFFFFFFFULL);
      d.per_lane_addr[lane] = addr;
    });
  }
}

} // namespace addr_calc
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_FLAT_H_
