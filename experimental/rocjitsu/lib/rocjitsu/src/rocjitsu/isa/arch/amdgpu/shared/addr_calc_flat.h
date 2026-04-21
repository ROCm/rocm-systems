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
///
/// SCRATCH addresses use the per-wavefront scratch_base plus a 32-bit
/// VGPR offset (one VGPR, not a 64-bit pair).
///
/// GLOBAL addresses use an optional 64-bit SADDR base plus a 32-bit VGPR
/// offset, or a 64-bit VGPR pair when SADDR==0x7F (NULL).
///
/// FLAT addresses use a 64-bit VGPR pair with unsigned 12-bit offset.

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <array>
#include <cstdint>

namespace rocjitsu {
namespace amdgpu {
namespace addr_calc {

/// @brief Compute per-lane addresses for FLAT/GLOBAL/SCRATCH encoding.
///
/// Handles all three segments:
/// - FLAT (seg==0): 64-bit VGPR pair + unsigned 12-bit offset.
/// - SCRATCH (seg==1): scratch_base + 32-bit VGPR + saddr + signed 13-bit offset.
/// - GLOBAL (seg==2): 64-bit saddr + 32-bit VGPR + signed 13-bit offset,
///   or 64-bit VGPR pair when saddr==0x7F.
///
/// Requires: inst.seg, inst.saddr, inst.addr, inst.offset, inst.pad_12.
template <typename FlatInst>
void flat_calculate_addresses(const FlatInst &inst, amdgpu::Wavefront &wf,
                              std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  lane_mask = exec;

  // Compute signed 13-bit offset for GLOBAL/SCRATCH, unsigned 12-bit for FLAT.
  int64_t offset;
  if (inst.seg != 0) {
    uint32_t raw = inst.offset | (inst.pad_12 << 12);
    offset = static_cast<int64_t>(static_cast<int32_t>(raw << 19) >> 19);
  } else {
    offset = inst.offset & 0xFFF;
  }

  if (inst.seg == 1) {
    // SCRATCH: address = scratch_base + VGPR[lane] (32-bit) + saddr + offset.
    // VGPR is a single 32-bit register (not a pair).
    // saddr provides an optional SGPR offset (0x7F = no saddr).
    uint64_t scratch_base = wf.scratch_base();
    uint32_t saddr_val = 0;
    if (inst.saddr != 0x7F) {
      uint32_t sb = wf.sgpr_alloc().base + inst.saddr;
      saddr_val = cu.read_sgpr(sb);
    }
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      if (!(exec & (1ULL << lane)))
        continue;
      uint32_t vbase = wf.vgpr_alloc().base + inst.addr;
      uint32_t vaddr = cu.read_vgpr(vbase, lane);
      addrs[lane] = scratch_base + vaddr + saddr_val + offset;
    }
  } else if (inst.seg == 2) {
    // GLOBAL: saddr (64-bit SGPR pair) + VGPR (32-bit) + offset,
    //         or VGPR pair (64-bit) + offset when saddr==0x7F.
    uint64_t saddr_val = 0;
    if (inst.saddr != 0x7F) {
      uint32_t sb = wf.sgpr_alloc().base + inst.saddr;
      saddr_val = (static_cast<uint64_t>(cu.read_sgpr(sb + 1)) << 32) | cu.read_sgpr(sb);
    }
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      if (!(exec & (1ULL << lane)))
        continue;
      uint32_t vbase = wf.vgpr_alloc().base + inst.addr;
      uint64_t vaddr;
      if (inst.saddr != 0x7F) {
        vaddr = cu.read_vgpr(vbase, lane); // 32-bit VGPR offset
      } else {
        vaddr = (static_cast<uint64_t>(cu.read_vgpr(vbase + 1, lane)) << 32) |
                cu.read_vgpr(vbase, lane); // 64-bit VGPR pair
      }
      addrs[lane] = saddr_val + vaddr + offset;
    }
  } else {
    // FLAT: 64-bit VGPR pair + unsigned 12-bit offset.
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      if (!(exec & (1ULL << lane)))
        continue;
      uint32_t vbase = wf.vgpr_alloc().base + inst.addr;
      uint64_t vaddr =
          (static_cast<uint64_t>(cu.read_vgpr(vbase + 1, lane)) << 32) | cu.read_vgpr(vbase, lane);
      addrs[lane] = vaddr + offset;
    }
  }
}

} // namespace addr_calc
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_FLAT_H_
