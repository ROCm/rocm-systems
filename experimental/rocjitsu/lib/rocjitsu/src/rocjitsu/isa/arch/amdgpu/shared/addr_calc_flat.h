// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_FLAT_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_FLAT_H_

/// @file Shared FLAT/GLOBAL/SCRATCH address calculation.
///
/// Templated on the FlatMachineInst type to work across ISA generations that
/// share the same field names.  CDNA3/4 FlatMachineInst fields (seg, saddr,
/// addr, offset, pad_12) are confirmed identical.
///
/// CDNA3/4 use a dedicated hardware register for FLAT_SCRATCH base; the
/// scratch offset is computed from the wavefront's hardware scratch base.
/// CDNA1/2 and RDNA compute FLAT_SCRATCH from an SGPR pair — those ISAs
/// should use their own flat_calculate_addresses or a separate template
/// specialization.

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <array>
#include <cstdint>

namespace rocjitsu {
namespace amdgpu {
namespace addr_calc {

/// @brief Compute per-lane addresses for FLAT/GLOBAL/SCRATCH encoding (CDNA3/4 layout).
///
/// Handles flat (unsigned 12-bit offset), global/scratch (signed 13-bit
/// offset with optional SGPR base via saddr).
///
/// Requires: inst.seg, inst.saddr, inst.addr, inst.offset, inst.pad_12.
template <typename FlatInst>
void flat_calculate_addresses(const FlatInst &inst, amdgpu::Wavefront &wf,
                              std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  lane_mask = exec;

  int64_t offset;
  if (inst.seg != 0) {
    uint32_t raw = inst.offset | (inst.pad_12 << 12);
    offset = static_cast<int64_t>(static_cast<int32_t>(raw << 19) >> 19);
  } else {
    offset = inst.offset & 0xFFF;
  }

  uint64_t saddr_val = 0;
  if (inst.seg != 0 && inst.saddr != 0x7F) {
    uint32_t sb = wf.sgpr_alloc().base + inst.saddr;
    saddr_val = (static_cast<uint64_t>(cu.read_sgpr(sb + 1)) << 32) | cu.read_sgpr(sb);
  }

  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t vbase = wf.vgpr_alloc().base + inst.addr;
    uint64_t vaddr;
    if (inst.seg != 0 && inst.saddr != 0x7F) {
      vaddr = cu.read_vgpr(vbase, lane);
    } else {
      vaddr =
          (static_cast<uint64_t>(cu.read_vgpr(vbase + 1, lane)) << 32) | cu.read_vgpr(vbase, lane);
    }
    addrs[lane] = saddr_val + vaddr + offset;
  }
}

} // namespace addr_calc
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_FLAT_H_
