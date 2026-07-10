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

#include <cassert>
#include <cstdint>

namespace rocjitsu {
namespace amdgpu {
namespace addr_calc {

/// Decode the GFX12 flat-private address form used by LLVM-generated code.
///
/// The address is based on the wave scratch address, with the lane id encoded
/// in high address bits (bits 52:56 for wave32 and 51:56 for wave64). The
/// simulator stores scratch in contiguous per-lane slices, so translate the
/// encoded lane back to the corresponding slice before accessing memory.
inline bool decode_gfx12_flat_private_address(const amdgpu::Wavefront &wf, uint64_t addr,
                                              uint64_t *translated) {
  const uint32_t lane_stride = wf.scratch_lane_size();
  if (lane_stride == 0)
    return false;

  const uint32_t wf_size = wf.wf_size();
  assert(wf_size == 32 || wf_size == 64);
  const uint32_t lane_shift = wf_size == 64 ? 51 : 52;
  const uint64_t lane_mask = static_cast<uint64_t>(wf_size - 1) << lane_shift;
  const uint64_t scratch_base = wf.scratch_base();
  const uint64_t base_without_lane = scratch_base & ~lane_mask;
  const uint64_t addr_without_lane = addr & ~lane_mask;
  if (addr_without_lane < base_without_lane)
    return false;

  const uint64_t private_offset = addr_without_lane - base_without_lane;
  if (private_offset > 0xFFFF'FFFFULL)
    return false;

  if (translated != nullptr) {
    const uint32_t encoded_lane = static_cast<uint32_t>((addr & lane_mask) >> lane_shift);
    *translated = scratch_base + static_cast<uint64_t>(encoded_lane) * lane_stride + private_offset;
  }
  return true;
}

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
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      if (!(exec & (1ULL << lane)))
        continue;
      uint32_t vaddr = 0;
      if (has_vaddr) {
        uint32_t vbase = wf.vgpr_alloc().base + inst.addr;
        vaddr = cu.read_vgpr(vbase, lane);
      }
      d.per_lane_addr[lane] =
          scratch_base + static_cast<uint64_t>(lane) * lane_stride + vaddr + saddr_val + offset;
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
        vaddr = static_cast<uint64_t>(static_cast<int64_t>(
            static_cast<int32_t>(cu.read_vgpr(vbase, lane)))); // sign-extended 32-bit offset
      } else {
        vaddr = (static_cast<uint64_t>(cu.read_vgpr(vbase + 1, lane)) << 32) |
                cu.read_vgpr(vbase, lane); // 64-bit VGPR pair
      }
      d.per_lane_addr[lane] = saddr_val + vaddr + offset;
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
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      if (!(exec & (1ULL << lane)))
        continue;
      uint32_t vbase = wf.vgpr_alloc().base + inst.addr;
      uint64_t vaddr =
          (static_cast<uint64_t>(cu.read_vgpr(vbase + 1, lane)) << 32) | cu.read_vgpr(vbase, lane);
      uint64_t addr = vaddr + offset;
      if (priv_hi != 0 && static_cast<uint32_t>(addr >> 32) == priv_hi)
        addr = scratch_base + static_cast<uint64_t>(lane) * lane_stride + (addr & 0xFFFFFFFFULL);
      d.per_lane_addr[lane] = addr;
    }
  }
}

} // namespace addr_calc
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_FLAT_H_
