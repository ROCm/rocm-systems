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

#include "rocjitsu/isa/arch/amdgpu/shared/flat_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cassert>
#include <cstdint>
#include <optional>

namespace rocjitsu {
namespace amdgpu {
namespace addr_calc {

// Before CDNA3, bit 13 redirects the destination to LDS. GFX10 uses NULL
// (125) for a vector address; GFX10.3 additionally uses 127 for no registers.
enum class ScratchAddressMode { VectorEnable, LegacyCdna, Rdna1, Rdna2 };

/// @brief Compute per-lane addresses for FLAT/GLOBAL/SCRATCH encoding.
///
/// @details Populates d.per_lane_addr, d.lane_mask, and d.exec_mask.
/// Handles all three segments:
/// - FLAT (seg==0): 64-bit VGPR pair + unsigned 12-bit offset.
/// - SCRATCH (seg==1): scratch_base + 32-bit VGPR + saddr + signed 12-bit (RDNA) or 13-bit (CDNA)
/// offset.
/// - GLOBAL (seg==2): 64-bit saddr + unsigned 32-bit VGPR offset + signed 13-bit offset,
///   or 64-bit VGPR pair when saddr==0x7F.
///
/// Requires: inst.seg, inst.saddr, inst.addr, inst.offset, inst.pad_12.
template <ScratchAddressMode mode = ScratchAddressMode::VectorEnable, typename FlatInst>
void flat_calculate_addresses(const FlatInst &inst, amdgpu::Wavefront &wf, VectorMemState &d) {
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  RegisterAccess regs(cu);
  d.lane_mask = exec;
  d.exec_mask = exec;
  d.wf_size = wf.wf_size();

  // GLOBAL/SCRATCH immediates are signed; CDNA FLAT uses an unsigned offset.
  int64_t offset;
  constexpr bool rdna = mode == ScratchAddressMode::Rdna1 || mode == ScratchAddressMode::Rdna2;
  if constexpr (rdna) {
    // RDNA wrappers use this helper only for SCRATCH. Their GLOBAL/FLAT
    // paths have different vector-offset and aperture rules.
    if (inst.seg != 1) {
      assert(false && "RDNA shared address calculation requires SCRATCH");
      reject_vector_memory_access(d);
      return;
    }
    offset = static_cast<int32_t>(inst.offset << 20) >> 20;
  } else if (inst.seg != 0) {
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
    // SCRATCH: wave-private memory with a dword-interleaved backing layout.
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
    if (inst.saddr != 0x7F && (!rdna || inst.saddr != 0x7D)) {
      const uint32_t sb_sel = inst.saddr;
      auto saddr = amdgpu::try_read_scalar_selector(wf, sb_sel);
      if (!saddr) {
        reject_vector_memory_access(d);
        return;
      }
      if constexpr (rdna)
        saddr_val = *saddr;
      else
        // Preserve CDNA's existing arithmetic for high-bit scalar offsets.
        // LLVM avoids these inputs before GFX12; changing their behavior
        // requires separate hardware qualification.
        saddr_val = static_cast<int32_t>(*saddr);
    }
    bool has_vaddr = true;
    if constexpr (mode == ScratchAddressMode::LegacyCdna) {
      has_vaddr = inst.saddr == 0x7F;
    } else if constexpr (mode == ScratchAddressMode::Rdna1) {
      has_vaddr = inst.saddr == 0x7D || inst.saddr == 0x7F;
    } else if constexpr (mode == ScratchAddressMode::Rdna2) {
      has_vaddr = inst.saddr == 0x7D;
    } else if constexpr (requires { inst.sve; }) {
      has_vaddr = inst.sve == 1;
    } else {
      has_vaddr = inst.lds == 1;
    }
    uint32_t vbase = wf.vgpr_alloc().base + inst.addr;
    std::optional<RegisterAccess::VgprReadRegion> vaddr_region;
    if (has_vaddr)
      vaddr_region.emplace(regs.read_vgpr_region(vbase, 1, exec));
    d.scratch_swizzle = true;
    d.requires_scratch_backing = true;
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
    // GLOBAL: saddr (64-bit SGPR pair) + unsigned VGPR offset (32-bit) + offset,
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
        vaddr = amdgpu::flat_vector_offset(vaddr_region.lane(0, lane), /*signed_offset=*/false);
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
    uint32_t vbase = wf.vgpr_alloc().base + inst.addr;
    auto vaddr_region = regs.read_vgpr_region(vbase, 2, exec);
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      if (!(exec & (1ULL << lane)))
        continue;
      uint64_t vaddr = vaddr_region.lane64(0, lane);
      uint64_t addr = vaddr + offset;
      const auto translated =
          translate_flat_address(wf, addr, lane, FlatPrivateLayout::Interleaved);
      if (translated.private_address) {
        addr = translated.value;
        d.scratch_swizzle = true;
        d.requires_scratch_backing = true;
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
