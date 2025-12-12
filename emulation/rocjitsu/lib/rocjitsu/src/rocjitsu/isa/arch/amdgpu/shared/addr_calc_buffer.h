// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_BUFFER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_BUFFER_H_

/// @file Shared address calculation for MUBUF and MTBUF (buffer) instructions.
///
/// @details These are vector memory operations that access global memory through
/// a buffer resource descriptor (SRD). Templated on the machine instruction
/// type so they work with any ISA family whose encoding struct exposes the
/// required field names.

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/log.h"

#include <bit>
#include <cstdint>

namespace rocjitsu {
namespace amdgpu {
namespace addr_calc {

constexpr uint32_t buffer_offset_part(uint32_t voffset, int64_t inst_offset) {
  // Hardware forms this sub-expression in 32-bit offset space before it is
  // widened and added to the descriptor base address.
  return voffset + static_cast<uint32_t>(inst_offset);
}

constexpr uint64_t buffer_total_offset(uint32_t index, uint32_t stride, uint32_t offset_part,
                                       uint32_t soffset) {
  return static_cast<uint64_t>(index) * stride + offset_part + soffset;
}

/// @brief Compute per-lane addresses for MUBUF encoding.
///
/// @details Populates d.per_lane_addr, d.lane_mask, and d.exec_mask.
/// Implements GFX9 buffer bounds checking: out-of-bounds lanes are
/// excluded from lane_mask (loads return 0, stores are dropped).
///
/// Requires: inst.srsrc, inst.soffset, inst.idxen, inst.offen, inst.vaddr,
///           inst.offset.
template <typename MubufInst>
void mubuf_calculate_addresses(const MubufInst &inst, amdgpu::Wavefront &wf, VectorMemState &d) {
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d.lane_mask = exec;
  d.exec_mask = exec;
  d.wf_size = wf.wf_size();
  d.wg_id = wf.wg_id();
  d.wf_id = wf.wf_id();
  d.cu_path = wf.cu().full_path();
  uint32_t sb = wf.sgpr_alloc().base + inst.srsrc * 4;
  uint32_t srd0 = cu.read_sgpr(sb);
  uint32_t srd1 = cu.read_sgpr(sb + 1);
  uint32_t srd2 = cu.read_sgpr(sb + 2);
  uint32_t srd3 = cu.read_sgpr(sb + 3);
  uint64_t base_addr = (static_cast<uint64_t>(srd1 & 0xFFFF) << 32) | srd0;
  // soffset field: 0-105 = SGPR index, 128 (0x80) = inline constant 0.
  uint32_t soffset_val =
      (inst.soffset == 0x80) ? 0u : cu.read_sgpr(wf.sgpr_alloc().base + inst.soffset);
  // GFX9 buffer bounds checking: OOB loads return 0, OOB stores are dropped.
  // Per ISA spec (structured mode, stride=0): num_records is the buffer size
  // in bytes. The OOB check uses voffset + inst_offset only — soffset is NOT
  // included in the bounds comparison, but IS added to the final address.
  uint32_t num_records = srd2;
  util::Logger::vm([&](auto &os) {
    uint32_t wgid = wf.wg_id();
    if (wgid == 0)
      os << std::format("{} wg[{}] wf[{}] MUBUF addr: srsrc=s[{}:{}]"
                        " srd=[{:#x},{:#x},{:#x},{:#x}] base={:#x}"
                        " soff={:#x} offset={:#x} offen={} idxen={} vaddr=v{}"
                        " num_records={}",
                        wf.cu().full_path(), wf.wg_id(), wf.wf_id(), inst.srsrc * 4,
                        inst.srsrc * 4 + 3, srd0, srd1, srd2, srd3, base_addr, soffset_val,
                        inst.offset, inst.offen, inst.idxen, inst.vaddr, num_records);
  });
  // GFX9 MUBUF address calculation per ISA Table 42 / Section 9.1.5.2:
  //
  // VGPR assignment (idxen × offen):
  //   0,0 → no VGPRs          0,1 → vaddr = offset
  //   1,0 → vaddr = index     1,1 → vaddr = index, vaddr+1 = offset
  //
  // Address = base + soffset + (index * stride) + voffset + inst_offset
  //
  // OOB modes (srd[3] bit 31):
  //   1 = raw:        OOB if (voffset + inst_offset) >= num_records
  //   0 = structured: OOB if stride > 0 ? (index >= num_records)
  //                              else    (voffset + inst_offset) >= num_records
  uint32_t stride = (srd1 >> 16) & 0x3FFF;
  bool oob_raw = (srd3 >> 31) & 1;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t vgpr_base = wf.vgpr_alloc().base + inst.vaddr;
    uint32_t index = 0;
    uint32_t voffset = 0;
    if (inst.idxen && inst.offen) {
      index = cu.read_vgpr(vgpr_base, lane);
      voffset = cu.read_vgpr(vgpr_base + 1, lane);
    } else if (inst.idxen) {
      index = cu.read_vgpr(vgpr_base, lane);
    } else if (inst.offen) {
      voffset = cu.read_vgpr(vgpr_base, lane);
    }
    uint32_t offset_part = buffer_offset_part(voffset, inst.offset);
    uint64_t total_offset = buffer_total_offset(index, stride, offset_part, soffset_val);
    // OOB check.
    bool oob;
    if (oob_raw) {
      oob = offset_part >= num_records;
    } else if (stride > 0) {
      oob = index >= num_records;
    } else {
      oob = offset_part >= num_records;
    }
    if (oob) {
      d.lane_mask &= ~(1ULL << lane);
      d.per_lane_addr[lane] = 0;
    } else {
      d.per_lane_addr[lane] = (base_addr + total_offset) & 0xFFFFFFFFFFFFULL;
    }
  }
  // Per-lane address trace: log the first 4 active lanes so we can verify
  // that each lane's voffset (and thus effective address) is correct.
  util::Logger::vm([&](auto &os) {
    static uint64_t pl_count = 0;
    if (wf.wg_id() != 0 && ++pl_count > 500)
      return;
    os << std::format("{} wg[{}] wf[{}] MUBUF per-lane: stride={} oob_raw={}", wf.cu().full_path(),
                      wf.wg_id(), wf.wf_id(), stride, oob_raw);
    for (uint32_t ln = 0; ln < wf.wf_size(); ++ln) {
      if (!(d.lane_mask & (1ULL << ln)))
        continue;
      os << std::format(" L{}:{:#x}", ln, d.per_lane_addr[ln]);
    }
    os << std::format(" exec={:#x} lane_mask={:#x}", exec, d.lane_mask);
  });
}

/// @brief Compute per-lane addresses for MTBUF encoding.
///
/// @details Populates d.per_lane_addr, d.lane_mask, and d.exec_mask.
///
/// Requires: inst.srsrc, inst.soffset, inst.idxen, inst.offen, inst.vaddr,
///           inst.offset.
template <typename MtbufInst>
void mtbuf_calculate_addresses(const MtbufInst &inst, amdgpu::Wavefront &wf, VectorMemState &d) {
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d.lane_mask = exec;
  d.exec_mask = exec;
  d.wf_size = wf.wf_size();
  d.wg_id = wf.wg_id();
  d.wf_id = wf.wf_id();
  d.cu_path = wf.cu().full_path();
  uint32_t sb = wf.sgpr_alloc().base + inst.srsrc * 4;
  uint32_t srd0 = cu.read_sgpr(sb);
  uint32_t srd1 = cu.read_sgpr(sb + 1);
  uint32_t srd2 = cu.read_sgpr(sb + 2);
  uint32_t srd3 = cu.read_sgpr(sb + 3);
  uint64_t base_addr = (static_cast<uint64_t>(srd1 & 0xFFFF) << 32) | srd0;
  uint32_t soffset_val =
      (inst.soffset == 0x80) ? 0u : cu.read_sgpr(wf.sgpr_alloc().base + inst.soffset);
  uint32_t num_records = srd2;
  uint32_t stride = (srd1 >> 16) & 0x3FFF;
  bool oob_raw = (srd3 >> 31) & 1;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t vgpr_base = wf.vgpr_alloc().base + inst.vaddr;
    uint32_t index = 0;
    uint32_t voffset = 0;
    if (inst.idxen && inst.offen) {
      index = cu.read_vgpr(vgpr_base, lane);
      voffset = cu.read_vgpr(vgpr_base + 1, lane);
    } else if (inst.idxen) {
      index = cu.read_vgpr(vgpr_base, lane);
    } else if (inst.offen) {
      voffset = cu.read_vgpr(vgpr_base, lane);
    }
    uint32_t offset_part = buffer_offset_part(voffset, inst.offset);
    uint64_t total_offset = buffer_total_offset(index, stride, offset_part, soffset_val);
    bool oob;
    if (oob_raw) {
      oob = offset_part >= num_records;
    } else if (stride > 0) {
      oob = index >= num_records;
    } else {
      oob = offset_part >= num_records;
    }
    if (oob) {
      d.lane_mask &= ~(1ULL << lane);
      d.per_lane_addr[lane] = 0;
    } else {
      d.per_lane_addr[lane] = (base_addr + total_offset) & 0xFFFFFFFFFFFFULL;
    }
  }
}

} // namespace addr_calc
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_BUFFER_H_
