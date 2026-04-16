// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_BUFFER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_BUFFER_H_

/// @file Shared address calculation for MUBUF and MTBUF (buffer) instructions.
///
/// These are vector memory operations that access global memory through a
/// buffer resource descriptor (SRD). Templated on the machine instruction
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
    static uint64_t mubuf_count = 0;
    if (base_addr == 0 || ++mubuf_count <= 40)
      os << std::format("MUBUF addr: srsrc=s[{}:{}] srd=[{:#x},{:#x},{:#x},{:#x}] base={:#x}"
                        " soff={:#x} offset={:#x} offen={} idxen={} vaddr=v{}"
                        " num_records={}",
                        inst.srsrc * 4, inst.srsrc * 4 + 3, srd0, srd1, srd2, srd3, base_addr,
                        soffset_val, inst.offset, inst.offen, inst.idxen, inst.vaddr, num_records);
  });
  // GFX9 OOB modes (srd[3] bit 31):
  //   0 = structured: OOB check uses voffset + inst_offset only (no soffset).
  //   1 = raw: OOB check uses voffset + soffset + inst_offset.
  bool oob_raw = (srd3 >> 31) & 1;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t voffset = 0;
    if (inst.offen)
      voffset = cu.read_vgpr(wf.vgpr_alloc().base + inst.vaddr, lane);
    if (inst.idxen)
      voffset = cu.read_vgpr(wf.vgpr_alloc().base + inst.vaddr, lane);
    uint64_t total_offset = static_cast<uint64_t>(voffset) + inst.offset + soffset_val;
    uint64_t oob_offset = oob_raw ? total_offset : static_cast<uint64_t>(voffset) + inst.offset;
    if (num_records == 0 || oob_offset >= num_records) {
      d.lane_mask &= ~(1ULL << lane);
      d.per_lane_addr[lane] = 0;
    } else {
      d.per_lane_addr[lane] = base_addr + total_offset;
    }
  }
  // Per-lane address trace: log the first 4 active lanes so we can verify
  // that each lane's voffset (and thus effective address) is correct.
  util::Logger::vm([&](auto &os) {
    static uint64_t lane_trace_count = 0;
    if (++lane_trace_count > 80)
      return;
    os << "MUBUF per-lane:";
    uint64_t rm = d.lane_mask;
    int cnt = 0;
    while (rm && cnt < 4) {
      uint32_t ln = std::countr_zero(rm);
      rm &= rm - 1;
      uint32_t voff = 0;
      if (inst.offen)
        voff = cu.read_vgpr(wf.vgpr_alloc().base + inst.vaddr, ln);
      os << std::format(" L{}:voff={:#x},addr={:#x}", ln, voff, d.per_lane_addr[ln]);
      ++cnt;
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
  assert(!inst.idxen && "Mtbuf idxen not yet supported");
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d.lane_mask = exec;
  d.exec_mask = exec;
  uint32_t sb = wf.sgpr_alloc().base + inst.srsrc * 4;
  uint64_t base_addr =
      (static_cast<uint64_t>(cu.read_sgpr(sb + 1) & 0xFFFF) << 32) | cu.read_sgpr(sb);
  uint32_t soffset_val =
      (inst.soffset == 0x80) ? 0u : cu.read_sgpr(wf.sgpr_alloc().base + inst.soffset);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t voffset = 0;
    if (inst.offen)
      voffset = cu.read_vgpr(wf.vgpr_alloc().base + inst.vaddr, lane);
    d.per_lane_addr[lane] = base_addr + voffset + inst.offset + soffset_val;
  }
}

} // namespace addr_calc
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_BUFFER_H_
