// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_SCALAR_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_SCALAR_H_

/// @file Shared address calculation for SMEM and DS instructions.
///
/// SMEM is scalar memory (constant cache / kernarg loads).
/// DS is local data share (LDS) operations.
///
/// These functions are templated on the machine instruction type so they work
/// with any ISA family whose encoding struct exposes the required field names.

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/log.h"

#include <array>
#include <cstdint>

namespace rocjitsu {
namespace amdgpu {
namespace addr_calc {

/// @brief Compute scalar address for SMEM encoding.
///
/// Requires: inst.sbase, inst.soffset_en, inst.soffset, inst.imm, inst.offset.
template <typename SmemInst>
uint64_t smem_calculate_address(const SmemInst &inst, amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t sbase = wf.sgpr_alloc().base + inst.sbase * 2;
  uint64_t base = (static_cast<uint64_t>(cu.read_sgpr(sbase + 1)) << 32) | cu.read_sgpr(sbase);
  uint64_t off = 0;
  if (inst.soffset_en)
    off += cu.read_sgpr(wf.sgpr_alloc().base + inst.soffset);
  if (inst.imm)
    off += static_cast<int64_t>(static_cast<int32_t>(inst.offset << 11) >> 11);
  uint64_t addr = base + off;
  util::Logger::vm([&](auto &os) {
    static thread_local uint64_t smem_count = 0;
    if (++smem_count <= 12 || (smem_count % 240) == 0)
      os << std::format("SMEM #{} base={:#x} off={} imm={} soff_en={} addr={:#x} raw_off={}",
                        smem_count, base, static_cast<int64_t>(off), inst.imm, inst.soffset_en,
                        addr, inst.offset);
  });
  return addr;
}

/// @brief Compute per-lane addresses for DS encoding.
///
/// @details Populates d.per_lane_addr, d.lane_mask, and d.exec_mask.
/// The DS encoding splits the 16-bit offset into two 8-bit fields (offset0
/// and offset1). For non-dual operations (ds_write_b32, ds_read_b32, etc.),
/// these form a single 16-bit byte offset: (offset1 << 8) | offset0.
template <typename DsInst>
void ds_calculate_addresses(const DsInst &inst, amdgpu::Wavefront &wf, VectorMemState &d) {
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d.lane_mask = exec;
  d.exec_mask = exec;
  d.wf_size = wf.wf_size();
  d.wg_id = wf.wg_id();
  d.wf_id = wf.wf_id();
  d.cu_path = wf.cu().full_path();
  uint32_t offset = (static_cast<uint32_t>(inst.offset1) << 8) | inst.offset0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    d.per_lane_addr[lane] =
        cu.read_vgpr(wf.vgpr_alloc().base + inst.addr, lane) + offset + wf.lds_base();
  }
  util::Logger::vm([&](auto &os) {
    static uint64_t ds_addr_count = 0;
    if (++ds_addr_count > 100)
      return;
    os << std::format("DS addr: {} wg[{}] wf[{}] v{}+{:#x} lds_base={} is_load={}",
                      wf.cu().full_path(), wf.wg_id(), wf.wf_id(), inst.addr, offset, wf.lds_base(),
                      d.is_load);
    for (uint32_t ln = 0; ln < wf.wf_size(); ++ln) {
      if (!(d.lane_mask & (1ULL << ln)))
        continue;
      os << std::format(" L{}:{:#x}", ln, d.per_lane_addr[ln]);
    }
  });
}

} // namespace addr_calc
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_SCALAR_H_
