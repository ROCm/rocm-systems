// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_SCALAR_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_SCALAR_H_

/// @file Shared address calculation for SMEM, MUBUF, MTBUF, DS instructions.
///
/// These functions are templated on the machine instruction type so they work
/// with any ISA family whose encoding struct exposes the required field names.
/// CDNA3/4 machine_inst types are confirmed compatible.  CDNA1/2 are compatible
/// for SMEM, DS, and the address-calculation subset of MUBUF/MTBUF (the
/// coherency fields differ but are not used by address calculation).

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/log.h"

#include <array>
#include <cassert>
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
  uint64_t addr = (base + off) & ~0x3ULL;
  // Trace: log SMEM address calculations for kernarg loads.
  util::Logger::vm([&](auto &os) {
    static thread_local uint64_t smem_count = 0;
    if (++smem_count <= 12 || (smem_count % 240) == 0)
      os << std::format("SMEM #{} base={:#x} off={} imm={} soff_en={} addr={:#x} raw_off={}",
                        smem_count, base, static_cast<int64_t>(off), inst.imm, inst.soffset_en,
                        addr, inst.offset);
  });
  return addr;
}

/// @brief Compute per-lane addresses for MUBUF encoding.
///
/// Requires: inst.srsrc, inst.soffset, inst.idxen, inst.offen, inst.vaddr,
///           inst.offset.
template <typename MubufInst>
void mubuf_calculate_addresses(const MubufInst &inst, amdgpu::Wavefront &wf,
                               std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  lane_mask = exec;
  uint32_t sb = wf.sgpr_alloc().base + inst.srsrc * 4;
  uint64_t base_addr =
      (static_cast<uint64_t>(cu.read_sgpr(sb + 1) & 0xFFFF) << 32) | cu.read_sgpr(sb);
  uint32_t soffset_val = cu.read_sgpr(wf.sgpr_alloc().base + inst.soffset);
  assert(!inst.idxen && "Mubuf idxen not yet supported");
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t voffset = 0;
    if (inst.offen)
      voffset = cu.read_vgpr(wf.vgpr_alloc().base + inst.vaddr, lane);
    addrs[lane] = base_addr + voffset + inst.offset + soffset_val;
  }
}

/// @brief Compute per-lane addresses for MTBUF encoding.
///
/// Requires: inst.srsrc, inst.soffset, inst.idxen, inst.offen, inst.vaddr,
///           inst.offset.
template <typename MtbufInst>
void mtbuf_calculate_addresses(const MtbufInst &inst, amdgpu::Wavefront &wf,
                               std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  assert(!inst.idxen && "Mtbuf idxen not yet supported");
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  lane_mask = exec;
  uint32_t sb = wf.sgpr_alloc().base + inst.srsrc * 4;
  uint64_t base_addr =
      (static_cast<uint64_t>(cu.read_sgpr(sb + 1) & 0xFFFF) << 32) | cu.read_sgpr(sb);
  uint32_t soffset_val = cu.read_sgpr(wf.sgpr_alloc().base + inst.soffset);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t voffset = 0;
    if (inst.offen)
      voffset = cu.read_vgpr(wf.vgpr_alloc().base + inst.vaddr, lane);
    addrs[lane] = base_addr + voffset + inst.offset + soffset_val;
  }
}

/// @brief Compute per-lane addresses for DS encoding.
///
/// Requires: inst.addr, inst.offset0.
template <typename DsInst>
void ds_calculate_addresses(const DsInst &inst, amdgpu::Wavefront &wf,
                            std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  lane_mask = exec;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    addrs[lane] = cu.read_vgpr(wf.vgpr_alloc().base + inst.addr, lane) + inst.offset0;
  }
}

} // namespace addr_calc
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ADDR_CALC_SCALAR_H_
