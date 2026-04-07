// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/cdna3/addr_calc.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cassert>
#include <cstdint>

namespace rocjitsu {
namespace cdna3 {

uint64_t smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t sbase = wf.sgpr_alloc().base + inst.sbase * 2;
  uint64_t base = (static_cast<uint64_t>(cu.read_sgpr(sbase + 1)) << 32) | cu.read_sgpr(sbase);
  uint64_t off = 0;
  if (inst.soffset_en)
    off += cu.read_sgpr(wf.sgpr_alloc().base + inst.soffset);
  if (inst.imm)
    off += static_cast<int64_t>(static_cast<int32_t>(inst.offset << 11) >> 11);
  return (base + off) & ~0x3ULL;
}

void flat_calculate_addresses(const FlatMachineInst &inst, amdgpu::Wavefront &wf,
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

  for (uint32_t lane = 0; lane < 64; ++lane) {
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

void mubuf_calculate_addresses(const MubufMachineInst &inst, amdgpu::Wavefront &wf,
                               std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  lane_mask = exec;
  uint32_t sb = wf.sgpr_alloc().base + inst.srsrc * 4;
  uint64_t base_addr =
      (static_cast<uint64_t>(cu.read_sgpr(sb + 1) & 0xFFFF) << 32) | cu.read_sgpr(sb);
  uint32_t soffset_val = cu.read_sgpr(wf.sgpr_alloc().base + inst.soffset);
  assert(!inst.idxen && "Mubuf idxen not yet supported");
  for (uint32_t lane = 0; lane < 64; ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t voffset = 0;
    if (inst.offen)
      voffset = cu.read_vgpr(wf.vgpr_alloc().base + inst.vaddr, lane);
    addrs[lane] = base_addr + voffset + inst.offset + soffset_val;
  }
}

void mtbuf_calculate_addresses(const MtbufMachineInst &inst, amdgpu::Wavefront &wf,
                               std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  assert(!inst.idxen && "Mtbuf idxen not yet supported");
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  lane_mask = exec;
  uint32_t sb = wf.sgpr_alloc().base + inst.srsrc * 4;
  uint64_t base_addr =
      (static_cast<uint64_t>(cu.read_sgpr(sb + 1) & 0xFFFF) << 32) | cu.read_sgpr(sb);
  uint32_t soffset_val = cu.read_sgpr(wf.sgpr_alloc().base + inst.soffset);
  for (uint32_t lane = 0; lane < 64; ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t voffset = 0;
    if (inst.offen)
      voffset = cu.read_vgpr(wf.vgpr_alloc().base + inst.vaddr, lane);
    addrs[lane] = base_addr + voffset + inst.offset + soffset_val;
  }
}

void ds_calculate_addresses(const DsMachineInst &inst, amdgpu::Wavefront &wf,
                            std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  lane_mask = exec;
  for (uint32_t lane = 0; lane < 64; ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    addrs[lane] = cu.read_vgpr(wf.vgpr_alloc().base + inst.addr, lane) + inst.offset0;
  }
}

} // namespace cdna3
} // namespace rocjitsu
