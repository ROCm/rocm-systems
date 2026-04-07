// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cassert>
#include <cstdint>

namespace rocjitsu {
namespace rdna4 {

uint64_t smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf) {
  // GFX12 SMEM: sbase is an aligned SGPR pair, ioffset is a 24-bit signed immediate,
  // soffset is an SGPR index (0x7F = no SGPR offset).
  auto &cu = wf.cu();
  uint32_t sbase = wf.sgpr_alloc().base + inst.sbase * 2;
  uint64_t base = (static_cast<uint64_t>(cu.read_sgpr(sbase + 1)) << 32) | cu.read_sgpr(sbase);
  int64_t off = static_cast<int64_t>(static_cast<int32_t>(inst.ioffset << 8) >> 8);
  if (inst.soffset != 0x7F)
    off += cu.read_sgpr(wf.sgpr_alloc().base + inst.soffset);
  return (base + off) & ~0x3ULL;
}

void flat_calculate_addresses(const VflatMachineInst &inst, amdgpu::Wavefront &wf,
                              std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  // GFX12 VFLAT: 24-bit signed offset, optional SGPR base via saddr.
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  lane_mask = exec;
  int64_t offset = static_cast<int64_t>(static_cast<int32_t>(inst.ioffset << 8) >> 8);
  uint64_t saddr_val = 0;
  if (inst.saddr != 0x7F) {
    uint32_t sb = wf.sgpr_alloc().base + inst.saddr;
    saddr_val = (static_cast<uint64_t>(cu.read_sgpr(sb + 1)) << 32) | cu.read_sgpr(sb);
  }
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t vbase = wf.vgpr_alloc().base + inst.vaddr;
    uint64_t vaddr;
    if (inst.saddr != 0x7F) {
      vaddr = cu.read_vgpr(vbase, lane);
    } else {
      vaddr =
          (static_cast<uint64_t>(cu.read_vgpr(vbase + 1, lane)) << 32) | cu.read_vgpr(vbase, lane);
    }
    addrs[lane] = saddr_val + vaddr + offset;
  }
}

void flat_calculate_addresses(const VglobalMachineInst &inst, amdgpu::Wavefront &wf,
                              std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  // GFX12 VGLOBAL: 24-bit signed offset, optional SGPR base via saddr.
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  lane_mask = exec;
  int64_t offset = static_cast<int64_t>(static_cast<int32_t>(inst.ioffset << 8) >> 8);
  uint64_t saddr_val = 0;
  if (inst.saddr != 0x7F) {
    uint32_t sb = wf.sgpr_alloc().base + inst.saddr;
    saddr_val = (static_cast<uint64_t>(cu.read_sgpr(sb + 1)) << 32) | cu.read_sgpr(sb);
  }
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t vbase = wf.vgpr_alloc().base + inst.vaddr;
    uint64_t vaddr;
    if (inst.saddr != 0x7F) {
      vaddr = cu.read_vgpr(vbase, lane);
    } else {
      vaddr =
          (static_cast<uint64_t>(cu.read_vgpr(vbase + 1, lane)) << 32) | cu.read_vgpr(vbase, lane);
    }
    addrs[lane] = saddr_val + vaddr + offset;
  }
}

void flat_calculate_addresses(const VscratchMachineInst &inst, amdgpu::Wavefront &wf,
                              std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  // GFX12 VSCRATCH: scratch_base + VGPR (32-bit) + saddr + signed 24-bit offset.
  // VGPR is always 32-bit for scratch (not a 64-bit pair).
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  lane_mask = exec;
  int64_t offset = static_cast<int64_t>(static_cast<int32_t>(inst.ioffset << 8) >> 8);
  uint64_t scratch_base = wf.scratch_base();
  uint32_t saddr_val = 0;
  if (inst.saddr != 0x7F) {
    uint32_t sb = wf.sgpr_alloc().base + inst.saddr;
    saddr_val = cu.read_sgpr(sb);
  }
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t vbase = wf.vgpr_alloc().base + inst.vaddr;
    uint32_t vaddr = cu.read_vgpr(vbase, lane);
    addrs[lane] = scratch_base + vaddr + saddr_val + offset;
  }
}

void mubuf_calculate_addresses(const VbufferMachineInst &inst, amdgpu::Wavefront &wf,
                               std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  // GFX12 VBUFFER: rsrc is a 9-bit SGPR index (quad-aligned), soffset is a 7-bit
  // SGPR index, ioffset is a 24-bit signed immediate.
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  lane_mask = exec;
  uint32_t sb = wf.sgpr_alloc().base + inst.rsrc * 4;
  uint64_t base_addr =
      (static_cast<uint64_t>(cu.read_sgpr(sb + 1) & 0xFFFF) << 32) | cu.read_sgpr(sb);
  uint32_t soffset_val = cu.read_sgpr(wf.sgpr_alloc().base + inst.soffset);
  int64_t ioff = static_cast<int64_t>(static_cast<int32_t>(inst.ioffset << 8) >> 8);
  assert(!inst.idxen && "Vbuffer idxen not yet supported");
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t voffset = 0;
    if (inst.offen)
      voffset = cu.read_vgpr(wf.vgpr_alloc().base + inst.vaddr, lane);
    addrs[lane] = base_addr + voffset + ioff + soffset_val;
  }
}

void ds_calculate_addresses(const VdsMachineInst &inst, amdgpu::Wavefront &wf,
                            std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  amdgpu::addr_calc::ds_calculate_addresses(inst, wf, addrs, lane_mask);
}

} // namespace rdna4
} // namespace rocjitsu
