// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/rdna3/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cassert>
#include <cstdint>

namespace rocjitsu {
namespace rdna3 {

namespace {

bool has_saddr(uint32_t saddr) {
  // LLVM prints saddr=0x7c as "off" for RDNA3 global/flat memory ops.
  // Keep 0x7f as a no-base sentinel for older generated configs/tests.
  return saddr != 0x7C && saddr != 0x7F;
}

int64_t sign_extend(uint32_t value, uint32_t bits) {
  const uint32_t shift = 32u - bits;
  return static_cast<int64_t>(static_cast<int32_t>(value << shift) >> shift);
}

uint32_t read_smem_offset(uint32_t soffset, amdgpu::Wavefront &wf) {
  if (soffset == OPR_SMEM_OFFSET_NULL || soffset == 0x7F)
    return 0;
  if (soffset == OPR_SMEM_OFFSET_M0)
    return wf.m0();
  return wf.cu().read_sgpr(wf.sgpr_alloc().base + soffset);
}

} // namespace

uint64_t smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t sbase = wf.sgpr_alloc().base + inst.sbase * 2;
  uint64_t base = (static_cast<uint64_t>(cu.read_sgpr(sbase + 1)) << 32) | cu.read_sgpr(sbase);
  int64_t off = static_cast<int64_t>(static_cast<int32_t>(inst.offset << 11) >> 11);
  off += read_smem_offset(inst.soffset, wf);
  return (base + off) & ~0x3ULL;
}

void flat_calculate_addresses(const FlatMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d) {
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d.lane_mask = exec;
  d.exec_mask = exec;
  d.wf_size = wf.wf_size();
  int64_t offset = sign_extend(inst.offset, 13);

  if (inst.seg == 1) {
    uint32_t saddr_val = 0;
    if (has_saddr(inst.saddr)) {
      uint32_t sb = wf.sgpr_alloc().base + inst.saddr;
      saddr_val = cu.read_sgpr(sb);
    }
    uint64_t scratch_base = wf.scratch_base();
    uint32_t lane_stride = wf.scratch_lane_size();
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      if (!(exec & (1ULL << lane)))
        continue;
      uint32_t vaddr = 0;
      if (inst.sve) {
        uint32_t vbase = wf.vgpr_alloc().base + inst.addr;
        vaddr = cu.read_vgpr(vbase, lane);
      }
      d.per_lane_addr[lane] =
          scratch_base + static_cast<uint64_t>(lane) * lane_stride + vaddr + saddr_val + offset;
    }
    return;
  }

  uint64_t saddr_val = 0;
  if (has_saddr(inst.saddr)) {
    uint32_t sb = wf.sgpr_alloc().base + inst.saddr;
    saddr_val = (static_cast<uint64_t>(cu.read_sgpr(sb + 1)) << 32) | cu.read_sgpr(sb);
  }
  uint32_t priv_hi = static_cast<uint32_t>(wf.private_aperture_base() >> 32);
  uint64_t scratch_base = wf.scratch_base();
  uint32_t lane_stride = wf.scratch_lane_size();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t vbase = wf.vgpr_alloc().base + inst.addr;
    uint64_t vaddr;
    if (has_saddr(inst.saddr)) {
      vaddr = cu.read_vgpr(vbase, lane);
    } else {
      vaddr =
          (static_cast<uint64_t>(cu.read_vgpr(vbase + 1, lane)) << 32) | cu.read_vgpr(vbase, lane);
    }
    uint64_t addr = saddr_val + vaddr + offset;
    if (inst.seg == 0 && priv_hi != 0 && static_cast<uint32_t>(addr >> 32) == priv_hi)
      addr = scratch_base + static_cast<uint64_t>(lane) * lane_stride + (addr & 0xFFFFFFFFULL);
    d.per_lane_addr[lane] = addr;
  }
}

void mubuf_calculate_addresses(const MubufMachineInst &inst, amdgpu::Wavefront &wf,
                               amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::mubuf_calculate_addresses(inst, wf, d);
}

void mtbuf_calculate_addresses(const MtbufMachineInst &inst, amdgpu::Wavefront &wf,
                               amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::mtbuf_calculate_addresses(inst, wf, d);
}

void ds_calculate_addresses(const DsMachineInst &inst, amdgpu::Wavefront &wf,
                            amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::ds_calculate_addresses(inst, wf, d);
}

} // namespace rdna3
} // namespace rocjitsu
