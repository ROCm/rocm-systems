// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/rdna3/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cassert>
#include <cstdint>

namespace rocjitsu {
namespace rdna3 {

uint64_t smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t sbase = wf.sgpr_alloc().base + inst.sbase * 2;
  uint64_t base = (static_cast<uint64_t>(cu.read_sgpr(sbase + 1)) << 32) | cu.read_sgpr(sbase);
  int64_t off = static_cast<int64_t>(static_cast<int32_t>(inst.offset << 11) >> 11);
  if (inst.soffset != 0x7F)
    off += cu.read_sgpr(wf.sgpr_alloc().base + inst.soffset);
  return (base + off) & ~0x3ULL;
}

void flat_calculate_addresses(const FlatMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d) {
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d.lane_mask = exec;
  d.exec_mask = exec;
  int64_t offset = static_cast<int64_t>(static_cast<int32_t>(inst.offset << 20) >> 20);
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
      vaddr = cu.read_vgpr(vbase, lane);
    } else {
      vaddr =
          (static_cast<uint64_t>(cu.read_vgpr(vbase + 1, lane)) << 32) | cu.read_vgpr(vbase, lane);
    }
    d.per_lane_addr[lane] = saddr_val + vaddr + offset;
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
