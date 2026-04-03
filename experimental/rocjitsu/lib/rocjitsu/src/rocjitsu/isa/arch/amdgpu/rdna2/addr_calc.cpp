// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/rdna2/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/except.h"

#include <cassert>
#include <cstdint>

namespace rocjitsu {
namespace rdna2 {

uint64_t smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf) {
  (void)inst;
  (void)wf;
  // RDNA2 SmemMachineInst lacks soffset_en/imm fields required by shared template.
  throw util::UnimplementedInst("rdna2 smem_calculate_address not yet implemented");
  return 0;
}

void flat_calculate_addresses(const FlatMachineInst &inst, amdgpu::Wavefront &wf,
                              std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  (void)inst;
  (void)wf;
  (void)addrs;
  (void)lane_mask;
  // RDNA2 FlatMachineInst lacks pad_12 field required by shared flat template.
  throw util::UnimplementedInst("rdna2 flat_calculate_addresses not yet implemented");
}

void mubuf_calculate_addresses(const MubufMachineInst &inst, amdgpu::Wavefront &wf,
                               std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  amdgpu::addr_calc::mubuf_calculate_addresses(inst, wf, addrs, lane_mask);
}

void mtbuf_calculate_addresses(const MtbufMachineInst &inst, amdgpu::Wavefront &wf,
                               std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  amdgpu::addr_calc::mtbuf_calculate_addresses(inst, wf, addrs, lane_mask);
}

void ds_calculate_addresses(const DsMachineInst &inst, amdgpu::Wavefront &wf,
                            std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  amdgpu::addr_calc::ds_calculate_addresses(inst, wf, addrs, lane_mask);
}

} // namespace rdna2
} // namespace rocjitsu
