// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/except.h"

#include <cassert>
#include <cstdint>

namespace rocjitsu {
namespace rdna4 {

uint64_t smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf) {
  (void)inst;
  (void)wf;
  // RDNA4 SmemMachineInst uses ioffset (not offset) and lacks soffset_en/imm fields.
  throw util::UnimplementedInst("rdna4 smem_calculate_address not yet implemented");
  return 0;
}

void flat_calculate_addresses(const VflatMachineInst &inst, amdgpu::Wavefront &wf,
                              std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  (void)inst;
  (void)wf;
  (void)addrs;
  (void)lane_mask;
  // RDNA4 uses VflatMachineInst which is incompatible with shared flat template.
  throw util::UnimplementedInst("rdna4 flat_calculate_addresses not yet implemented");
}

void flat_calculate_addresses(const VglobalMachineInst &inst, amdgpu::Wavefront &wf,
                              std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  (void)inst;
  (void)wf;
  (void)addrs;
  (void)lane_mask;
  // RDNA4 uses VglobalMachineInst which is incompatible with shared flat template.
  throw util::UnimplementedInst("rdna4 global_calculate_addresses not yet implemented");
}

void flat_calculate_addresses(const VscratchMachineInst &inst, amdgpu::Wavefront &wf,
                              std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  (void)inst;
  (void)wf;
  (void)addrs;
  (void)lane_mask;
  // RDNA4 uses VscratchMachineInst which is incompatible with shared flat template.
  throw util::UnimplementedInst("rdna4 scratch_calculate_addresses not yet implemented");
}

void mubuf_calculate_addresses(const VbufferMachineInst &inst, amdgpu::Wavefront &wf,
                               std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  (void)inst;
  (void)wf;
  (void)addrs;
  (void)lane_mask;
  // RDNA4 VbufferMachineInst uses rsrc/ioffset instead of srsrc/offset.
  throw util::UnimplementedInst("rdna4 mubuf_calculate_addresses not yet implemented");
}

void ds_calculate_addresses(const VdsMachineInst &inst, amdgpu::Wavefront &wf,
                            std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  amdgpu::addr_calc::ds_calculate_addresses(inst, wf, addrs, lane_mask);
}

} // namespace rdna4
} // namespace rocjitsu
