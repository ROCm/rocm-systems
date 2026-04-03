// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/except.h"

#include <array>
#include <cstdint>

namespace rocjitsu {
namespace rdna4 {

void flat_calculate_addresses(const VflatMachineInst &inst, amdgpu::Wavefront &wf,
                              std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  (void)inst;
  (void)wf;
  (void)addrs;
  (void)lane_mask;
  throw util::UnimplementedInst("rdna4 flat_calculate_addresses not yet implemented");
}

void flat_calculate_addresses(const VglobalMachineInst &inst, amdgpu::Wavefront &wf,
                              std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  (void)inst;
  (void)wf;
  (void)addrs;
  (void)lane_mask;
  throw util::UnimplementedInst("rdna4 global_calculate_addresses not yet implemented");
}

void flat_calculate_addresses(const VscratchMachineInst &inst, amdgpu::Wavefront &wf,
                              std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  (void)inst;
  (void)wf;
  (void)addrs;
  (void)lane_mask;
  throw util::UnimplementedInst("rdna4 scratch_calculate_addresses not yet implemented");
}

void mubuf_calculate_addresses(const VbufferMachineInst &inst, amdgpu::Wavefront &wf,
                               std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  (void)inst;
  (void)wf;
  (void)addrs;
  (void)lane_mask;
  throw util::UnimplementedInst("rdna4 mubuf_calculate_addresses not yet implemented");
}

void ds_calculate_addresses(const VdsMachineInst &inst, amdgpu::Wavefront &wf,
                            std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  (void)inst;
  (void)wf;
  (void)addrs;
  (void)lane_mask;
  throw util::UnimplementedInst("rdna4 ds_calculate_addresses not yet implemented");
}

} // namespace rdna4
} // namespace rocjitsu
