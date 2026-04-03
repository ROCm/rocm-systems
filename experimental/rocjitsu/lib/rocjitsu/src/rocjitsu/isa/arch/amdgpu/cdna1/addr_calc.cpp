// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/cdna1/addr_calc.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/except.h"

#include <array>
#include <cstdint>

namespace rocjitsu {
namespace cdna1 {

uint64_t smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf) {
  (void)inst;
  (void)wf;
  throw util::UnimplementedInst("cdna1 smem_calculate_address not yet implemented");
  return 0;
}

void flat_calculate_addresses(const FlatMachineInst &inst, amdgpu::Wavefront &wf,
                              std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  (void)inst;
  (void)wf;
  (void)addrs;
  (void)lane_mask;
  throw util::UnimplementedInst("cdna1 flat_calculate_addresses not yet implemented");
}

void mubuf_calculate_addresses(const MubufMachineInst &inst, amdgpu::Wavefront &wf,
                               std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  (void)inst;
  (void)wf;
  (void)addrs;
  (void)lane_mask;
  throw util::UnimplementedInst("cdna1 mubuf_calculate_addresses not yet implemented");
}

void ds_calculate_addresses(const DsMachineInst &inst, amdgpu::Wavefront &wf,
                            std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  (void)inst;
  (void)wf;
  (void)addrs;
  (void)lane_mask;
  throw util::UnimplementedInst("cdna1 ds_calculate_addresses not yet implemented");
}

} // namespace cdna1
} // namespace rocjitsu
