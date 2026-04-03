// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/rdna3_5/addr_calc.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/except.h"

#include <array>
#include <cstdint>

namespace rocjitsu {
namespace rdna3_5 {

void flat_calculate_addresses(const FlatMachineInst &inst, amdgpu::Wavefront &wf,
                              std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  (void)inst;
  (void)wf;
  (void)addrs;
  (void)lane_mask;
  throw util::UnimplementedInst("rdna3_5 flat_calculate_addresses not yet implemented");
}

void mubuf_calculate_addresses(const MubufMachineInst &inst, amdgpu::Wavefront &wf,
                               std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  (void)inst;
  (void)wf;
  (void)addrs;
  (void)lane_mask;
  throw util::UnimplementedInst("rdna3_5 mubuf_calculate_addresses not yet implemented");
}

void ds_calculate_addresses(const DsMachineInst &inst, amdgpu::Wavefront &wf,
                            std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  (void)inst;
  (void)wf;
  (void)addrs;
  (void)lane_mask;
  throw util::UnimplementedInst("rdna3_5 ds_calculate_addresses not yet implemented");
}

} // namespace rdna3_5
} // namespace rocjitsu
