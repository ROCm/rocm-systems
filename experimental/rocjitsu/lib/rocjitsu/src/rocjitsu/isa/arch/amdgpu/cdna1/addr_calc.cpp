// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/cdna1/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cassert>
#include <cstdint>

namespace rocjitsu {
namespace cdna1 {

uint64_t smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf) {
  return amdgpu::addr_calc::smem_calculate_address(inst, wf);
}

void flat_calculate_addresses(const FlatMachineInst &inst, amdgpu::Wavefront &wf,
                              std::array<uint64_t, 64> &addrs, uint64_t &lane_mask) {
  amdgpu::addr_calc::flat_calculate_addresses(inst, wf, addrs, lane_mask);
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

} // namespace cdna1
} // namespace rocjitsu
