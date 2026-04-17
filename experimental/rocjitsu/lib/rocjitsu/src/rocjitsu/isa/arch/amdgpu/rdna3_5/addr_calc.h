// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_ADDR_CALC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_ADDR_CALC_H_

/// @file Address calculation stubs for RDNA3.5 memory instructions.
///
/// Phase A stubs — implementations are placeholders. Phase B will provide
/// correct address calculation logic shared with the RDNA3 implementation.

#include "rocjitsu/isa/arch/amdgpu/rdna3_5/machine_insts.h"
#include "rocjitsu/vm/amdgpu/mtype.h"

#include <array>
#include <cstdint>

namespace rocjitsu {
namespace amdgpu {
class Wavefront;
} // namespace amdgpu

namespace rdna3_5 {

uint64_t smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf);

void flat_calculate_addresses(const FlatMachineInst &inst, amdgpu::Wavefront &wf,
                              std::array<uint64_t, 64> &addrs, uint64_t &lane_mask);

void mubuf_calculate_addresses(const MubufMachineInst &inst, amdgpu::Wavefront &wf,
                               std::array<uint64_t, 64> &addrs, uint64_t &lane_mask);

void mtbuf_calculate_addresses(const MtbufMachineInst &inst, amdgpu::Wavefront &wf,
                               std::array<uint64_t, 64> &addrs, uint64_t &lane_mask);

void ds_calculate_addresses(const DsMachineInst &inst, amdgpu::Wavefront &wf,
                            std::array<uint64_t, 64> &addrs, uint64_t &lane_mask);

/// @brief Derive Mtype from sc0/sc1 encoding bits (GFX11 coherency model stub).
inline amdgpu::Mtype mtype_from_bits(bool sc0, bool sc1) {
  if (sc1)
    return amdgpu::Mtype::UC;
  if (sc0)
    return amdgpu::Mtype::CC;
  return amdgpu::Mtype::RW;
}

} // namespace rdna3_5
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_ADDR_CALC_H_
