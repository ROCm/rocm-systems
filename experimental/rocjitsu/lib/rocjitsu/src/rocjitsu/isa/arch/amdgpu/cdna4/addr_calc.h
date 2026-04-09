// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_ADDR_CALC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_ADDR_CALC_H_

/// @file Address calculation free functions for CDNA4 memory instructions.
///
/// One function per encoding format. Called from generated execute() bodies
/// which have access to the encoding's inst_ member. These populate the
/// DynamicInstState data objects with computed addresses.

#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/vm/amdgpu/mtype.h"

#include <array>
#include <cstdint>

namespace rocjitsu {
namespace amdgpu {
class Wavefront;
} // namespace amdgpu

namespace cdna4 {

/// @brief Compute scalar address for SMEM encoding.
///
/// @param inst The SMEM machine instruction encoding fields.
/// @param wf The issuing wavefront (provides SGPR access).
/// @returns The dword-aligned scalar address.
uint64_t smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf);

/// @brief Compute per-lane addresses for FLAT/GLOBAL/SCRATCH encoding.
///
/// Handles flat (unsigned 12-bit offset), global/scratch (signed 13-bit
/// offset with optional SGPR base via saddr).
void flat_calculate_addresses(const FlatMachineInst &inst, amdgpu::Wavefront &wf,
                              std::array<uint64_t, 64> &addrs, uint64_t &lane_mask);

/// @brief Compute per-lane addresses for MUBUF encoding.
void mubuf_calculate_addresses(const MubufMachineInst &inst, amdgpu::Wavefront &wf,
                               std::array<uint64_t, 64> &addrs, uint64_t &lane_mask);

/// @brief Compute per-lane addresses for MTBUF encoding.
void mtbuf_calculate_addresses(const MtbufMachineInst &inst, amdgpu::Wavefront &wf,
                               std::array<uint64_t, 64> &addrs, uint64_t &lane_mask);

/// @brief Compute per-lane addresses for DS encoding.
void ds_calculate_addresses(const DsMachineInst &inst, amdgpu::Wavefront &wf,
                            std::array<uint64_t, 64> &addrs, uint64_t &lane_mask);

/// @brief Derive Mtype from sc0/sc1 encoding bits.
inline amdgpu::Mtype mtype_from_bits(bool sc0, bool sc1) {
  if (sc1)
    return amdgpu::Mtype::UC;
  if (sc0)
    return amdgpu::Mtype::CC;
  return amdgpu::Mtype::RW;
}

} // namespace cdna4
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_ADDR_CALC_H_
