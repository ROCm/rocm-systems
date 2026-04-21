// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_ADDR_CALC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_ADDR_CALC_H_

/// @file Address calculation stubs for RDNA4 memory instructions.
///
/// RDNA4 uses restructured flat memory encodings (VflatMachineInst,
/// VbufferMachineInst, VdsMachineInst) instead of the FlatMachineInst /
/// MubufMachineInst used by earlier ISAs.
///
/// Phase A stubs — implementations are placeholders. Phase B will provide
/// correct address calculation logic for the GFX12 memory model.
///
/// Note: This header is included from rdna4/isa.h because the generated
/// vflat.cpp, vglobal.cpp, vscratch.cpp, vbuffer.cpp, and vds.cpp do not
/// include it directly (codegen limitation to be fixed in Phase B.9).

#include "rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h"
#include "rocjitsu/vm/amdgpu/mtype.h"

#include <array>
#include <cstdint>

namespace rocjitsu {
namespace amdgpu {
class Wavefront;
} // namespace amdgpu

namespace rdna4 {

uint64_t smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf);

void flat_calculate_addresses(const VflatMachineInst &inst, amdgpu::Wavefront &wf,
                              std::array<uint64_t, 64> &addrs, uint64_t &lane_mask);

void flat_calculate_addresses(const VglobalMachineInst &inst, amdgpu::Wavefront &wf,
                              std::array<uint64_t, 64> &addrs, uint64_t &lane_mask);

void flat_calculate_addresses(const VscratchMachineInst &inst, amdgpu::Wavefront &wf,
                              std::array<uint64_t, 64> &addrs, uint64_t &lane_mask);

void mubuf_calculate_addresses(const VbufferMachineInst &inst, amdgpu::Wavefront &wf,
                               std::array<uint64_t, 64> &addrs, uint64_t &lane_mask);

void ds_calculate_addresses(const VdsMachineInst &inst, amdgpu::Wavefront &wf,
                            std::array<uint64_t, 64> &addrs, uint64_t &lane_mask);

/// @brief Derive Mtype from sc0/sc1 encoding bits (GFX12 coherency model stub).
inline amdgpu::Mtype mtype_from_bits(bool sc0, bool sc1) {
  if (sc1)
    return amdgpu::Mtype::UC;
  if (sc0)
    return amdgpu::Mtype::CC;
  return amdgpu::Mtype::RW;
}

} // namespace rdna4
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_ADDR_CALC_H_
