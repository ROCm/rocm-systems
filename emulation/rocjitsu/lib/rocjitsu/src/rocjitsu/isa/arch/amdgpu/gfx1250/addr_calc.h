// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_GFX1250_ADDR_CALC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_GFX1250_ADDR_CALC_H_

/// @file Address calculation helpers for gfx1250 memory instructions.

#include "rocjitsu/isa/arch/amdgpu/gfx1250/machine_insts.h"
#include "rocjitsu/vm/amdgpu/mtype.h"

#include <cstdint>

namespace rocjitsu::amdgpu {
class Wavefront;
struct VectorMemState;
} // namespace rocjitsu::amdgpu

namespace rocjitsu::gfx1250 {

uint64_t smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf,
                                uint32_t access_size_bytes);

void flat_calculate_addresses(const VflatMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d);

void flat_calculate_addresses(const VglobalMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d);

void flat_calculate_addresses(const VscratchMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d);

void mubuf_calculate_addresses(const VbufferMachineInst &inst, amdgpu::Wavefront &wf,
                               amdgpu::VectorMemState &d);

void ds_calculate_addresses(const VdsMachineInst &inst, amdgpu::Wavefront &wf,
                            amdgpu::VectorMemState &d);

inline amdgpu::Mtype mtype_from_bits(bool sc0, bool sc1) {
  if (sc1)
    return amdgpu::Mtype::UC;
  if (sc0)
    return amdgpu::Mtype::CC;
  return amdgpu::Mtype::RW;
}

} // namespace rocjitsu::gfx1250

#endif // ROCJITSU_ISA_ARCH_AMDGPU_GFX1250_ADDR_CALC_H_
