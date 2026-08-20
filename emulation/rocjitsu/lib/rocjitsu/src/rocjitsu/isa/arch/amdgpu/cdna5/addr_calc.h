// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA5_ADDR_CALC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA5_ADDR_CALC_H_

/// @file Address calculation helpers for gfx1250 memory instructions.

#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mtype.h"

#include <cstdint>

namespace rocjitsu::amdgpu {
class Wavefront;
struct VectorMemState;
} // namespace rocjitsu::amdgpu

namespace rocjitsu::cdna5 {

/// @brief Decoded CDNA5 buffer resource descriptor fields.
struct BufferResource {
  uint64_t base_address = 0;
  uint64_t num_records = 0;
  uint32_t raw_stride = 0;
  uint32_t stride = 0;
  uint8_t stride_scale = 0;
  bool swizzle_enabled = false;
  bool oob_select = false;
  uint8_t type = 0;
};

BufferResource decode_buffer_resource(uint32_t srd0, uint32_t srd1, uint32_t srd2, uint32_t srd3);

/// @brief Sign-extend a CDNA5 24-bit IOFFSET field.
inline int32_t signed_ioffset(uint32_t ioffset) { return static_cast<int32_t>(ioffset << 8) >> 8; }

uint64_t smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf,
                                uint32_t access_size_bytes);

void flat_calculate_addresses(const VflatMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d);

void flat_calculate_addresses(const VglobalMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d);

/// @brief Compute an absolute async LDS address within the workgroup allocation.
///
/// @returns amdgpu::kInvalidLdsAddress when any byte in the access is outside
/// the allocation or the absolute address equals the reserved invalid value or
/// cannot be represented in 32 bits.
uint32_t async_lds_lane_address(const VglobalMachineInst &inst, const amdgpu::Wavefront &wf,
                                uint32_t lds_operand, uint32_t access_size_bytes);

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

} // namespace rocjitsu::cdna5

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA5_ADDR_CALC_H_
