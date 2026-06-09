// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_ADDR_CALC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_ADDR_CALC_H_

#include "rocjitsu/isa/arch/amdgpu/rdna3_5/machine_insts.h"
#include "rocjitsu/vm/amdgpu/mtype.h"

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {
class Wavefront;
struct VectorMemState;
} // namespace amdgpu

namespace rdna3_5 {

uint64_t smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf);

void flat_calculate_addresses(const FlatMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d);

void mubuf_calculate_addresses(const MubufMachineInst &inst, amdgpu::Wavefront &wf,
                               amdgpu::VectorMemState &d);

void mtbuf_calculate_addresses(const MtbufMachineInst &inst, amdgpu::Wavefront &wf,
                               amdgpu::VectorMemState &d);

void ds_calculate_addresses(const DsMachineInst &inst, amdgpu::Wavefront &wf,
                            amdgpu::VectorMemState &d);

} // namespace rdna3_5
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA3_5_ADDR_CALC_H_
