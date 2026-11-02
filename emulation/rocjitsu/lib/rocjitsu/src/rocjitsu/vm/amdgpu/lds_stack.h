// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#ifndef ROCJITSU_VM_AMDGPU_LDS_STACK_H_
#define ROCJITSU_VM_AMDGPU_LDS_STACK_H_
#include <cstdint>
namespace rocjitsu::amdgpu {
class Wavefront;
class VectorMemState;
void prepare_lds_stack(Wavefront &wf, VectorMemState &d, uint32_t addr, uint32_t last,
                       uint32_t nodes, uint32_t dst, uint32_t push, uint32_t pop, uint32_t size,
                       uint32_t flags);
void execute_lds_stack(Wavefront &wf, VectorMemState &d);
} // namespace rocjitsu::amdgpu
#endif
