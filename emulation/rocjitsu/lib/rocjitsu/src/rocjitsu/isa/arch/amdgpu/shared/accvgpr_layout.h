// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ACCVGPR_LAYOUT_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ACCVGPR_LAYOUT_H_

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

/// AccVGPR offset within a unified VGPR block. On CDNA3/4, AccVGPRs occupy
/// the second half of the 512-register block: acc0 = vgpr_base + 256.
constexpr uint32_t ACC_VGPR_OFFSET = 256;

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_ACCVGPR_LAYOUT_H_
