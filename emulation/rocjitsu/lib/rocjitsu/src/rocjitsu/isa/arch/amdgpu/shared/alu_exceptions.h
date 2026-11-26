// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_ALU_EXCEPTIONS_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_ALU_EXCEPTIONS_H_

#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cstdint>

namespace rocjitsu::amdgpu {

inline constexpr uint32_t kAluExceptionModeShift = 12;
inline constexpr uint32_t kAluExceptionModeMask = 0x7fu << kAluExceptionModeShift;
inline constexpr uint32_t kAluExceptionTrapstsMask = 0x7fu;

/// @brief Return the enabled ALU trap causes in EXCP_FLAG bit positions.
///
/// GFX12 moved these enables out of MODE[18:12], where those bits now select
/// VGPR high banks, and into TRAP_CTRL[6:0]. Keeping the normalized mask here
/// prevents exception checks from treating a debugger's trap enables as VGPR
/// selectors, or a shader's VGPR selectors as enabled exceptions.
inline uint32_t alu_exception_trap_enables(const Wavefront &wf) {
  return wf.uses_separate_trap_ctrl()
             ? wf.gfx12_trap_ctrl_raw() & kAluExceptionTrapstsMask
             : (wf.mode_raw() & kAluExceptionModeMask) >> kAluExceptionModeShift;
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_ALU_EXCEPTIONS_H_
