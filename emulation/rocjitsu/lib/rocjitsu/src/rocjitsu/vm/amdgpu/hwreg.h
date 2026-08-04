// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hwreg.h
/// @brief Shader-visible AMDGPU hardware register access helpers.

#ifndef ROCJITSU_VM_AMDGPU_HWREG_H_
#define ROCJITSU_VM_AMDGPU_HWREG_H_

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

class Wavefront;

/// @brief Result of a shader HWREG read or write.
///
/// @details Success means the addressed HWREG field is backed by wave state and
/// the raw bits were read or updated. It does not imply that every writable MODE
/// bit has an implemented execution side effect in the simulator.
enum class HwregAccessResult : uint8_t {
  Success,
  Unsupported,
  ReadOnly,
  Privileged,
};

/// @brief Extract the register ID field from an encoded HWREG operand.
[[nodiscard]] uint32_t hwreg_id(uint16_t hwreg);

/// @brief Return the architecture-specific name for an encoded HWREG operand.
[[nodiscard]] const char *hwreg_name(const Wavefront &wf, uint16_t hwreg);

/// @brief Return a stable diagnostic string for an HWREG access result.
[[nodiscard]] const char *hwreg_access_result_name(HwregAccessResult result);

/// @brief Read an encoded HWREG bitfield into the low bits of value.
///
/// @details On failed reads, `value` is set to zero before returning the
/// non-success result.
[[nodiscard]] HwregAccessResult read_hwreg_field(Wavefront &wf, uint16_t hwreg, uint32_t &value);

/// @brief Write low source bits into an encoded HWREG bitfield.
///
/// @details Failed writes leave wave state unchanged. Unknown registers report
/// Unsupported; known read-only or privileged registers report that policy
/// before checking whether rocjitsu backs the register state.
[[nodiscard]] HwregAccessResult write_hwreg_field(Wavefront &wf, uint16_t hwreg, uint32_t src);

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_HWREG_H_
