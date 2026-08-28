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

/// @brief Encode the GFX12 STATE_PRIV fields backed by internal STATUS.
[[nodiscard]] uint32_t gfx12_state_priv_from_status(uint32_t status, bool scratch_enabled);

/// @brief Replace internal STATUS fields represented by GFX12 STATE_PRIV.
[[nodiscard]] uint32_t update_status_from_gfx12_state_priv(uint32_t status, uint32_t state_priv);

/// @brief Encode GFX12 EXCP_FLAG_PRIV from RocJITsu's common TRAPSTS state.
[[nodiscard]] uint32_t gfx12_excp_flag_priv_from_trapsts(uint32_t trapsts);

/// @brief Replace common TRAPSTS fields represented by GFX12 EXCP_FLAG_PRIV.
[[nodiscard]] uint32_t update_trapsts_from_gfx12_excp_flag_priv(uint32_t trapsts,
                                                                uint32_t excp_flag_priv);

/// @brief Replace common TRAPSTS fields represented by GFX12 EXCP_FLAG_USER.
[[nodiscard]] uint32_t update_trapsts_from_gfx12_excp_flag_user(uint32_t trapsts,
                                                                uint32_t excp_flag_user);

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
