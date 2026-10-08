// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_WORKGROUP_KEY_H_
#define ROCJITSU_VM_AMDGPU_WORKGROUP_KEY_H_

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

inline constexpr uint64_t wg_key(uint32_t dispatch_id, uint32_t wg_id) {
  return (uint64_t(dispatch_id) << 32) | wg_id;
}

/// @brief Key for dispatch-global state shared by every workgroup of a dispatch
/// (e.g. GWS resources). Keeps the high dispatch_id bits so the shared
/// @c wg_key high-bit convention (``key >> 32 == dispatch_id``) still selects
/// all of a dispatch's entries.
inline constexpr uint64_t gws_key(uint32_t dispatch_id) { return uint64_t(dispatch_id) << 32; }

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_WORKGROUP_KEY_H_
