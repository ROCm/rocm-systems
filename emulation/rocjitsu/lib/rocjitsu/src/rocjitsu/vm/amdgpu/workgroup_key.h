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

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_WORKGROUP_KEY_H_
