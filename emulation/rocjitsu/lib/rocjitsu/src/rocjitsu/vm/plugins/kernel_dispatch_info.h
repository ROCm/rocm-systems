// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

namespace rocjitsu {

/// @brief Metadata for an AMDGPU kernel dispatch, passed to plugins.
struct KernelDispatchInfo {
  uint32_t dispatch_id = 0;
  uint64_t kernel_object = 0;
  uint64_t entry_pc = 0;
  std::string kernel_name;
  uint32_t grid_size_x = 0, grid_size_y = 0, grid_size_z = 0;
  uint32_t workgroup_size_x = 0, workgroup_size_y = 0, workgroup_size_z = 0;
  uint32_t workgroup_count = 0;
  uint32_t wfs_per_workgroup = 0;
  uint32_t sgprs_per_wf = 0;
  uint32_t vgprs_per_wf = 0;
};

} // namespace rocjitsu
