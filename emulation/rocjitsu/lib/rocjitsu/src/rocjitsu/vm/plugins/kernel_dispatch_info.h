// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

namespace rocjitsu {

/// @brief Metadata for an AMDGPU kernel dispatch, passed to plugins.
struct KernelDispatchInfo {
  uint64_t kernel_object;
  uint64_t entry_pc;
  std::string kernel_name;
  uint32_t grid_size_x, grid_size_y, grid_size_z;
  uint32_t workgroup_size_x, workgroup_size_y, workgroup_size_z;
  uint32_t workgroup_count;
  uint32_t wfs_per_workgroup;
  uint32_t sgprs_per_wf;
  uint32_t vgprs_per_wf;
};

} // namespace rocjitsu
