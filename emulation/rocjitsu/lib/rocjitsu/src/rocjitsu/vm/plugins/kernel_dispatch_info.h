// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

namespace rocjitsu {

inline constexpr char kUnknownKernelIdentity[] = "?";

/// @brief Metadata for an AMDGPU kernel dispatch, passed to plugins.
struct KernelDispatchInfo {
  uint32_t dispatch_id = 0;
  /// @brief Hardware queue the packet was submitted on.
  ///
  /// @details dispatch_id is allocated per command processor, and a multi-XCD
  /// part has one command processor per XCD, so ids collide across them.
  /// Anything correlating dispatches over a whole device — a profiler, a timing
  /// model — has to key on the pair, not on dispatch_id alone.
  uint32_t queue_id = 0;
  uint64_t kernel_object = 0;
  uint64_t entry_pc = 0;
  std::string kernel_symbol;
  std::string kernel_name;
  uint32_t grid_size_x = 0, grid_size_y = 0, grid_size_z = 0;
  uint32_t workgroup_size_x = 0, workgroup_size_y = 0, workgroup_size_z = 0;
  uint32_t workgroup_count = 0;
  uint32_t wfs_per_workgroup = 0;
  uint32_t sgprs_per_wf = 0;
  uint32_t vgprs_per_wf = 0;
  /// @brief Group segment reserved per workgroup, in bytes, as allocated.
  ///
  /// @details The allocated size rather than the kernel descriptor's request:
  /// LDS is handed out in granules, and occupancy is decided by what was taken
  /// rather than by what was asked for.
  uint32_t lds_bytes_per_workgroup = 0;
  /// @brief Lanes per wavefront on the compute units this dispatch runs on.
  uint32_t wave_size = 0;

  std::string kernelNameOrUnknown() const {
    return kernel_name.empty() ? kUnknownKernelIdentity : kernel_name;
  }
  std::string kernelSymbolOrUnknown() const {
    return kernel_symbol.empty() ? kUnknownKernelIdentity : kernel_symbol;
  }
};

} // namespace rocjitsu
