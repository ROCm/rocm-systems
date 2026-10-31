// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <string>

namespace rocjitsu {

namespace diagnostics {
class HazardSites;
} // namespace diagnostics

inline constexpr char kUnknownKernelIdentity[] = "?";

/// @brief Metadata for an AMDGPU kernel dispatch, passed to plugins.
struct KernelDispatchInfo {
  uint32_t dispatch_id = 0;
  uint64_t kernel_object = 0;
  uint64_t entry_pc = 0;
  std::string kernel_symbol;
  std::string kernel_name;
  uint32_t lds_size_bytes = 0;
  uint32_t wave_size = 0;
  rj_code_target_id_t code_target = ROCJITSU_CODE_TARGET_INVALID;
  uint32_t grid_size_x = 0, grid_size_y = 0, grid_size_z = 0;
  uint32_t workgroup_size_x = 0, workgroup_size_y = 0, workgroup_size_z = 0;
  uint32_t cluster_size_x = 1, cluster_size_y = 1, cluster_size_z = 1;
  uint32_t workgroup_count = 0;
  uint32_t wfs_per_workgroup = 0;
  uint32_t sgprs_per_wf = 0;
  uint32_t vgprs_per_wf = 0;
  /// Submitting process, as the core's hazard records report it.
  uint32_t process_id = 0;
  /// Hazard sites of the dispatch's SoC, whose settings, GPU ordinal and writer every
  /// checker shares. Null when the dispatch has no command processor.
  const diagnostics::HazardSites *hazard_sites = nullptr;

  std::string kernelNameOrUnknown() const {
    return kernel_name.empty() ? kUnknownKernelIdentity : kernel_name;
  }
  std::string kernelSymbolOrUnknown() const {
    return kernel_symbol.empty() ? kUnknownKernelIdentity : kernel_symbol;
  }
};

} // namespace rocjitsu
