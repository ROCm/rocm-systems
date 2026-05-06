// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

namespace rocjitsu {

/// @brief Metadata for an AMDGPU kernel dispatch, passed to plugins.
struct KernelDispatchInfo {
  uint64_t kernel_object{};
  uint64_t entry_pc{};
  std::string kernel_name;
  uint32_t grid_size_x{}, grid_size_y{}, grid_size_z{};
  uint32_t workgroup_size_x{}, workgroup_size_y{}, workgroup_size_z{};
  uint32_t workgroup_count{};
  uint32_t wfs_per_workgroup{};
  uint32_t sgprs_per_wf{};
  uint32_t vgprs_per_wf{};

  KernelDispatchInfo &setKernelObject(uint64_t v) {
    kernel_object = v;
    return *this;
  }
  KernelDispatchInfo &setEntryPc(uint64_t v) {
    entry_pc = v;
    return *this;
  }
  KernelDispatchInfo &setKernelName(std::string v) {
    kernel_name = std::move(v);
    return *this;
  }
  KernelDispatchInfo &setGridSize(uint32_t x, uint32_t y, uint32_t z) {
    grid_size_x = x;
    grid_size_y = y;
    grid_size_z = z;
    return *this;
  }
  KernelDispatchInfo &setWorkgroupSize(uint32_t x, uint32_t y, uint32_t z) {
    workgroup_size_x = x;
    workgroup_size_y = y;
    workgroup_size_z = z;
    return *this;
  }
  KernelDispatchInfo &setWorkgroupCount(uint32_t v) {
    workgroup_count = v;
    return *this;
  }
  KernelDispatchInfo &setWfsPerWorkgroup(uint32_t v) {
    wfs_per_workgroup = v;
    return *this;
  }
  KernelDispatchInfo &setSgprsPerWf(uint32_t v) {
    sgprs_per_wf = v;
    return *this;
  }
  KernelDispatchInfo &setVgprsPerWf(uint32_t v) {
    vgprs_per_wf = v;
    return *this;
  }
};

} // namespace rocjitsu
