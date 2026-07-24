// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::cli {

struct VisibleGpu {
  uint32_t ordinal = 0;
  uint32_t gpu_id = 0;
  uint32_t gfx_target_version = 0;
  uint64_t unique_id = 0;
};

enum class HostSelectionStatus {
  Selected,
  ExplicitGpuHidden,
  ExplicitGpuIsaMismatch,
  NoIsaMatch,
};

struct HostSelection {
  HostSelectionStatus status = HostSelectionStatus::NoIsaMatch;
  uint32_t gpu_id = 0;
};

struct VisibilityOverride {
  std::string name;
  std::string value;
};

std::vector<VisibleGpu> enumerate_kfd_gpus(const std::vector<VisibleGpu> &candidates);

std::vector<VisibleGpu> filter_rocr_visible_gpus(const std::vector<VisibleGpu> &gpus,
                                                 std::optional<std::string_view> selector);

std::vector<VisibleGpu> filter_client_visible_gpus(const std::vector<VisibleGpu> &gpus,
                                                   std::optional<std::string_view> selector);

std::vector<VisibleGpu> effective_visible_gpus(const std::vector<VisibleGpu> &topology,
                                               std::optional<std::string_view> rocr_visible,
                                               std::optional<std::string_view> hip_visible,
                                               std::optional<std::string_view> cuda_visible);

std::optional<VisibilityOverride> normalized_client_visible_devices(
    const std::vector<VisibleGpu> &topology, std::optional<std::string_view> rocr_visible,
    std::optional<std::string_view> hip_visible, std::optional<std::string_view> cuda_visible,
    std::optional<uint32_t> first_gpu_id = std::nullopt);

std::optional<std::string>
expanded_rocr_visible_devices(const std::vector<VisibleGpu> &topology,
                              std::optional<std::string_view> rocr_visible);

HostSelection select_host_gpu(const std::vector<VisibleGpu> &visible_gpus,
                              uint32_t configured_gpu_id, uint32_t gfx_target_version);

} // namespace rocjitsu::cli
