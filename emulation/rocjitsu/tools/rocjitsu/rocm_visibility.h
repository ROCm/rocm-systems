// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rocm_visibility.h
/// @brief ROCm GPU visibility filtering, normalization, and DBT host selection.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::cli {

/// @brief GPU identity used while applying ROCm visibility selectors.
struct VisibleGpu {
  uint32_t ordinal = 0;            ///< Ordinal in the current KFD enumeration.
  uint32_t gpu_id = 0;             ///< Nonzero KFD topology GPU ID.
  uint32_t gfx_target_version = 0; ///< Numeric GFX target version.
  uint64_t unique_id = 0;          ///< KFD unique ID used in GPU UUID selectors.
};

/// @brief Result category for DBT host GPU selection.
enum class HostSelectionStatus {
  Selected,               ///< A visible GPU satisfies the request.
  ExplicitGpuHidden,      ///< The configured GPU is not client-visible.
  ExplicitGpuIsaMismatch, ///< The configured GPU does not match the host ISA.
  NoIsaMatch,             ///< No client-visible GPU matches the host ISA.
};

/// @brief Selected GPU ID and status returned by DBT host selection.
struct HostSelection {
  HostSelectionStatus status = HostSelectionStatus::NoIsaMatch;
  uint32_t gpu_id = 0;
};

/// @brief Environment variable replacement produced by visibility normalization.
struct VisibilityOverride {
  std::string name;
  std::string value;
};

/// @brief Remove non-GPU KFD nodes and assign compact GPU ordinals.
std::vector<VisibleGpu> enumerate_kfd_gpus(const std::vector<VisibleGpu> &candidates);

/// @brief Apply ROCR_VISIBLE_DEVICES filtering and ordering semantics.
std::vector<VisibleGpu> filter_rocr_visible_gpus(const std::vector<VisibleGpu> &gpus,
                                                 std::optional<std::string_view> selector);

/// @brief Apply HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES filtering semantics.
std::vector<VisibleGpu> filter_client_visible_gpus(const std::vector<VisibleGpu> &gpus,
                                                   std::optional<std::string_view> selector);

/// @brief Return GPUs visible after applying ROCR and client selectors in runtime order.
std::vector<VisibleGpu> effective_visible_gpus(const std::vector<VisibleGpu> &topology,
                                               std::optional<std::string_view> rocr_visible,
                                               std::optional<std::string_view> hip_visible,
                                               std::optional<std::string_view> cuda_visible);

/// @brief Normalize or synthesize a client selector using post-ROCR numeric ordinals.
/// @details Without an existing client selector, HIP_VISIBLE_DEVICES is synthesized only when
/// first_gpu_id is present because HIP is the primary ROCm client visibility control.
/// @param first_gpu_id GPU selected from effective_visible_gpus() using the same selectors.
std::optional<VisibilityOverride> normalized_client_visible_devices(
    const std::vector<VisibleGpu> &topology, std::optional<std::string_view> rocr_visible,
    std::optional<std::string_view> hip_visible, std::optional<std::string_view> cuda_visible,
    std::optional<uint32_t> first_gpu_id = std::nullopt);

/// @brief Append the synthetic DBT guest ordinal to a nonempty ROCR selection.
std::optional<std::string>
expanded_rocr_visible_devices(const std::vector<VisibleGpu> &topology,
                              std::optional<std::string_view> rocr_visible);

/// @brief Select an explicit or automatic DBT host from client-visible GPUs.
HostSelection select_host_gpu(const std::vector<VisibleGpu> &visible_gpus,
                              uint32_t configured_gpu_id, uint32_t gfx_target_version);

} // namespace rocjitsu::cli
