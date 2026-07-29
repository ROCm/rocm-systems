// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sidecar_metadata.h
/// @brief Generic metadata for alternate kernel descriptor variants.

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

/// @brief Non-allocated ELF section carrying generic sidecar variant records.
inline constexpr std::string_view kSidecarMetadataSectionName = ".rocjitsu.sidecar";

/// @brief One alternate descriptor attached to a normal kernel descriptor.
///
/// @details Sidecars only describe the relationship between two descriptor
/// variants. The feature that selects a variant owns its condition and any
/// additional runtime metadata. This keeps sidecars usable by DBI and future
/// DBT features that do not extend kernargs or virtualize LDS.
struct SidecarVariantMetadata {
  std::string kernel_name;
  std::string variant_name;
  uint64_t normal_descriptor_vaddr = 0;
  uint64_t variant_descriptor_vaddr = 0;
};

/// @brief Serialize sidecar records for @ref kSidecarMetadataSectionName.
[[nodiscard]] std::vector<uint8_t>
serialize_sidecar_metadata(std::span<const SidecarVariantMetadata> variants);

/// @brief Parse a @ref kSidecarMetadataSectionName payload.
[[nodiscard]] std::optional<std::vector<SidecarVariantMetadata>>
parse_sidecar_metadata(std::span<const uint8_t> bytes);

} // namespace rocjitsu
