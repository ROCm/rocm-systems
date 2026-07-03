// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

/// @brief Non-allocated ELF section carrying rocjitsu virtual-LDS dispatch facts.
inline constexpr std::string_view kVirtualLdsMetadataSectionName = ".rocjitsu.lds";

/// @brief Runtime state uses a rocjitsu kernarg-wrapper payload, not a raw pointer.
inline constexpr uint16_t kVirtualLdsFlagRuntimeStateBlock = 1u << 1;

/// @brief Source descriptor exposes the matching workgroup-id SGPR.
inline constexpr uint16_t kVirtualLdsFlagWorkgroupIdX = 1u << 2;
inline constexpr uint16_t kVirtualLdsFlagWorkgroupIdY = 1u << 3;
inline constexpr uint16_t kVirtualLdsFlagWorkgroupIdZ = 1u << 4;

/// @brief One kernel's static virtual-LDS dispatch metadata.
///
/// @details Descriptor addresses are ELF virtual addresses, matching the value
/// that ROCR exposes as a loaded `kernel_object` after adding the code-object
/// load base. Dispatch rewriting can switch from the normal descriptor VA to
/// the virtual descriptor VA when static plus dynamic LDS exceeds the host
/// limit.
struct VirtualLdsKernelMetadata {
  std::string kernel_name;
  uint64_t normal_descriptor_vaddr = 0;
  uint64_t virtual_descriptor_vaddr = 0;
  uint32_t static_lds_bytes = 0;
  /// @brief Source kernarg byte count copied into the wrapper prefix.
  uint32_t kernarg_size = 0;
  /// @brief Wrapper byte offset where dispatch rewriting writes runtime state.
  uint32_t backing_pointer_kernarg_offset = 0;
  uint16_t virtual_lds_base_sgpr = 0;
  uint16_t flags = 0;
};

/// @brief Serialize virtual-LDS metadata for storage in @ref kVirtualLdsMetadataSectionName.
[[nodiscard]] std::vector<uint8_t>
serialize_virtual_lds_metadata(std::span<const VirtualLdsKernelMetadata> kernels);

/// @brief Parse a @ref kVirtualLdsMetadataSectionName payload.
[[nodiscard]] std::optional<std::vector<VirtualLdsKernelMetadata>>
parse_virtual_lds_metadata(std::span<const uint8_t> bytes);

} // namespace rocjitsu
