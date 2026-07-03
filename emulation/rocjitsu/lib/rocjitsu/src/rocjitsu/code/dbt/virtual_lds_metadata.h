// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocr::llvm::amdhsa {
struct kernel_descriptor_t;
} // namespace rocr::llvm::amdhsa

namespace rocjitsu {

/// @brief Non-allocated ELF section carrying rocjitsu virtual-LDS dispatch facts.
inline constexpr std::string_view kVirtualLdsMetadataSectionName = ".rocjitsu.lds";

/// @brief The virtual-LDS runtime-state pointer is stored in the dispatch packet, not kernarg.
///
/// @details The default ABI appends runtime state to an extended kernarg
/// buffer. Some hand-written kernels advertise `kernarg_size == 0` while still
/// reading the original kernarg pointer, so rocjitsu cannot safely copy and
/// redirect that argument block. For those kernels, the translated prologue
/// loads a state-block pointer from rocjitsu-owned dispatch-packet state instead.
inline constexpr uint16_t kVirtualLdsFlagBackingPointerInDispatchPacket = 1u << 0;

/// @brief Runtime state uses a rocjitsu per-dispatch state block, not a raw pointer.
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
  /// @brief Source kernarg byte count to copy before DBT appends runtime data.
  uint32_t kernarg_size = 0;
  /// @brief Byte offset where dispatch rewriting writes or references runtime state.
  uint32_t backing_pointer_kernarg_offset = 0;
  uint16_t virtual_lds_base_sgpr = 0;
  uint16_t flags = 0;
};

/// @brief Per-descriptor virtual-LDS dispatch facts stored in reserved KD bytes.
///
/// @details The `.rocjitsu.lds` ELF section is still the authoritative offline
/// metadata, but the runtime fast path can see only the AQL packet's
/// `kernel_object`. Mirroring the small dispatch-critical subset into AMDHSA
/// descriptor reserved bytes lets the HSA hook patch packets even when ROCR
/// rings the queue through a direct doorbell write path that bypasses symbol
/// callbacks. These bytes are rocjitsu-private and are ignored by hardware.
struct VirtualLdsDescriptorDispatchMetadata {
  int64_t virtual_descriptor_delta = 0;
  uint32_t kernarg_size = 0;
  uint32_t backing_pointer_kernarg_offset = 0;
  uint16_t flags = 0;
};

/// @brief Store virtual-LDS dispatch facts in AMDHSA descriptor reserved bytes.
void write_virtual_lds_descriptor_dispatch_metadata(
    rocr::llvm::amdhsa::kernel_descriptor_t &descriptor,
    const VirtualLdsDescriptorDispatchMetadata &metadata);

/// @brief Read rocjitsu virtual-LDS dispatch facts from descriptor reserved bytes.
[[nodiscard]] std::optional<VirtualLdsDescriptorDispatchMetadata>
read_virtual_lds_descriptor_dispatch_metadata(
    const rocr::llvm::amdhsa::kernel_descriptor_t &descriptor);

/// @brief Serialize virtual-LDS metadata for storage in @ref kVirtualLdsMetadataSectionName.
[[nodiscard]] std::vector<uint8_t>
serialize_virtual_lds_metadata(std::span<const VirtualLdsKernelMetadata> kernels);

/// @brief Parse a @ref kVirtualLdsMetadataSectionName payload.
[[nodiscard]] std::optional<std::vector<VirtualLdsKernelMetadata>>
parse_virtual_lds_metadata(std::span<const uint8_t> bytes);

} // namespace rocjitsu
