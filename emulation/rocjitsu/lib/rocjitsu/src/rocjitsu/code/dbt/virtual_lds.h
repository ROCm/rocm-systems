// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file virtual_lds.h
/// @brief Shared virtual-LDS ABI, metadata, and dispatch-planning interfaces.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace rocjitsu {

/// @brief Name of the virtual-LDS payload in generic kernarg extension metadata.
inline constexpr std::string_view kVirtualLdsRuntimeStatePayloadName = "virtual-lds-state";

/// @brief GPU-visible state written by the HSA hook and read by an entry prologue.
struct VirtualLdsDispatchState {
  uint64_t backing_base = 0;
  uint32_t stride_x = 0;
  uint32_t stride_y = 0;
  uint32_t stride_z = 0;
  uint32_t reserved = 0;
};

inline constexpr uint32_t kVirtualLdsStateBackingBaseOffset =
    offsetof(VirtualLdsDispatchState, backing_base);
inline constexpr uint32_t kVirtualLdsStateStrideXOffset =
    offsetof(VirtualLdsDispatchState, stride_x);
inline constexpr uint32_t kVirtualLdsStateStrideYOffset =
    offsetof(VirtualLdsDispatchState, stride_y);
inline constexpr uint32_t kVirtualLdsStateStrideZOffset =
    offsetof(VirtualLdsDispatchState, stride_z);
inline constexpr uint32_t kVirtualLdsRuntimeStateBytes = sizeof(VirtualLdsDispatchState);

static_assert(std::is_standard_layout_v<VirtualLdsDispatchState>);
static_assert(sizeof(VirtualLdsDispatchState) == 24);

/// @brief Sidecar variant name selected by the virtual-LDS policy.
inline constexpr std::string_view kVirtualLdsSidecarVariantName = "virtual-lds";

/// @brief Non-allocated ELF section carrying virtual-LDS policy facts.
inline constexpr std::string_view kVirtualLdsMetadataSectionName = ".rocjitsu.lds";

/// @brief Runtime state uses a @c VirtualLdsDispatchState extension payload.
inline constexpr uint16_t kVirtualLdsFlagRuntimeStateBlock = 1u << 1;

/// @brief Source descriptor exposes the matching workgroup-id SGPR.
inline constexpr uint16_t kVirtualLdsFlagWorkgroupIdX = 1u << 2;
inline constexpr uint16_t kVirtualLdsFlagWorkgroupIdY = 1u << 3;
inline constexpr uint16_t kVirtualLdsFlagWorkgroupIdZ = 1u << 4;

/// @brief All flag bits this build understands.
///
/// @details Metadata is produced and consumed by the same rocjitsu build, so an
/// unknown flag bit means either a version skew or a corrupt section. The parser
/// rejects records carrying bits outside this mask rather than silently ignoring
/// them, so a future feature bit cannot be misinterpreted as "off".
inline constexpr uint16_t kVirtualLdsKnownFlagsMask =
    kVirtualLdsFlagRuntimeStateBlock | kVirtualLdsFlagWorkgroupIdX | kVirtualLdsFlagWorkgroupIdY |
    kVirtualLdsFlagWorkgroupIdZ;

/// @brief One kernel's static virtual-LDS selection and resource facts.
///
/// @details Descriptor addresses live in generic sidecar metadata and kernarg
/// wrapper layouts live in kernarg-extension metadata. This record contains
/// only facts owned by virtual LDS, plus the sidecar name that joins the three
/// independent mechanisms at runtime.
struct VirtualLdsKernelMetadata {
  std::string kernel_name;
  std::string sidecar_variant_name;
  uint32_t static_lds_bytes = 0;
  uint32_t normal_private_segment_size = 0;
  uint32_t virtual_private_segment_size = 0;
  uint16_t virtual_lds_base_sgpr = 0;
  uint16_t flags = 0;
};

/// @brief Serialize virtual-LDS records for @ref kVirtualLdsMetadataSectionName.
[[nodiscard]] std::vector<uint8_t>
serialize_virtual_lds_metadata(std::span<const VirtualLdsKernelMetadata> kernels);

/// @brief Parse a @ref kVirtualLdsMetadataSectionName payload.
[[nodiscard]] std::optional<std::vector<VirtualLdsKernelMetadata>>
parse_virtual_lds_metadata(std::span<const uint8_t> bytes);

/// @brief Delivery-independent AQL fields needed to plan a virtual-LDS rewrite.
struct VirtualLdsDispatchPacketFields {
  uint32_t group_segment_size = 0;
  uint32_t private_segment_size = 0;
  uint16_t workgroup_size_x = 0;
  uint16_t workgroup_size_y = 0;
  uint16_t workgroup_size_z = 0;
  uint32_t grid_size_x = 0;
  uint32_t grid_size_y = 0;
  uint32_t grid_size_z = 0;
};

/// @brief Static descriptor and ABI facts needed by dispatch rewriting.
struct VirtualLdsDispatchKernelFacts {
  uint32_t static_lds_bytes = 0;
  uint32_t normal_private_segment_size = 0;
  uint32_t virtual_private_segment_size = 0;
  bool has_workgroup_id_x = false;
  bool has_workgroup_id_y = false;
  bool has_workgroup_id_z = false;
};

/// @brief Dense backing allocation and runtime strides for one dispatch.
struct VirtualLdsDispatchGeometry {
  uint32_t groups_x = 0;
  uint32_t groups_y = 0;
  uint32_t groups_z = 0;
  uint32_t stride_x = 0;
  uint32_t stride_y = 0;
  uint32_t stride_z = 0;
  size_t backing_bytes = 0;
};

enum class VirtualLdsDispatchDecision { KeepNormal, UseSidecar, Reject };

/// @brief Complete side-effect-free plan consumed by HSA delivery adapters.
struct VirtualLdsDispatchRewrite {
  VirtualLdsDispatchDecision decision = VirtualLdsDispatchDecision::KeepNormal;
  std::string diagnostic;
  uint64_t requested_lds = 0;
  uint32_t private_segment_size = 0;
  VirtualLdsDispatchGeometry geometry;
};

/// @brief Plan a virtual-LDS rewrite without allocating or publishing anything.
[[nodiscard]] VirtualLdsDispatchRewrite
plan_virtual_lds_dispatch(const VirtualLdsDispatchPacketFields &packet,
                          const VirtualLdsDispatchKernelFacts &kernel, uint32_t host_lds_bytes);

} // namespace rocjitsu
