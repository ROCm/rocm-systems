// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file amdgpu_kernel_metadata.h
/// @brief Internal payload-level AMDGPU kernel metadata parser interface.

#ifndef ROCJITSU_CODE_AMDGPU_KERNEL_METADATA_H_
#define ROCJITSU_CODE_AMDGPU_KERNEL_METADATA_H_

#include <array>
#include <concepts>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>

namespace rocjitsu::amdgpu_code_object_detail {

inline constexpr unsigned kMaximumKernelMetadataNestingDepth = 64;

struct KernelMetadata {
  bool has_dynamic_lds = false;
  std::optional<bool> uses_dynamic_stack;
  std::optional<uint16_t> sgpr_count;
  std::optional<std::array<uint32_t, 3>> required_workgroup_size;
};

enum class KernelMetadataVisitStatus : uint8_t {
  Complete,
  Malformed,
  VisitorRejected,
};

using KernelMetadataVisitorCallback = bool (*)(const void *, std::string_view,
                                               const KernelMetadata &);

/// Type-erased implementation for the allocation-free visitor wrapper below.
[[nodiscard]] KernelMetadataVisitStatus
visit_kernel_metadata_payload_erased(std::span<const uint8_t> payload, const void *context,
                                     KernelMetadataVisitorCallback visitor);

/// Visit retained fields from each named kernel in one msgpack metadata payload.
///
/// The callback is non-owning and invoked synchronously. Kernels without
/// retained fields are not visited. After the first callback refusal, the
/// remaining kernel array is structurally validated without further callbacks,
/// so malformation takes precedence over refusal. Root entries and bytes after
/// the first `amdhsa.kernels` array are ignored for forwards compatibility,
/// matching the code-object loader's historical behavior.
template <typename Visitor>
  requires std::invocable<Visitor &, std::string_view, const KernelMetadata &> &&
           std::convertible_to<
               std::invoke_result_t<Visitor &, std::string_view, const KernelMetadata &>, bool>
[[nodiscard]] KernelMetadataVisitStatus
visit_kernel_metadata_payload(std::span<const uint8_t> payload, Visitor &&visitor) {
  auto visitor_reference = std::ref(visitor);
  using VisitorReference = decltype(visitor_reference);
  return visit_kernel_metadata_payload_erased(
      payload, std::addressof(visitor_reference),
      [](const void *context, std::string_view name, const KernelMetadata &metadata) {
        return static_cast<bool>(
            std::invoke(*static_cast<const VisitorReference *>(context), name, metadata));
      });
}

} // namespace rocjitsu::amdgpu_code_object_detail

#endif // ROCJITSU_CODE_AMDGPU_KERNEL_METADATA_H_
