// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sidecar_registry.h
/// @brief Runtime resolution for generic sidecar kernel variants.

#pragma once

#include "rocjitsu/code/patch/sidecar_metadata.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocjitsu::hooks {

/// @brief Read a named non-allocated section from an AMDGPU ELF image.
///
/// @returns Empty bytes when the input is not an AMDGPU ELF or the section is
/// absent, and std::nullopt when an ELF section table is malformed.
[[nodiscard]] std::optional<std::vector<uint8_t>>
read_metadata_section(std::span<const uint8_t> image, std::string_view section_name);

/// @brief Parse generic sidecar records directly from an AMDGPU ELF image.
[[nodiscard]] std::optional<std::vector<SidecarVariantMetadata>>
parse_sidecar_metadata_section(std::span<const uint8_t> image);

/// @brief One loaded and address-resolved sidecar descriptor variant.
struct ResolvedSidecarVariant {
  std::string variant_name;
  uint64_t kernel_object = 0;
};

/// @brief Normal kernel object and every sidecar variant attached to it.
struct ResolvedSidecarKernel {
  uint64_t executable = 0;
  uint64_t load_id = 0;
  uint64_t symbol = 0;
  std::string kernel_name;
  uint64_t normal_kernel_object = 0;
  std::vector<ResolvedSidecarVariant> variants;

  /// @brief Find the loaded descriptor address for @p variant_name.
  [[nodiscard]] std::optional<uint64_t> variant_object(std::string_view variant_name) const;
};

/// @brief Process-local resolver for sidecar metadata and HSA symbol observations.
///
/// @details The registry uses integer handles deliberately: sidecar resolution
/// is independent of HSA headers and virtual LDS. An HSA hook, DBI adapter, or
/// test can feed load/symbol events and select variants using its own policy.
class SidecarRegistry {
public:
  static SidecarRegistry &instance();

  /// @brief Record sidecars found in one code object loaded into @p executable.
  void record_load(uint64_t executable, uint64_t load_id,
                   std::vector<SidecarVariantMetadata> metadata);

  /// @brief Associate a queried runtime symbol with its kernel name and sidecars.
  void record_symbol(uint64_t executable, std::string_view symbol_name, uint64_t symbol);

  /// @brief Resolve descriptor virtual addresses after the runtime reports the normal object.
  void note_kernel_object(uint64_t symbol, uint64_t kernel_object, uint32_t private_segment_size);

  /// @brief Find sidecars attached to a normal descriptor in an AQL packet.
  [[nodiscard]] std::optional<ResolvedSidecarKernel> find_by_kernel_object(uint64_t kernel_object);

  /// @brief Return the best-known symbol name for diagnostics.
  [[nodiscard]] std::optional<std::string> kernel_name_for_object(uint64_t kernel_object);

  /// @brief Return the runtime-reported fixed private segment size.
  [[nodiscard]] uint32_t private_segment_size_for_object(uint64_t kernel_object);

  void erase_executable(uint64_t executable);
  void clear();

private:
  struct LoadEntry {
    uint64_t executable = 0;
    uint64_t load_id = 0;
    std::vector<SidecarVariantMetadata> variants;
  };

  struct SymbolRecord {
    uint64_t executable = 0;
    std::string kernel_name;
    uint64_t kernel_object = 0;
    uint32_t private_segment_size = 0;
  };

  struct PendingResolvedKernel {
    ResolvedSidecarKernel resolved;
    uint64_t normal_descriptor_vaddr = 0;
    std::vector<uint64_t> variant_descriptor_vaddrs;
  };

  std::mutex mutex_;
  std::vector<LoadEntry> loads_;
  std::unordered_map<uint64_t, PendingResolvedKernel> sidecar_symbols_;
  std::unordered_map<uint64_t, SymbolRecord> known_symbols_;
  std::unordered_map<uint64_t, uint64_t> kernel_object_symbols_;
  std::unordered_map<uint64_t, uint64_t> sidecar_object_symbols_;
};

} // namespace rocjitsu::hooks
