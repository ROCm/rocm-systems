// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kernarg_extension.h
/// @brief Helpers for building rocjitsu-owned kernarg wrapper buffers.

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

/// @brief Non-allocated ELF section describing kernarg extension layouts.
inline constexpr std::string_view kKernargExtensionMetadataSectionName = ".rocjitsu.kernarg";

/// @brief One DBT/DBI payload appended to a kernarg wrapper.
struct KernargExtensionPayloadLayout {
  uint32_t size = 0;
  uint32_t alignment = 8;
};

/// @brief A named payload in a serialized kernarg extension contract.
///
/// @details Payload names let an independent feature locate the extension it
/// owns without making the generic kernarg mechanism depend on that feature's
/// C++ type or selection policy.
struct KernargExtensionPayloadMetadata {
  uint32_t size = 0;
  uint32_t alignment = 8;
  std::string name;
};

/// @brief Kernarg extension contract used by one named sidecar variant.
struct KernargExtensionMetadata {
  std::string kernel_name;
  std::string variant_name;
  uint32_t original_kernarg_size = 0;
  std::vector<KernargExtensionPayloadMetadata> payloads;
};

/// @brief Concrete byte layout for a kernarg wrapper.
///
/// @details The wrapper starts with a byte-for-byte copy of the original
/// kernarg image. That invariant keeps CP kernarg preloads and any source
/// kernarg offsets valid while the entry prologue still sees the wrapper
/// pointer. After the copied prefix, rocjitsu stores the original kernarg
/// pointer followed by one or more aligned extension payloads.
struct KernargExtensionLayout {
  uint32_t original_kernarg_size = 0;
  uint32_t original_kernarg_pointer_offset = 0;
  std::vector<uint32_t> payload_offsets;
  uint32_t wrapper_size = 0;
};

/// @brief Runtime payload bytes written into a concrete wrapper.
struct KernargExtensionPayloadWrite {
  const void *data = nullptr;
  uint32_t size = 0;
};

/// @brief Compute a wrapper layout for @p payloads.
///
/// @returns std::nullopt if the layout would overflow 32-bit descriptor sizes
/// or if an alignment is not a power of two.
[[nodiscard]] std::optional<KernargExtensionLayout>
make_kernarg_extension_layout(uint32_t original_kernarg_size,
                              std::span<const KernargExtensionPayloadLayout> payloads);

/// @brief Fill a wrapper buffer according to @p layout.
///
/// @details The caller owns the wrapper lifetime. `original_kernarg` may be
/// null only when `layout.original_kernarg_size == 0`; the original pointer
/// value is still recorded so the translated prologue can restore the
/// guest-visible kernarg segment pointer before original code executes.
[[nodiscard]] bool
write_kernarg_extension_wrapper(std::span<uint8_t> wrapper, const KernargExtensionLayout &layout,
                                const void *original_kernarg, uint64_t original_kernarg_pointer,
                                std::span<const KernargExtensionPayloadWrite> payloads);

/// @brief Serialize independent kernarg extension contracts into an ELF payload.
[[nodiscard]] std::vector<uint8_t>
serialize_kernarg_extension_metadata(std::span<const KernargExtensionMetadata> extensions);

/// @brief Parse a @ref kKernargExtensionMetadataSectionName payload.
[[nodiscard]] std::optional<std::vector<KernargExtensionMetadata>>
parse_kernarg_extension_metadata(std::span<const uint8_t> bytes);

} // namespace rocjitsu
