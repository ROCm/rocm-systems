// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kernarg_extension.h
/// @brief Helpers for building rocjitsu-owned kernarg wrapper buffers.

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace rocjitsu {

/// @brief One DBT/DBI payload appended to a kernarg wrapper.
struct KernargExtensionPayloadLayout {
  uint32_t size = 0;
  uint32_t alignment = 8;
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

} // namespace rocjitsu
