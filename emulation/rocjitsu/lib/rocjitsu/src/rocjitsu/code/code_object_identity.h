// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file code_object_identity.h
/// @brief Stable identity for exact code-object bytes.

#ifndef ROCJITSU_CODE_CODE_OBJECT_IDENTITY_H_
#define ROCJITSU_CODE_CODE_OBJECT_IDENTITY_H_

#include <cstddef>
#include <cstdint>

namespace rocjitsu {

/// Return the 64-bit FNV-1a identity of an exact code-object byte sequence.
///
/// This intentionally covers the full ELF image, including metadata, so an
/// offline-selected embedded object and the runtime-loaded object can be joined
/// without relying on a process-local reader handle or filename.
inline uint64_t stable_code_object_id(const void *data, size_t size) noexcept {
  constexpr uint64_t kOffsetBasis = 14695981039346656037ULL;
  constexpr uint64_t kPrime = 1099511628211ULL;

  uint64_t identity = kOffsetBasis;
  const auto *bytes = static_cast<const uint8_t *>(data);
  for (size_t index = 0; index < size; ++index) {
    identity ^= bytes[index];
    identity *= kPrime;
  }
  return identity;
}

} // namespace rocjitsu

#endif // ROCJITSU_CODE_CODE_OBJECT_IDENTITY_H_
