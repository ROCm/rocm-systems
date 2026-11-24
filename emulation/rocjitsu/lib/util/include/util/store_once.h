// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file store_once.h
/// @brief Copy into memory that others update atomically, storing each byte once.

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace util {

namespace detail {

/// @brief Width in bytes of the next piece store_once() stores at @p address.
/// @details The widest of 1, 2, 4 and 8 bytes that both fits @p remaining,
/// which must be nonzero, and is naturally aligned at @p address.
constexpr std::size_t store_piece_width(std::uintptr_t address, std::size_t remaining) {
  // Or-ing in the widest piece caps the alignment found at eight bytes.
  const std::size_t alignment = std::size_t{1} << std::countr_zero(address | sizeof(uint64_t));
  return std::min(alignment, std::bit_floor(remaining));
}

/// @brief Store one @p WordType with a single relaxed atomic store.
/// @details The destination must be aligned for @p WordType. A relaxed atomic
/// store is one machine store, is defined against concurrent std::atomic_ref
/// users of the same word, and is never folded back into a memcpy call.
template <typename WordType>
inline void store_word_once(std::byte *destination, const std::byte *source) {
  static_assert(std::atomic_ref<WordType>::is_always_lock_free,
                "a locked fallback would not exclude writers in another process");
  WordType word;
  std::memcpy(&word, source, sizeof(word));
  std::atomic_ref<WordType>(*reinterpret_cast<WordType *>(destination))
      .store(word, std::memory_order_relaxed);
}

} // namespace detail

/// @brief Copy @p length bytes, storing every destination byte exactly once.
///
/// @details For memory that another thread or process may update with atomic
/// operations while it is being written, such as guest RAM or a device
/// aperture shared by descriptor. libc memcpy is free to store an overlapping
/// span twice, and glibc on x86-64 does so for 4- and 8-byte copies, so a
/// compare-exchange landing between its two stores is silently undone.
///
/// The destination is written in ascending address order as 1-, 2-, 4- and
/// 8-byte pieces, each the widest that both fits the bytes remaining and is
/// naturally aligned at its address. A 2-, 4- or 8-byte copy to a destination
/// aligned for that width is therefore one atomic store; a misaligned one is
/// split into narrower aligned stores and is not atomic as a whole.
///
/// The ranges must not overlap. A zero @p length reads and writes nothing, so
/// both pointers may then be null.
///
/// @param[out] destination First byte to store.
/// @param[in] source First byte to copy.
/// @param[in] length Number of bytes to copy.
inline void store_once(std::byte *destination, const std::byte *source, std::size_t length) {
  while (length != 0) {
    const std::size_t width =
        detail::store_piece_width(reinterpret_cast<std::uintptr_t>(destination), length);
    if (width == sizeof(uint64_t)) {
      // The destination stays eight-byte aligned across this run, so whole
      // words need no further width selection.
      for (; length >= sizeof(uint64_t); length -= sizeof(uint64_t)) {
        detail::store_word_once<uint64_t>(destination, source);
        destination += sizeof(uint64_t);
        source += sizeof(uint64_t);
      }
      continue;
    }
    if (width == sizeof(uint32_t)) {
      detail::store_word_once<uint32_t>(destination, source);
    } else if (width == sizeof(uint16_t)) {
      detail::store_word_once<uint16_t>(destination, source);
    } else {
      detail::store_word_once<uint8_t>(destination, source);
    }
    destination += width;
    source += width;
    length -= width;
  }
}

} // namespace util
