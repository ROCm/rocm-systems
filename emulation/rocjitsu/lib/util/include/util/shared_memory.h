// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file shared_memory.h
/// @brief Loads, stores and compare/exchanges on bytes that other threads and
///        another process access at the same time, at widths of their choosing.
///
/// @details Guest RAM and the video memory aperture are mapped by descriptor
/// into this process and into the VMM. Device threads here and guest
/// processors there read, write and compare/exchange the same bytes with no
/// common lock and no agreed width: a guest may exchange a 64-bit word whose
/// halves a device thread loads as two 32-bit words.
///
/// Every access to such bytes goes through this header, and every access this
/// header makes is an aligned `__atomic` load, store or compare/exchange of 1,
/// 2, 4 or 8 bytes. What a caller may rely on:
///
/// - a copy in stores each shared byte once, and a copy out loads each once;
/// - a 2-, 4- or 8-byte access at an address aligned for its width is a single
///   machine access, indivisible against every access that overlaps it;
/// - any other span is several such accesses and is not indivisible as a whole.
///
/// What this rests on. It is not ISO C++, and no language-level model defines
/// it:
///
/// - ISO C++ has no object model for bytes that are accessed atomically at
///   several widths. A std::atomic_ref<T> refers to an object of type T, which
///   may be accessed only through atomic_ref while one exists
///   ([atomics.ref.generic]), and the same bytes cannot be one 64-bit object
///   and two 32-bit objects at once. std::atomic_ref is therefore not used.
/// - LLVM's memory model does not define it either. Two accesses that are not
///   ordered by happens-before form a data race there if "they do not access
///   the exact same set of bytes", atomic or not, and "overlapping atomic
///   accesses that are not ordered by happens-before must be perfectly
///   overlapping to act atomically" (LLVM Language Reference, "Memory Model
///   for Concurrent Operations" and "Atomic Memory Ordering Constraints").
///   What the model does say is that a load which races returns undef unless
///   the load is volatile, in which case "the result is target-dependent".
///   Every access here is volatile for that reason: it leaves the value
///   loaded to the processor rather than to the optimizer.
///
/// What it does rest on:
///
/// - The word types below carry `may_alias`, which GCC documents as exempting
///   accesses through them from type-based aliasing, as accesses through a
///   character type are (GCC manual, "Common Type Attributes"). So no access
///   here is an aliasing violation, whatever objects the bytes hold. They are
///   also volatile, and LLVM neither removes, repeats, splits nor merges a
///   volatile access of a width the target supports (LLVM Language Reference,
///   "Volatile Memory Accesses").
/// - Each `__atomic` built-in used here is of a size the target makes
///   lock-free (asserted below), so it involves no lock that a writer in
///   another process would not take. That the compiler emits it as one load,
///   store or locked exchange instruction on exactly those bytes is how GCC
///   and Clang generate code for these built-ins; it is relied on, not
///   promised by either.
/// - The indivisibility of each such instruction against overlapping accesses
///   of other widths is a property of the processor, not of the language. On
///   x86-64 a load or store of at most eight bytes at a naturally aligned
///   address is atomic, and a locked instruction excludes every other access
///   to the bytes it covers (Intel SDM volume 3A, "Guaranteed Atomic
///   Operations" and "Bus Locking"; the AMD64 Architecture Programmer's
///   Manual, volume 2, on access atomicity).
///
/// x86-64 is the only processor this has been established and tested for, so
/// the header refuses to compile for any other. A port has to establish the
/// last property for its processor before it lifts that check.
///
/// A plain access (memcpy, std::copy, a dereference) to bytes that are also
/// reached through this header is a data race in every model, and a libc
/// memcpy may in addition store a byte twice.

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if !defined(__GNUC__) && !defined(__clang__)
#error "util/shared_memory.h needs the GCC/Clang __atomic built-ins and may_alias"
#endif

#ifdef __x86_64__

namespace util {

/// @brief A width at which shared bytes can be loaded, stored and exchanged.
template <typename WordType>
concept SharedWord = std::same_as<WordType, uint8_t> || std::same_as<WordType, uint16_t> ||
                     std::same_as<WordType, uint32_t> || std::same_as<WordType, uint64_t>;

namespace detail {

// Concrete typedefs: GCC ignores may_alias on a dependent type.
typedef volatile uint8_t __attribute__((__may_alias__)) SharedU8;
typedef volatile uint16_t __attribute__((__may_alias__)) SharedU16;
typedef volatile uint32_t __attribute__((__may_alias__)) SharedU32;
typedef volatile uint64_t __attribute__((__may_alias__)) SharedU64;

template <typename WordType> struct SharedAlias;
template <> struct SharedAlias<uint8_t> {
  using Type = SharedU8;
};
template <> struct SharedAlias<uint16_t> {
  using Type = SharedU16;
};
template <> struct SharedAlias<uint32_t> {
  using Type = SharedU32;
};
template <> struct SharedAlias<uint64_t> {
  using Type = SharedU64;
};

// A locked fallback would not exclude a writer in another process.
static_assert(__atomic_always_lock_free(sizeof(uint8_t), nullptr));
static_assert(__atomic_always_lock_free(sizeof(uint16_t), nullptr));
static_assert(__atomic_always_lock_free(sizeof(uint32_t), nullptr));
static_assert(__atomic_always_lock_free(sizeof(uint64_t), nullptr));

template <typename WordType>
[[nodiscard]] inline typename SharedAlias<WordType>::Type *shared_word(std::byte *shared) {
  return reinterpret_cast<typename SharedAlias<WordType>::Type *>(shared);
}

template <typename WordType>
[[nodiscard]] inline const typename SharedAlias<WordType>::Type *
shared_word(const std::byte *shared) {
  return reinterpret_cast<const typename SharedAlias<WordType>::Type *>(shared);
}

[[nodiscard]] inline bool is_aligned_for(const std::byte *shared, std::size_t width) {
  return reinterpret_cast<std::uintptr_t>(shared) % width == 0;
}

/// @brief Width in bytes of the next piece a copy accesses at @p shared_address.
/// @details The widest of 1, 2, 4 and 8 bytes that both fits @p remaining,
/// which must be nonzero, and is naturally aligned at @p shared_address.
constexpr std::size_t shared_piece_width(std::uintptr_t shared_address, std::size_t remaining) {
  // Or-ing in the widest piece caps the alignment found at eight bytes.
  const std::size_t alignment = std::size_t{1}
                                << std::countr_zero(shared_address | sizeof(uint64_t));
  const std::size_t fitting = std::bit_floor(remaining);
  return alignment < fitting ? alignment : fitting;
}

template <typename WordType>
inline void store_piece_once(std::byte *shared_destination, const std::byte *source) {
  WordType word;
  std::memcpy(&word, source, sizeof(word));
  __atomic_store_n(shared_word<WordType>(shared_destination), word, __ATOMIC_RELAXED);
}

template <typename WordType>
inline void load_piece_once(std::byte *destination, const std::byte *shared_source) {
  const WordType word = __atomic_load_n(shared_word<WordType>(shared_source), __ATOMIC_RELAXED);
  std::memcpy(destination, &word, sizeof(word));
}

/// @brief Stop on a width or address that no shared word can have.
/// @details In every build type: carrying on would access bytes the caller did
/// not name, or split an access the caller was promised is indivisible.
[[noreturn]] inline void reject_shared_word() noexcept { std::abort(); }

inline void require_aligned_for(const std::byte *shared, std::size_t width) noexcept {
  if (!is_aligned_for(shared, width)) [[unlikely]] {
    reject_shared_word();
  }
}

} // namespace detail

/// @brief Copy @p length bytes into shared memory, storing each byte once.
///
/// @details libc memcpy is free to store an overlapping span twice, and glibc
/// on x86-64 does so for most short lengths, so a compare/exchange that lands
/// between its two stores is silently undone.
///
/// The destination is written in ascending address order as 1-, 2-, 4- and
/// 8-byte relaxed atomic stores, each the widest that both fits the bytes
/// remaining and is naturally aligned at its address. A 2-, 4- or 8-byte copy
/// to a destination aligned for that width is therefore one store; any other
/// copy is several, and is not indivisible as a whole. A piece may be wider
/// than the words other users access: one 64-bit store may cover two words
/// that others exchange as 32 bits each. The file comment says what makes
/// that sound.
///
/// The ranges must not overlap, and @p source must not itself be shared. A
/// zero @p length reads and writes nothing, so both pointers may then be null.
///
/// @param[out] shared_destination First shared byte to store.
/// @param[in] source First byte to copy.
/// @param[in] length Number of bytes to copy.
inline void store_once(std::byte *shared_destination, const std::byte *source, std::size_t length) {
  // One aligned word is one store, and the commonest copy: skip the width
  // selection that a fixed length would otherwise still pay for.
  if (length == sizeof(uint64_t) && detail::is_aligned_for(shared_destination, sizeof(uint64_t))) {
    return detail::store_piece_once<uint64_t>(shared_destination, source);
  }
  if (length == sizeof(uint32_t) && detail::is_aligned_for(shared_destination, sizeof(uint32_t))) {
    return detail::store_piece_once<uint32_t>(shared_destination, source);
  }
  while (length != 0) {
    const std::size_t width =
        detail::shared_piece_width(reinterpret_cast<std::uintptr_t>(shared_destination), length);
    if (width == sizeof(uint64_t)) {
      // The destination stays eight-byte aligned across this run, so whole
      // words need no further width selection.
      for (; length >= sizeof(uint64_t); length -= sizeof(uint64_t)) {
        detail::store_piece_once<uint64_t>(shared_destination, source);
        shared_destination += sizeof(uint64_t);
        source += sizeof(uint64_t);
      }
      continue;
    }
    if (width == sizeof(uint32_t)) {
      detail::store_piece_once<uint32_t>(shared_destination, source);
    } else if (width == sizeof(uint16_t)) {
      detail::store_piece_once<uint16_t>(shared_destination, source);
    } else {
      detail::store_piece_once<uint8_t>(shared_destination, source);
    }
    shared_destination += width;
    source += width;
    length -= width;
  }
}

/// @brief Copy @p length bytes out of shared memory, loading each byte once.
///
/// @details The mirror of store_once(): the source is read in ascending
/// address order as 1-, 2-, 4- and 8-byte relaxed atomic loads chosen by the
/// same rule, applied to the source address. A 2-, 4- or 8-byte copy from a
/// source aligned for that width is one load, and so never observes part of a
/// store or exchange of that word. Any other copy may return some pieces from
/// before a concurrent update and some from after it.
///
/// libc memcpy is not a substitute: its loads are plain, so one that overlaps
/// a concurrent store is a data race, and the compiler may load the bytes
/// again after the caller has checked them.
///
/// The ranges must not overlap, and @p destination must not itself be shared.
/// A zero @p length reads and writes nothing, so both pointers may then be
/// null.
///
/// @param[out] destination First byte to fill.
/// @param[in] shared_source First shared byte to load.
/// @param[in] length Number of bytes to copy.
inline void load_once(std::byte *destination, const std::byte *shared_source, std::size_t length) {
  if (length == sizeof(uint64_t) && detail::is_aligned_for(shared_source, sizeof(uint64_t))) {
    return detail::load_piece_once<uint64_t>(destination, shared_source);
  }
  if (length == sizeof(uint32_t) && detail::is_aligned_for(shared_source, sizeof(uint32_t))) {
    return detail::load_piece_once<uint32_t>(destination, shared_source);
  }
  while (length != 0) {
    const std::size_t width =
        detail::shared_piece_width(reinterpret_cast<std::uintptr_t>(shared_source), length);
    if (width == sizeof(uint64_t)) {
      for (; length >= sizeof(uint64_t); length -= sizeof(uint64_t)) {
        detail::load_piece_once<uint64_t>(destination, shared_source);
        destination += sizeof(uint64_t);
        shared_source += sizeof(uint64_t);
      }
      continue;
    }
    if (width == sizeof(uint32_t)) {
      detail::load_piece_once<uint32_t>(destination, shared_source);
    } else if (width == sizeof(uint16_t)) {
      detail::load_piece_once<uint16_t>(destination, shared_source);
    } else {
      detail::load_piece_once<uint8_t>(destination, shared_source);
    }
    destination += width;
    shared_source += width;
    length -= width;
  }
}

/// @brief Load one word of shared memory with acquire ordering.
/// @details Like the two operations that follow, terminates the process, in
/// every build type, when @p shared is not aligned for @p WordType: the access
/// could then cross a cache line and would not be indivisible.
/// @param[in] shared Address of the word, which must be aligned for
///            @p WordType.
template <SharedWord WordType> [[nodiscard]] inline WordType shared_load(const std::byte *shared) {
  detail::require_aligned_for(shared, sizeof(WordType));
  return __atomic_load_n(detail::shared_word<WordType>(shared), __ATOMIC_ACQUIRE);
}

/// @brief Store one word of shared memory with release ordering.
/// @param[out] shared Address of the word, which must be aligned for
///             @p WordType.
/// @param[in] value Word to store.
template <SharedWord WordType> inline void shared_store(std::byte *shared, WordType value) {
  detail::require_aligned_for(shared, sizeof(WordType));
  __atomic_store_n(detail::shared_word<WordType>(shared), value, __ATOMIC_RELEASE);
}

/// @brief Replace one word of shared memory if it holds @p expected.
/// @param[in,out] shared Address of the word, which must be aligned for
///                @p WordType.
/// @param[in,out] expected Value required; on return, the value found.
/// @param[in] desired Value to store when the word holds @p expected.
/// @return True when the word was replaced. Never fails spuriously.
template <SharedWord WordType>
[[nodiscard]] inline bool shared_compare_exchange(std::byte *shared, WordType &expected,
                                                  WordType desired) {
  detail::require_aligned_for(shared, sizeof(WordType));
  return __atomic_compare_exchange_n(detail::shared_word<WordType>(shared), &expected, desired,
                                     /*weak=*/false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

/// @brief Whether @p shared can hold one shared word of @p width bytes.
/// @details True for a width of 1, 2, 4 or 8 at an address aligned for it,
/// which is what the run-time-width operations below require.
[[nodiscard]] inline bool is_shared_word(const std::byte *shared, std::size_t width) {
  return (width == sizeof(uint8_t) || width == sizeof(uint16_t) || width == sizeof(uint32_t) ||
          width == sizeof(uint64_t)) &&
         detail::is_aligned_for(shared, width);
}

/// @brief shared_load() of a word whose width is known only at run time.
/// @details Terminates the process, in every build type, unless
/// is_shared_word() holds for the arguments. Callers validate a width or
/// address that a guest chose before they get here.
/// @param[in] shared Address of the word, aligned for @p width.
/// @param[in] width 1, 2, 4 or 8.
/// @return The word, zero-extended.
[[nodiscard]] inline uint64_t shared_load_word(const std::byte *shared, std::size_t width) {
  if (is_shared_word(shared, width)) [[likely]] {
    switch (width) {
    case sizeof(uint64_t):
      return shared_load<uint64_t>(shared);
    case sizeof(uint32_t):
      return shared_load<uint32_t>(shared);
    case sizeof(uint16_t):
      return shared_load<uint16_t>(shared);
    case sizeof(uint8_t):
      return shared_load<uint8_t>(shared);
    }
  }
  detail::reject_shared_word();
}

/// @brief shared_store() of the low @p width bytes of @p value.
/// @details Terminates the process, in every build type, unless
/// is_shared_word() holds for the arguments.
/// @param[out] shared Address of the word, aligned for @p width.
/// @param[in] width 1, 2, 4 or 8.
/// @param[in] value Word to store; bytes above @p width are ignored.
inline void shared_store_word(std::byte *shared, std::size_t width, uint64_t value) {
  if (is_shared_word(shared, width)) [[likely]] {
    switch (width) {
    case sizeof(uint64_t):
      return shared_store<uint64_t>(shared, value);
    case sizeof(uint32_t):
      return shared_store<uint32_t>(shared, static_cast<uint32_t>(value));
    case sizeof(uint16_t):
      return shared_store<uint16_t>(shared, static_cast<uint16_t>(value));
    case sizeof(uint8_t):
      return shared_store<uint8_t>(shared, static_cast<uint8_t>(value));
    }
  }
  detail::reject_shared_word();
}

/// @brief shared_compare_exchange() on the low @p width bytes of its operands.
/// @details Terminates the process, in every build type, unless
/// is_shared_word() holds for the arguments.
/// @param[in,out] shared Address of the word, aligned for @p width.
/// @param[in] width 1, 2, 4 or 8.
/// @param[in,out] expected Value required, of which bytes above @p width are
///                ignored; on return, the value found, zero-extended.
/// @param[in] desired Value to store when the word holds @p expected; bytes
///            above @p width are ignored.
/// @return True when the word was replaced. Never fails spuriously.
[[nodiscard]] inline bool shared_compare_exchange_word(std::byte *shared, std::size_t width,
                                                       uint64_t &expected, uint64_t desired) {
  const auto exchange = [&]<SharedWord WordType>(WordType) {
    WordType found = static_cast<WordType>(expected);
    const bool exchanged =
        shared_compare_exchange<WordType>(shared, found, static_cast<WordType>(desired));
    expected = found;
    return exchanged;
  };
  if (is_shared_word(shared, width)) [[likely]] {
    switch (width) {
    case sizeof(uint64_t):
      return exchange(uint64_t{});
    case sizeof(uint32_t):
      return exchange(uint32_t{});
    case sizeof(uint16_t):
      return exchange(uint16_t{});
    case sizeof(uint8_t):
      return exchange(uint8_t{});
    }
  }
  detail::reject_shared_word();
}

} // namespace util

#else // !defined(__x86_64__)

// Stub implementations for non-x86-64.  The VRAM aperture is never mapped on
// these hosts (vram_store.cpp gates mmap() on __x86_64__), so the stubs are
// dead code but must compile to let the locked pread/pwrite path build on
// every Linux target supported by TheRock.
namespace util {

inline void store_once(std::byte *dst, const std::byte *src, std::size_t n) {
  std::memcpy(dst, src, n);
}

inline void load_once(std::byte *dst, const std::byte *src, std::size_t n) {
  std::memcpy(dst, src, n);
}

[[nodiscard]] inline bool is_shared_word(const std::byte *, std::size_t) { return false; }

[[nodiscard]] inline uint64_t shared_load_word(const std::byte *, std::size_t) { std::abort(); }

inline void shared_store_word(std::byte *, std::size_t, uint64_t) { std::abort(); }

[[nodiscard]] inline bool shared_compare_exchange_word(std::byte *, std::size_t, uint64_t &,
                                                       uint64_t) {
  std::abort();
}

} // namespace util

#endif // defined(__x86_64__)
