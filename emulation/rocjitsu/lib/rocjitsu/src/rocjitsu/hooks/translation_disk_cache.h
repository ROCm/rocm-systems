// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hooks/translation_disk_cache.h
/// @brief Session-scoped store for translated gfx1250 code objects.
///
/// @details Translation is deterministic for a given input and profile, so a
/// process can reuse an object another process already produced. The store lives
/// under the per-user runtime directory, which the system clears on logout and
/// reboot, so entries never outlive a boot session; nothing here is durable
/// storage and no entry can survive into a differently-built runtime.
///
/// Every operation is best effort. A lookup that cannot be satisfied and a store
/// that cannot be completed both leave the caller to translate normally, so an
/// unusable store degrades throughput and never correctness.

#ifndef ROCJITSU_HOOKS_TRANSLATION_DISK_CACHE_H_
#define ROCJITSU_HOOKS_TRANSLATION_DISK_CACHE_H_

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu::hotswap {

/// @brief Everything besides the source object that determines the output.
///
/// @details These are folded into the key so an entry produced under one
/// configuration can never satisfy a request made under another.
struct TranslationIdentity {
  uint32_t profile_id = 0;
  uint32_t input_revision = 0;
  uint32_t output_revision = 0;
  std::string_view target_isa;
};

/// @brief Digest identifying one translation request.
/// @details `valid` is false when the store is unavailable, in which case the
/// lookup and store entry points do nothing.
struct CacheKey {
  std::array<uint8_t, 32> digest{};
  bool valid = false;
};

/// @brief Derive the key for @p source under @p identity.
///
/// @details The digest covers the source bytes, the identity fields, and the
/// build identity of this library, so a rebuilt translator cannot match an entry
/// produced by the previous one. Returns an invalid key when the store is
/// disabled, which lets callers skip both the lookup and the store.
[[nodiscard]] CacheKey cache_key_for(std::span<const uint8_t> source,
                                     const TranslationIdentity &identity);

/// @brief Fetch a stored translation.
///
/// @details @p identity is re-checked against the record stored alongside the
/// object, so an entry can only satisfy a request made under the same
/// configuration even if the digest were somehow to match.
/// @returns The translated object, or an empty vector on a miss or any failure.
[[nodiscard]] std::vector<uint8_t> cache_lookup(const CacheKey &key,
                                                const TranslationIdentity &identity);

/// @brief Offer a translation for reuse. Never throws.
void cache_store(const CacheKey &key, std::span<const uint8_t> translated,
                 const TranslationIdentity &identity);

#if defined(RJ_HOTSWAP_TEST_HOOKS)
/// @brief Test-only: point the store at @p root and re-run its checks.
/// @details Passing nullptr restores the default location. Tests need this
/// because the production path is derived entirely from the runtime directory
/// and takes no configuration.
void set_cache_root_for_test(const char *root);

/// @brief Test-only: bytes currently occupied by stored objects.
[[nodiscard]] uint64_t cache_size_for_test();

/// @brief Test-only: lookups satisfied since the last root change.
/// @details Without this a test can only observe that two loads agree, which is
/// also true when both translated. This distinguishes reuse from repetition.
[[nodiscard]] uint64_t cache_hits_for_test();

/// @brief Test-only: override the byte cap so eviction can be exercised.
void set_cache_capacity_for_test(uint64_t bytes);

/// @brief Test-only: override the free-space floor below which writes stop.
/// @details Raising it above the real free space is the only way to reach the
/// low-space behaviour without filling the filesystem.
void set_cache_headroom_for_test(uint64_t bytes);
#endif // RJ_HOTSWAP_TEST_HOOKS

} // namespace rocjitsu::hotswap

#endif // ROCJITSU_HOOKS_TRANSLATION_DISK_CACHE_H_
