// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbt/translation_store.h
/// @brief Session-scoped store for translated code objects.
///
/// @details Translation is deterministic for a given input and configuration, so
/// a process can reuse an object another process already produced. The store
/// lives under the per-user runtime directory, which the system clears on logout
/// and reboot, so entries never outlive a boot session; nothing here is durable
/// storage and no entry can survive into a differently-built runtime.
///
/// Every operation is best effort. A lookup that cannot be satisfied and a store
/// that cannot be completed both leave the caller to translate normally, so an
/// unusable store degrades throughput and never correctness.
///
/// Nothing here is specific to one translation pair. A consumer names its own
/// domain, which selects an independent tree, and describes its configuration
/// through TranslationIdentity. Two consumers in one process never observe each
/// other's entries, because their domains differ -- but two PROGRAMS that call
/// the same translator do share entries, because the key names the translator
/// rather than the binary holding this code.

#ifndef ROCJITSU_CODE_DBT_TRANSLATION_STORE_H_
#define ROCJITSU_CODE_DBT_TRANSLATION_STORE_H_

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu {

/// @brief Everything besides the source object that determines the output.
///
/// @details These are folded into the key so an entry produced under one
/// configuration can never satisfy a request made under another. The meaning of
/// each numeric field is the consumer's to define; the store only requires that
/// a given consumer assigns them consistently.
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

/// @brief A domain's entries within the per-user runtime directory.
///
/// @details Construct one per translation pair and keep it for the process
/// lifetime; opening is deferred to first use and the verified directory
/// descriptor is then held rather than the path re-resolved, which is what
/// closes the path-swap window. Safe to use from several threads.
class TranslationStore {
public:
  /// @param domain Selects an independent tree, so two consumers in one process
  ///               never observe each other's entries. Must be a single
  ///               non-empty path component; anything else disables the store.
  /// @param translator Any address inside the translator that produces these
  ///               entries -- the address of one of its functions will do. The
  ///               build id of the module containing it becomes part of every
  ///               key, so entries name what PRODUCED them rather than whichever
  ///               binary was running. Two programs calling the same translator
  ///               therefore agree on keys and can reuse each other's work, which
  ///               is what lets a translation performed ahead of time be found at
  ///               runtime. Passing an address in the caller instead would key on
  ///               the caller, and two consumers of one translator would silently
  ///               keep separate caches of identical results.
  TranslationStore(std::string_view domain, const void *translator);
  ~TranslationStore();

  TranslationStore(const TranslationStore &) = delete;
  TranslationStore &operator=(const TranslationStore &) = delete;

  /// @brief Derive the key for @p source under @p identity.
  ///
  /// @details The digest covers the source bytes, the identity fields, and the
  /// build id of the translator named at construction, so a rebuilt translator
  /// cannot match an entry produced by the previous one. Returns an invalid key
  /// when the store is unavailable, which lets callers skip both the lookup and
  /// the store.
  [[nodiscard]] CacheKey key_for(std::span<const uint8_t> source,
                                 const TranslationIdentity &identity);

  /// @brief Fetch a stored translation.
  ///
  /// @details @p identity is re-checked against the record stored alongside the
  /// object, so an entry can only satisfy a request made under the same
  /// configuration even if the digest were somehow to match.
  /// @returns The translated object, or an empty vector on a miss or any failure.
  [[nodiscard]] std::vector<uint8_t> lookup(const CacheKey &key,
                                            const TranslationIdentity &identity);

  /// @brief Offer a translation for reuse. Never throws.
  void store(const CacheKey &key, std::span<const uint8_t> translated,
             const TranslationIdentity &identity);

#if defined(RJ_TRANSLATION_STORE_TEST_HOOKS)
  /// @brief Test-only: point the store at @p root and re-run its checks.
  /// @details Passing nullptr restores the default location. Tests need this
  /// because the production path is derived entirely from the runtime directory
  /// and takes no configuration.
  void set_root_for_test(const char *root);

  /// @brief Test-only: bytes currently occupied by stored objects.
  [[nodiscard]] uint64_t size_for_test();

  /// @brief Test-only: lookups satisfied since the last root change.
  /// @details Without this a test can only observe that two requests agree,
  /// which is also true when both translated. This distinguishes reuse from
  /// repetition.
  [[nodiscard]] uint64_t hits_for_test();

  /// @brief Test-only: override the byte cap so eviction can be exercised.
  void set_capacity_for_test(uint64_t bytes);

  /// @brief Test-only: override the free-space floor below which writes stop.
  /// @details Raising it above the real free space is the only way to reach the
  /// low-space behaviour without filling the filesystem.
  void set_headroom_for_test(uint64_t bytes);
#endif // RJ_TRANSLATION_STORE_TEST_HOOKS

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace rocjitsu

#endif // ROCJITSU_CODE_DBT_TRANSLATION_STORE_H_
