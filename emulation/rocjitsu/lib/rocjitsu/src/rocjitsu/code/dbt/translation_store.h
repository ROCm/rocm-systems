// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbt/translation_store.h
/// @brief On-disk store for translated code objects.
///
/// @details Translation is deterministic for a given input and configuration, so
/// a process can reuse an object another process already produced.
///
/// The store lives at a caller-supplied root, is populated ahead of time by a
/// separate tool, and is read-only to the processes that benefit from it. It is
/// curated rather than cached: the tool decides what belongs in it, so nothing
/// is ever evicted to make room, and the size ceiling is a sanity bound rather
/// than a capacity policy. That is what lets it hold the objects worth caching
/// at all -- a device library of a few hundred megabytes takes minutes to
/// translate.
///
/// Every operation is best effort. A lookup that cannot be satisfied and a store
/// that cannot be completed both leave the caller to translate normally, so an
/// unusable store degrades throughput and never correctness.
///
/// Nothing here is specific to one translation pair. A consumer names its own
/// domain, which selects an independent tree, and describes its configuration
/// through TranslationIdentity. Two consumers in one process never observe each
/// other's entries, because their domains differ. The key can name either the
/// exact translator build or the portable-prebuilt compatibility contract. That
/// is what lets a translation performed ahead of time, by a tool that is not the
/// eventual reader, be found.

#ifndef ROCJITSU_CODE_DBT_TRANSLATION_STORE_H_
#define ROCJITSU_CODE_DBT_TRANSLATION_STORE_H_

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

/// @brief Where the shared tier lives, relative to the module holding @p address.
///
/// @details Resolves `<prefix>/share/rocjitsu/translations` from the install
/// location of whichever loaded object contains @p address, so an install can be
/// moved or built into a container at any prefix and still be found with nothing
/// configured.
///
/// Callers pass an address in the TRANSLATOR, the same one they pass to the
/// store. That is the point: the object whose build id decides what an entry is
/// keyed on also decides where entries live, so a tool and a runtime that agree
/// on the translator cannot disagree about either. Deriving the root from the
/// caller instead would let a tool in `bin` and a hook in `lib` compute
/// different directories while computing identical keys.
///
/// @returns The root, or an empty string when the module cannot be located.
///          Assumes a two-level layout (`<prefix>/lib`, `<prefix>/lib64`); a
///          multiarch library directory would resolve elsewhere and simply miss.
[[nodiscard]] std::string shared_translation_root(const void *address);

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

/// @brief One domain's entries within one tier.
///
/// @details Construct one per translation pair per tier and keep it for the
/// process lifetime; opening is deferred to first use and the verified directory
/// descriptor is then held rather than the path re-resolved, which is what
/// closes the path-swap window. Safe to use from several threads.
class TranslationStore {
public:
  /// @brief What a store is allowed to do.
  enum class Access {
    /// Lookups only. store() does nothing and no directory is created, so a
    /// runtime consulting an install-owned tree cannot alter it -- or create one
    /// where the installer made none.
    kReadOnly,
    /// The ahead-of-time tool's mode: creates the tree and writes entries.
    kReadWrite,
  };

  /// @brief What identifies the translator in keys and manifests.
  enum class KeyMode {
    /// Bind entries to the translator ELF's GNU build ID. This is the safe
    /// default for caches whose producer and consumer can be upgraded apart.
    kTranslatorBuild,
    /// Use the cache-format compatibility contract instead of an ELF build ID.
    /// Reserved for prebuilt entries shipped with the code that consumes them.
    kPortablePrebuilt,
  };

  /// @brief Open the store rooted at @p root.
  ///
  /// @details The tree is curated rather than cached: the ahead-of-time tool
  /// decides what belongs in it and a runtime only reads. An entry written here
  /// by one program is found by another that never ran the translator itself.
  ///
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
  /// @param root Directory holding the tree, typically derived from the caller's
  ///             own install location rather than configured. Never created when
  ///             @p access is kReadOnly.
  /// @param access Whether this instance may write. A runtime passes kReadOnly;
  ///             only the ahead-of-time tool passes kReadWrite.
  /// @param key_mode Whether entries bind to the exact translator build or to
  ///             the portable-prebuilt cache contract. Use portable mode only
  ///             when cache and consumer are distributed as one product.
  TranslationStore(std::string_view domain, const void *translator, std::string_view root,
                   Access access, KeyMode key_mode);

  ~TranslationStore();

  TranslationStore(const TranslationStore &) = delete;
  TranslationStore &operator=(const TranslationStore &) = delete;

  /// @brief Whether this store can serve requests at all.
  ///
  /// @details Opens the tree if it is not open yet. Callers that translate
  /// anyway on failure have no reason to ask; it exists so a tool whose entire
  /// purpose is to populate the store can report an unusable root instead of
  /// appearing to succeed while writing nothing.
  [[nodiscard]] bool available();

  /// @brief Derive the key for @p source under @p identity.
  ///
  /// @details The digest covers the source bytes, identity fields, and the
  /// translator identity selected at construction. Build-specific stores use
  /// the translator's GNU build ID. Portable-prebuilt stores use a reserved
  /// format nonce and rely on the cache epoch and translation identity as their
  /// compatibility boundary. Returns an invalid key when the store is
  /// unavailable, which lets callers skip both the lookup and the store.
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
  /// @details Does nothing on a read-only store, and nothing on any store whose
  /// tier refuses an object this large.
  void store(const CacheKey &key, std::span<const uint8_t> translated,
             const TranslationIdentity &identity) noexcept;

#if defined(RJ_TRANSLATION_STORE_TEST_HOOKS)
  /// @brief Test-only: point the store at @p root and re-run its checks.
  /// @details Passing nullptr restores the default location. Tests need this
  /// because both tiers derive their location from the environment -- the
  /// runtime directory for one, the install prefix for the other -- and neither
  /// takes configuration. The tier's own trust policy still applies to @p root,
  /// which is what lets a test show that a badly-owned directory is refused.
  void set_root_for_test(const char *root);

  /// @brief Test-only: bytes currently occupied by stored objects.
  [[nodiscard]] uint64_t size_for_test();

  /// @brief Test-only: lookups satisfied since the last root change.
  /// @details Without this a test can only observe that two requests agree,
  /// which is also true when both translated. This distinguishes reuse from
  /// repetition.
  [[nodiscard]] uint64_t hits_for_test();

  /// @brief Test-only: override the byte cap so eviction can be exercised.

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
