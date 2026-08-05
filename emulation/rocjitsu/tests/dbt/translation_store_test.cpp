// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file translation_store_test.cpp
/// @brief Covers TranslationStore away from any hook.
///
/// @details The store is meant to serve more than one translation pair, so this
/// compiles it as an ordinary DBT component and drives it directly. That also
/// keeps the header honest: anything it needed from a hook would fail to build
/// here.

#include <gtest/gtest.h>

#include "rocjitsu/code/dbt/translation_store.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <dlfcn.h>
#include <sys/stat.h>

namespace {

using rocjitsu::TranslationIdentity;
using rocjitsu::TranslationStore;

constexpr TranslationIdentity kIdentity{
    .profile_id = 7,
    .input_revision = 2,
    .output_revision = 1,
    .target_isa = "gfx-example",
};

constexpr auto kSafeDirPerms =
    std::filesystem::perms::owner_all | std::filesystem::perms::group_read |
    std::filesystem::perms::group_exec | std::filesystem::perms::others_read |
    std::filesystem::perms::others_exec;

const std::vector<uint8_t> kSource{'s', 'r', 'c'};
const std::vector<uint8_t> kObject{'o', 'u', 't', 'p', 'u', 't'};

/// Stands in for the translator: its address identifies this test binary, which
/// is the module every store constructed here shares.
void translator_identity_anchor() {}

class TranslationStoreTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::strcpy(root_, "/tmp/rj-translation-store-XXXXXX");
    ASSERT_NE(mkdtemp(root_), nullptr);
  }
  void TearDown() override { std::filesystem::remove_all(root_); }

  /// @brief A session-tier store rooted in this test's private directory.
  [[nodiscard]] std::unique_ptr<TranslationStore> open(std::string_view domain) {
    // Any address inside the translator would do in production; a test that is
    // not translating anything only needs the two stores it compares to agree.
    auto store = std::make_unique<TranslationStore>(
        domain, reinterpret_cast<const void *>(&translator_identity_anchor));
    store->set_root_for_test(root_);
    return store;
  }

  /// @brief A shared-tier store over @p path, or over this test's directory.
  [[nodiscard]] std::unique_ptr<TranslationStore> open_shared(std::string_view domain,
                                                              TranslationStore::Access access,
                                                              const char *path = nullptr) {
    auto store = std::make_unique<TranslationStore>(
        domain, reinterpret_cast<const void *>(&translator_identity_anchor),
        path == nullptr ? root_ : path, access);
    return store;
  }

  static void put(TranslationStore &store, std::span<const uint8_t> object) {
    store.store(store.key_for(kSource, kIdentity), object, kIdentity);
  }

  [[nodiscard]] static std::vector<uint8_t> get(TranslationStore &store) {
    return store.lookup(store.key_for(kSource, kIdentity), kIdentity);
  }

  char root_[64] = {};
};

TEST_F(TranslationStoreTest, StoredObjectComesBack) {
  auto store = open("pair-a");
  put(*store, kObject);
  EXPECT_EQ(get(*store), kObject);
  EXPECT_EQ(store->hits_for_test(), 1u);
}

TEST_F(TranslationStoreTest, DomainsDoNotShareEntries) {
  auto first = open("pair-a");
  auto second = open("pair-b");
  put(*first, kObject);

  EXPECT_EQ(get(*first), kObject);
  EXPECT_TRUE(get(*second).empty());

  // And the second domain can hold a different object under the same source.
  const std::vector<uint8_t> other{'e', 'l', 's', 'e'};
  put(*second, other);
  EXPECT_EQ(get(*second), other);
  EXPECT_EQ(get(*first), kObject);
}

TEST_F(TranslationStoreTest, ADomainThatIsNotOneComponentDisablesTheStore) {
  for (const char *domain : {"", ".", "..", "a/b", "../escape"}) {
    auto store = open(domain);
    put(*store, kObject);
    EXPECT_TRUE(get(*store).empty()) << "domain: " << domain;
    EXPECT_FALSE(store->key_for(kSource, kIdentity).valid) << "domain: " << domain;
  }
  // Nothing was created outside the domain directories.
  for (const auto &item : std::filesystem::recursive_directory_iterator(root_))
    EXPECT_TRUE(std::filesystem::is_directory(item)) << item.path();
}

TEST_F(TranslationStoreTest, AnEmptyObjectIsNotStored) {
  auto store = open("pair-a");
  put(*store, {});
  EXPECT_TRUE(get(*store).empty());
  EXPECT_EQ(store->size_for_test(), 0u);
}

TEST_F(TranslationStoreTest, TwoHandlesOnOneDomainSeeEachOther) {
  auto writer = open("pair-a");
  put(*writer, kObject);

  // Independent handles do not share in-process state, so this is the same
  // path a second process would take.
  auto reader = open("pair-a");
  EXPECT_EQ(get(*reader), kObject);
}

TEST_F(TranslationStoreTest, WhatAToolWritesToTheSharedTierARuntimeReads) {
  auto tool = open_shared("pair-a", TranslationStore::Access::kReadWrite);
  put(*tool, kObject);

  auto runtime = open_shared("pair-a", TranslationStore::Access::kReadOnly);
  EXPECT_EQ(get(*runtime), kObject);
}

TEST_F(TranslationStoreTest, TheSharedTierIsReadableByWhoeverEventuallyRuns) {
  // A shared tree is usually written by root during an image build and read by
  // an unprivileged process afterwards. Modes only the writer can open would
  // make every one of those reads a miss, and nothing would report it.
  //
  // The umask is what makes this a real risk, and the reason the store sets
  // modes explicitly instead of trusting the ones it passes to open(). A build
  // running under a restrictive umask is exactly where this goes wrong, so the
  // test creates that condition rather than waiting to meet it.
  const mode_t previous = umask(077);
  auto tool = open_shared("pair-a", TranslationStore::Access::kReadWrite);
  put(*tool, kObject);
  umask(previous);

  size_t files = 0;
  for (const auto &item : std::filesystem::recursive_directory_iterator(root_)) {
    const auto mode = std::filesystem::status(item).permissions();
    EXPECT_NE(mode & std::filesystem::perms::others_read, std::filesystem::perms::none)
        << item.path();
    EXPECT_EQ(mode & (std::filesystem::perms::group_write | std::filesystem::perms::others_write),
              std::filesystem::perms::none)
        << item.path();
    files += std::filesystem::is_regular_file(item) ? 1 : 0;
  }
  EXPECT_EQ(files, 2u) << "expected one object and one manifest";
}

TEST_F(TranslationStoreTest, AReadOnlySharedStoreDoesNotWriteAnExistingTree) {
  // The tree has to already exist, or this would only be re-proving that a store
  // with nowhere to write does not write.
  auto tool = open_shared("pair-a", TranslationStore::Access::kReadWrite);
  put(*tool, kObject);
  const uint64_t before = tool->size_for_test();
  ASSERT_GT(before, 0u);

  const std::vector<uint8_t> replacement{'w', 'r', 'o', 'n', 'g'};
  auto runtime = open_shared("pair-a", TranslationStore::Access::kReadOnly);
  ASSERT_TRUE(runtime->available());
  put(*runtime, replacement);

  EXPECT_EQ(get(*runtime), kObject);
  EXPECT_EQ(tool->size_for_test(), before);
}

TEST_F(TranslationStoreTest, AReadOnlySharedStoreCreatesNoTree) {
  // An absent shared tier means nobody pre-translated. That is a miss, not
  // something to repair: a directory this process created could not be trusted
  // by the next one anyway, since it would fail its own ownership test under a
  // different user.
  auto runtime = open_shared("pair-a", TranslationStore::Access::kReadOnly);
  put(*runtime, kObject);
  EXPECT_TRUE(get(*runtime).empty());
  EXPECT_TRUE(std::filesystem::is_empty(root_));
}

TEST_F(TranslationStoreTest, ASharedTreeAnyoneCanWriteIsRefused) {
  // Distinct from the session tier's rule, which demands sole ownership. Here
  // the question is whether someone outside the trust boundary could have
  // placed an entry, and a directory writable by group or other says yes.
  //
  // The tree is planted rather than created and then loosened, because that is
  // the shape of the real thing: an attacker who can write the parent makes the
  // directory before the tool does. The tool must refuse it as found, not
  // silently correct the mode and carry on.
  int domain_index = 0;
  for (const auto extra :
       {std::filesystem::perms::group_write, std::filesystem::perms::others_write}) {
    const std::string domain = "pair-" + std::to_string(domain_index++);
    const std::filesystem::path leaf = std::filesystem::path(root_) / domain / "v1";
    std::filesystem::create_directories(leaf);
    std::filesystem::permissions(leaf, std::filesystem::perms::owner_all | extra);

    auto tool = open_shared(domain, TranslationStore::Access::kReadWrite);
    EXPECT_FALSE(tool->available()) << domain;
    put(*tool, kObject);
    EXPECT_TRUE(get(*tool).empty()) << domain;

    auto runtime = open_shared(domain, TranslationStore::Access::kReadOnly);
    EXPECT_FALSE(runtime->available()) << domain;
  }
}

#if defined(RJ_TRANSLATOR_PROBE_a) && defined(RJ_TRANSLATOR_PROBE_b)
/// @brief A store keyed on a translator loaded from @p module.
///
/// @details The module stays loaded for the process lifetime. Unloading it while
/// a store still names an address inside it would leave that address pointing at
/// nothing, and the store resolves it lazily.
[[nodiscard]] std::unique_ptr<TranslationStore>
open_against_module(const char *module, std::string_view domain, const char *root) {
  void *handle = dlopen(module, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
  EXPECT_NE(handle, nullptr) << dlerror();
  if (handle == nullptr)
    return nullptr;
  void *anchor = dlsym(handle, "rj_probe_translator_anchor");
  EXPECT_NE(anchor, nullptr) << module;
  if (anchor == nullptr)
    return nullptr;
  auto store = std::make_unique<TranslationStore>(domain, anchor);
  store->set_root_for_test(root);
  return store;
}

TEST_F(TranslationStoreTest, ARebuiltTranslatorDoesNotReadTheOldEntries) {
  // The anti-staleness guarantee. Translation output is a function of the
  // translator, so an entry produced by one build must never satisfy a request
  // made against another -- a miss costs a retranslation, whereas a stale hit
  // silently runs code the current translator would not have emitted.
  //
  // The two modules differ in build id and nothing else, which is what a rebuild
  // that changed emitted bytes looks like to the store.
  auto first = open_against_module(RJ_TRANSLATOR_PROBE_a, "pair-a", root_);
  ASSERT_NE(first, nullptr);
  put(*first, kObject);
  ASSERT_EQ(get(*first), kObject) << "the entry was never written";

  auto second = open_against_module(RJ_TRANSLATOR_PROBE_b, "pair-a", root_);
  ASSERT_NE(second, nullptr);
  EXPECT_TRUE(get(*second).empty()) << "a differently-built translator read a stale entry";

  // The keys must actually differ, rather than the miss coming from some
  // unrelated refusal that would also hide a real collision.
  EXPECT_NE(first->key_for(kSource, kIdentity).digest, second->key_for(kSource, kIdentity).digest);

  // And the second build can hold its own entry for the same source alongside
  // the first, so this is separation rather than the tier being unusable.
  const std::vector<uint8_t> rebuilt{'n', 'e', 'w', 'e', 'r'};
  put(*second, rebuilt);
  EXPECT_EQ(get(*second), rebuilt);
  EXPECT_EQ(get(*first), kObject);
}
#endif // RJ_TRANSLATOR_PROBE_a && RJ_TRANSLATOR_PROBE_b

TEST_F(TranslationStoreTest, AGroupWritableStoreRootIsStillUsable) {
  // $XDG_RUNTIME_DIR/rocjitsu is created by the daemon and is group-writable, so
  // applying the store's own no-group-or-other-write rule to the root it is
  // handed disables the session tier wherever the daemon has run -- silently,
  // because an unusable store is indistinguishable from a cold one. Ownership is
  // what matters for a root the store did not create; the stricter rule belongs
  // to the directories it creates and reads entries from.
  const std::filesystem::path root = std::filesystem::path(root_) / "daemon-made";
  std::filesystem::create_directories(root);
  std::filesystem::permissions(root, kSafeDirPerms | std::filesystem::perms::group_write);

  auto store = open_shared("pair-a", TranslationStore::Access::kReadWrite, root.string().c_str());
  ASSERT_TRUE(store->available()) << "a group-writable root must not disable the store";
  put(*store, kObject);
  EXPECT_EQ(get(*store), kObject);

  // The directories holding entries are still strict, whatever the root allows.
  for (const auto &item : std::filesystem::recursive_directory_iterator(root)) {
    if (!std::filesystem::is_directory(item))
      continue;
    EXPECT_EQ(std::filesystem::status(item).permissions() &
                  (std::filesystem::perms::group_write | std::filesystem::perms::others_write),
              std::filesystem::perms::none)
        << item.path();
  }
}

TEST_F(TranslationStoreTest, ASymlinkedDomainIsRefusedRatherThanFollowed) {
  // Checking only where an assembled path lands proves nothing about how it got
  // there. With the domain pre-created as a symlink, a writer walks out of the
  // root it was given and a reader finds the entry at the target, with every
  // check passing -- so a symlink at any component has to be refused rather than
  // followed and then described.
  //
  // The ordinary domain below is the control, and it is the whole reason this
  // test means anything: without it, a store refusing BOTH cases -- for any
  // unrelated reason -- would read as a pass, and the assertion would be
  // measuring nothing. Both halves run under identical ownership and modes, so
  // the only difference between them is the symlink.
  auto ordinary = open_shared("pair-ok", TranslationStore::Access::kReadWrite);
  ASSERT_TRUE(ordinary->available()) << "control: a plain domain must be usable here";
  put(*ordinary, kObject);
  ASSERT_EQ(get(*ordinary), kObject) << "control: a plain domain must round-trip here";

  const std::filesystem::path elsewhere = std::filesystem::path(root_) / "elsewhere";
  std::filesystem::create_directories(elsewhere);
  // Explicit, because the ambient umask decides otherwise: a group-writable
  // directory is refused on its own merits, which would make the symlink
  // assertion below pass without testing the symlink at all.
  std::filesystem::permissions(elsewhere, kSafeDirPerms);
  std::filesystem::create_directory_symlink(elsewhere, std::filesystem::path(root_) / "pair-a");

  for (auto access : {TranslationStore::Access::kReadWrite, TranslationStore::Access::kReadOnly}) {
    auto shared_store = open_shared("pair-a", access);
    EXPECT_FALSE(shared_store->available()) << "a symlinked domain must not be followed";
    put(*shared_store, kObject);
    EXPECT_TRUE(get(*shared_store).empty());
  }

  auto session = open("pair-a");
  EXPECT_FALSE(session->available());
  put(*session, kObject);
  EXPECT_TRUE(get(*session).empty());

  // And nothing was written through the link into the directory it targets.
  EXPECT_TRUE(std::filesystem::is_empty(elsewhere));
}

TEST_F(TranslationStoreTest, ASymlinkAboveTheDomainIsRefusedToo) {
  // The parent of the domain is just as load-bearing: redirecting it relocates
  // the whole tree, including entries a later reader would trust. Same structure
  // as above -- a real directory first, so refusing everything cannot pass.
  const std::filesystem::path real_root = std::filesystem::path(root_) / "real";
  std::filesystem::create_directories(real_root);
  std::filesystem::permissions(real_root, kSafeDirPerms);
  auto ordinary =
      open_shared("pair-a", TranslationStore::Access::kReadWrite, real_root.string().c_str());
  ASSERT_TRUE(ordinary->available()) << "control: a plain root must be usable here";
  put(*ordinary, kObject);
  ASSERT_EQ(get(*ordinary), kObject);

  const std::filesystem::path elsewhere = std::filesystem::path(root_) / "elsewhere";
  std::filesystem::create_directories(elsewhere);
  std::filesystem::permissions(elsewhere, kSafeDirPerms);
  const std::filesystem::path linked_root = std::filesystem::path(root_) / "link";
  std::filesystem::create_directory_symlink(elsewhere, linked_root);

  auto tool =
      open_shared("pair-a", TranslationStore::Access::kReadWrite, linked_root.string().c_str());
  EXPECT_FALSE(tool->available()) << "a symlinked root must not be followed";
  put(*tool, kObject);
  EXPECT_TRUE(get(*tool).empty());
  EXPECT_TRUE(std::filesystem::is_empty(elsewhere));
}

TEST_F(TranslationStoreTest, TheSharedTierHoldsWhatTheSessionTierRefuses) {
  // The reason the shared tier exists. The session tier is rooted in the
  // per-user runtime directory, which is memory-backed, so it caps entry size
  // well below the few hundred megabytes a large device library occupies --
  // exactly the objects whose translation dominates start-up. An object over
  // that cap has to land somewhere or pre-translating it accomplishes nothing.
  const std::vector<uint8_t> oversized(17u << 20, 0xab);

  auto session = open("pair-a");
  put(*session, oversized);
  EXPECT_TRUE(get(*session).empty());

  auto tool = open_shared("pair-b", TranslationStore::Access::kReadWrite);
  put(*tool, oversized);
  EXPECT_EQ(get(*tool), oversized);
}

} // namespace
