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

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <dlfcn.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

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

  /// @brief A writable store over @p path, or over this test's directory.
  /// @details Any address inside the translator would do in production; a test
  /// that is not translating anything only needs the stores it compares to agree.
  [[nodiscard]] std::unique_ptr<TranslationStore> open_shared(std::string_view domain,
                                                              TranslationStore::Access access,
                                                              const char *path = nullptr) {
    return std::make_unique<TranslationStore>(
        domain, reinterpret_cast<const void *>(&translator_identity_anchor),
        path == nullptr ? root_ : path, access, TranslationStore::KeyMode::kTranslatorBuild);
  }

  /// @brief The common case: a writable store over this test's directory.
  [[nodiscard]] std::unique_ptr<TranslationStore> open(std::string_view domain) {
    return open_shared(domain, TranslationStore::Access::kReadWrite);
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

TEST_F(TranslationStoreTest, AStoreRootMustBeAnAbsoluteDescendedPath) {
  EXPECT_FALSE(open_shared("pair-a", TranslationStore::Access::kReadWrite, "/")->available());

  // This spelling would reach the fixture directory if it were incorrectly
  // interpreted relative to `/`, making the assertion independent of whether a
  // coincidentally named top-level directory exists on the host.
  const std::string relative = std::string(root_).substr(1);
  auto store = open_shared("pair-a", TranslationStore::Access::kReadWrite, relative.c_str());
  EXPECT_FALSE(store->available());
  EXPECT_FALSE(store->key_for(kSource, kIdentity).valid);
  EXPECT_TRUE(std::filesystem::is_empty(root_));
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
  // The question is whether someone outside the trust boundary could have
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
  return std::make_unique<TranslationStore>(domain, anchor, root,
                                            TranslationStore::Access::kReadWrite,
                                            TranslationStore::KeyMode::kTranslatorBuild);
}

[[nodiscard]] std::unique_ptr<TranslationStore>
open_portable_against_module(const char *module, std::string_view domain, const char *root,
                             TranslationStore::Access access) {
  void *handle = dlopen(module, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
  EXPECT_NE(handle, nullptr) << dlerror();
  if (handle == nullptr)
    return nullptr;
  void *anchor = dlsym(handle, "rj_probe_translator_anchor");
  EXPECT_NE(anchor, nullptr) << module;
  if (anchor == nullptr)
    return nullptr;
  return std::make_unique<TranslationStore>(domain, anchor, root, access,
                                            TranslationStore::KeyMode::kPortablePrebuilt);
}

TEST_F(TranslationStoreTest, SharedRootUsesTheResolvedTranslatorPath) {
  const std::filesystem::path prefix(root_);
  std::filesystem::create_directories(prefix / "bin");
  std::filesystem::create_directories(prefix / "lib");
  const std::filesystem::path copied = prefix / "lib" / "librj-probe.so";
  std::filesystem::copy_file(RJ_TRANSLATOR_PROBE_a, copied);

  const std::string loader_path = (prefix / "bin" / ".." / "lib" / "librj-probe.so").string();
  void *handle = dlopen(loader_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(handle, nullptr) << dlerror();
  void *anchor = dlsym(handle, "rj_probe_translator_anchor");
  ASSERT_NE(anchor, nullptr);

  EXPECT_EQ(rocjitsu::shared_translation_root(anchor),
            (prefix / "share" / "rocjitsu" / "translations").string());
  dlclose(handle);
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

TEST_F(TranslationStoreTest, APortablePrebuiltEntrySurvivesATranslatorRebuild) {
  // A packaged cache and its consuming code are released as one product. Its
  // compatibility boundary is the cache epoch plus translation identity, not
  // an incidental ELF build ID that packaging may change without changing the
  // translation contract.
  auto tool = open_portable_against_module(RJ_TRANSLATOR_PROBE_a, "pair-a", root_,
                                           TranslationStore::Access::kReadWrite);
  ASSERT_NE(tool, nullptr);
  put(*tool, kObject);
  ASSERT_EQ(get(*tool), kObject) << "the portable entry was never written";

  std::filesystem::path manifest;
  for (const auto &item : std::filesystem::recursive_directory_iterator(root_)) {
    if (item.path().extension() == ".man")
      manifest = item.path();
  }
  ASSERT_FALSE(manifest.empty());
  std::ifstream manifest_stream(manifest);
  const std::string manifest_text((std::istreambuf_iterator<char>(manifest_stream)), {});
  EXPECT_NE(manifest_text.find("\nbuild=unversioned-prebuilt-v1\n"), std::string::npos);

  auto runtime = open_portable_against_module(RJ_TRANSLATOR_PROBE_b, "pair-a", root_,
                                              TranslationStore::Access::kReadOnly);
  ASSERT_NE(runtime, nullptr);
  EXPECT_EQ(get(*runtime), kObject);
  EXPECT_EQ(tool->key_for(kSource, kIdentity).digest, runtime->key_for(kSource, kIdentity).digest);

  // Exact and portable stores remain separate policies. Opting a packaged tier
  // into portability cannot weaken an ordinary build-specific cache.
  auto exact = open_against_module(RJ_TRANSLATOR_PROBE_b, "pair-a", root_);
  ASSERT_NE(exact, nullptr);
  EXPECT_TRUE(get(*exact).empty());
}
#endif // RJ_TRANSLATOR_PROBE_a && RJ_TRANSLATOR_PROBE_b

TEST_F(TranslationStoreTest, ADomainContainingNulDoesNotAliasAnother) {
  // The C path APIs stop at a NUL, so these two distinct views name one
  // directory. Without rejecting them the second domain reads back what the
  // first wrote, which is precisely the isolation a domain is supposed to give.
  const std::string_view first("pair\0a", 6);
  const std::string_view second("pair\0b", 6);

  auto writer = open(first);
  EXPECT_FALSE(writer->key_for(kSource, kIdentity).valid);
  put(*writer, kObject);

  auto reader = open(second);
  EXPECT_TRUE(get(*reader).empty()) << "a truncated domain must not alias another";
  EXPECT_TRUE(get(*writer).empty());
}

TEST_F(TranslationStoreTest, ALineBreakingIdentityIsRefusedRatherThanStored) {
  // The manifest is line oriented. A newline in the ISA forges an earlier size=
  // or sha= field, so the object is written but every later lookup rejects it --
  // a write-only entry that is retried for as long as the workload runs.
  constexpr TranslationIdentity injected{
      .profile_id = 7,
      .input_revision = 2,
      .output_revision = 1,
      .target_isa = "gfx-example\nsize=0\nsha=0",
  };

  auto store = open("pair-a");
  EXPECT_FALSE(store->key_for(kSource, injected).valid);
  store->store(store->key_for(kSource, injected), kObject, injected);
  EXPECT_TRUE(store->lookup(store->key_for(kSource, injected), injected).empty());
  EXPECT_EQ(store->size_for_test(), 0u) << "nothing may be written for a refused identity";

  // The ordinary identity still works, so this is refusal and not a dead store.
  put(*store, kObject);
  EXPECT_EQ(get(*store), kObject);
}

TEST_F(TranslationStoreTest, AManifestWithNoObjectIsReclaimed) {
  // What a process killed between the two renames leaves behind. It can never
  // satisfy a lookup, and if eviction does not remove it nothing ever will.
  auto store = open("pair-a");
  put(*store, kObject);
  ASSERT_EQ(get(*store), kObject);

  const std::filesystem::path entries = std::filesystem::path(root_) / "pair-a" / "v1";
  for (const auto &item : std::filesystem::directory_iterator(entries))
    if (item.path().extension() == ".obj")
      std::filesystem::remove(item.path());
  ASSERT_GT(store->size_for_test(), 0u) << "the orphan manifest must still be accounted";

  // Age it past the threshold that separates an abandoned write from one still
  // in progress, then re-open: the sweep runs once when the store opens.
  for (const auto &item : std::filesystem::directory_iterator(entries))
    std::filesystem::last_write_time(item.path(), std::filesystem::file_time_type::clock::now() -
                                                      std::chrono::hours(1));
  auto reopened = open("pair-a");
  EXPECT_TRUE(get(*reopened).empty());

  size_t orphans = 0;
  for (const auto &item : std::filesystem::directory_iterator(entries)) {
    if (item.path().extension() != ".man")
      continue;
    auto object = item.path();
    object.replace_extension(".obj");
    orphans += std::filesystem::exists(object) ? 0 : 1;
  }
  EXPECT_EQ(orphans, 0u) << "an orphan manifest must be reclaimed";
}

TEST_F(TranslationStoreTest, AbandonedFileSweepWaitsForTheDirectoryWriter) {
  {
    auto initial = open("pair-a");
    ASSERT_TRUE(initial->available());
  }
  const std::filesystem::path entries = std::filesystem::path(root_) / "pair-a" / "v1";
  const std::filesystem::path temporary = entries / "entry.tmp.writer";
  std::ofstream(temporary) << "still publishing";
  std::filesystem::last_write_time(temporary, std::filesystem::file_time_type::clock::now() -
                                                  std::chrono::hours(1));

  int ready[2];
  int release[2];
  ASSERT_EQ(pipe(ready), 0);
  ASSERT_EQ(pipe(release), 0);
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    close(ready[0]);
    close(release[1]);
    const int fd = ::open(entries.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    char token = 'x';
    if (fd < 0 || flock(fd, LOCK_EX) != 0 || write(ready[1], &token, 1) != 1 ||
        read(release[0], &token, 1) != 1)
      _exit(1);
    close(fd);
    _exit(0);
  }

  close(ready[1]);
  close(release[0]);
  char token = 0;
  ASSERT_EQ(read(ready[0], &token, 1), 1);
  auto reopening = std::async(std::launch::async, [this] {
    auto store = open("pair-a");
    return store->available();
  });
  EXPECT_EQ(reopening.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  EXPECT_TRUE(reopening.get());
  EXPECT_TRUE(std::filesystem::exists(temporary));

  ASSERT_EQ(write(release[1], &token, 1), 1);
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  auto next = open("pair-a");
  EXPECT_TRUE(next->available());
  EXPECT_FALSE(std::filesystem::exists(temporary));
  close(ready[0]);
  close(release[1]);
}

TEST_F(TranslationStoreTest, AGroupWritableStoreRootIsStillUsable) {
  // A root the store is handed may legitimately be group-writable -- an install
  // directory owned by a packaging group, for one -- and applying the store's own
  // no-group-or-other-write rule to it would disable the store silently, because
  // an unusable store is indistinguishable from a cold one. Ownership is what
  // matters for a root the store did not create; the stricter rule belongs to the
  // directories it creates and reads entries from.
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

TEST_F(TranslationStoreTest, AnObjectFarLargerThanAnInMemoryCacheRoundTrips) {
  // The reason the store exists. The objects whose translation dominates
  // start-up are the large device libraries -- a few hundred megabytes -- and
  // an in-process or memory-backed cache cannot hold them at any sensible size.
  // Anything that caps entries in the low megabytes would refuse exactly the
  // objects pre-translating is for, so the ceiling here is a sanity bound only.
  const std::vector<uint8_t> oversized(17u << 20, 0xab);

  auto tool = open_shared("pair-b", TranslationStore::Access::kReadWrite);
  put(*tool, oversized);
  EXPECT_EQ(get(*tool), oversized);
}

} // namespace
