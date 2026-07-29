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

namespace {

using rocjitsu::TranslationIdentity;
using rocjitsu::TranslationStore;

constexpr TranslationIdentity kIdentity{
    .profile_id = 7,
    .input_revision = 2,
    .output_revision = 1,
    .target_isa = "gfx-example",
};

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

  /// @brief A store rooted in this test's private directory.
  [[nodiscard]] std::unique_ptr<TranslationStore> open(std::string_view domain) {
    // Any address inside the translator would do in production; a test that is
    // not translating anything only needs the two stores it compares to agree.
    auto store = std::make_unique<TranslationStore>(
        domain, reinterpret_cast<const void *>(&translator_identity_anchor));
    store->set_root_for_test(root_);
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

} // namespace
