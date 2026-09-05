// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/hooks/hsa_code_object_reader_registry.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <vector>

namespace {

using rocjitsu::hooks::HsaCodeObjectReaderRegistry;

TEST(HsaCodeObjectReaderRegistryTest, ReplacesAndRemovesReaderBytes) {
  HsaCodeObjectReaderRegistry registry;
  const std::array<uint8_t, 2> first = {1u, 2u};
  const std::array<uint8_t, 3> second = {3u, 4u, 5u};

  ASSERT_TRUE(registry.store(7u, first.data(), first.size()));
  ASSERT_TRUE(registry.store(7u, second.data(), second.size()));
  const auto replaced = registry.lookup(7u);
  ASSERT_TRUE(replaced);
  EXPECT_EQ(replaced.bytes, second.data());
  EXPECT_EQ(replaced.size, second.size());

  registry.remove(7u);
  EXPECT_FALSE(registry.lookup(7u));
}

TEST(HsaCodeObjectReaderRegistryTest, LookupSnapshotRetainsOwnedStorage) {
  HsaCodeObjectReaderRegistry registry;
  auto owned =
      std::make_shared<const std::vector<uint8_t>>(std::initializer_list<uint8_t>{11u, 12u, 13u});
  ASSERT_TRUE(registry.store(9u, owned->data(), owned->size(), owned));

  const auto snapshot = registry.lookup(9u);
  registry.clear();
  owned.reset();

  ASSERT_TRUE(snapshot);
  ASSERT_TRUE(snapshot.owned);
  EXPECT_EQ(*snapshot.owned, (std::vector<uint8_t>{11u, 12u, 13u}));
}

TEST(HsaCodeObjectReaderRegistryTest, InstancesOwnIndependentReaderNamespaces) {
  HsaCodeObjectReaderRegistry first;
  HsaCodeObjectReaderRegistry second;
  const std::array<uint8_t, 1> bytes = {42u};

  ASSERT_TRUE(first.store(3u, bytes.data(), bytes.size()));
  EXPECT_TRUE(first.lookup(3u));
  EXPECT_FALSE(second.lookup(3u));
}

} // namespace
