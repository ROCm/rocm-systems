// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/hooks/hsa_api_function_patch.h"
#include "rocjitsu/hooks/hsa_code_object_file_snapshot.h"
#include "rocjitsu/hooks/hsa_code_object_reader_registry.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

using rocjitsu::hooks::HsaCodeObjectReaderRegistry;

int original_function(int value) { return value; }
int replacement_function(int value) { return value + 1; }
int later_tool_function(int value) { return value + 2; }

TEST(HsaApiFunctionPatchTest, RestoresOnlyItsOwnInstalledReplacement) {
  using Function = int (*)(int);
  Function slot = original_function;
  rocjitsu::hooks::HsaApiFunctionPatch<Function> patch;

  patch.capture(&slot);
  EXPECT_EQ(patch.original(), original_function);
  patch.install(replacement_function);
  EXPECT_EQ(slot, replacement_function);
  patch.restore();
  EXPECT_EQ(slot, original_function);
  EXPECT_EQ(patch.original(), original_function);

  patch.capture(&slot);
  patch.install(replacement_function);
  slot = later_tool_function;
  patch.restore();
  EXPECT_EQ(slot, later_tool_function);
  EXPECT_EQ(patch.original(), original_function);
  patch.clear();
  EXPECT_EQ(patch.original(), nullptr);
}

TEST(HsaApiFunctionPatchTest, CanCaptureAnUnpatchedSavedFunction) {
  using Function = int (*)(int);
  Function slot = original_function;
  rocjitsu::hooks::HsaApiFunctionPatch<Function> patch;

  patch.capture(&slot);
  EXPECT_EQ(patch.original(), original_function);
  EXPECT_EQ(slot, original_function);
  patch.restore();
  EXPECT_EQ(slot, original_function);
}

class ScopedFd {
public:
  explicit ScopedFd(int fd) : fd_(fd) {}
  ~ScopedFd() {
    if (fd_ >= 0)
      close(fd_);
  }
  int get() const { return fd_; }

private:
  int fd_;
};

TEST(HsaCodeObjectFileSnapshotTest, ReadsWholeFileAndRanges) {
  ScopedFd file(open("/bin/sh", O_RDONLY));
  ASSERT_GE(file.get(), 0);

  const auto whole = rocjitsu::snapshot_code_object_file(file.get());
  ASSERT_TRUE(whole);
  ASSERT_GE(whole->size(), 4u);

  const auto range = rocjitsu::snapshot_code_object_file_range(file.get(), 1, 3);
  ASSERT_TRUE(range);
  EXPECT_EQ(*range, (std::vector<uint8_t>(whole->begin() + 1, whole->begin() + 4)));
}

TEST(HsaCodeObjectFileSnapshotTest, RejectsInvalidAndEmptyRanges) {
  ScopedFd file(open("/bin/sh", O_RDONLY));
  ASSERT_GE(file.get(), 0);

  EXPECT_FALSE(rocjitsu::snapshot_code_object_file_range(file.get(), 0, 0));
  EXPECT_FALSE(rocjitsu::snapshot_code_object_file_range(file.get(), SIZE_MAX, 1));
  EXPECT_FALSE(rocjitsu::snapshot_code_object_file(-1));
}

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
