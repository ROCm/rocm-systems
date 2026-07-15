// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/hooks/sidecar_registry.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

using rocjitsu::SidecarVariantMetadata;
using rocjitsu::hooks::SidecarRegistry;

constexpr uint64_t kNormalVaddr = 0x1000;
constexpr uint64_t kVariantVaddr = 0x1800;
constexpr uint64_t kLoadBase = 0x100000;
constexpr uint64_t kNormalObject = kLoadBase + kNormalVaddr;
constexpr uint64_t kVariantObject = kLoadBase + kVariantVaddr;

std::vector<SidecarVariantMetadata> one_sidecar(std::string kernel = "kernel") {
  return {{.kernel_name = std::move(kernel),
           .variant_name = "fallback",
           .normal_descriptor_vaddr = kNormalVaddr,
           .variant_descriptor_vaddr = kVariantVaddr}};
}

class SidecarRegistryTest : public ::testing::Test {
protected:
  void SetUp() override { SidecarRegistry::instance().clear(); }
  void TearDown() override { SidecarRegistry::instance().clear(); }
};

TEST_F(SidecarRegistryTest, ResolvesLoadWithoutOptionalLoadedCodeObjectHandle) {
  auto &registry = SidecarRegistry::instance();
  registry.record_load(1, 0, one_sidecar());
  registry.record_symbol(1, "kernel.kd", 10);
  registry.note_kernel_object(10, kNormalObject, 32);

  const auto resolved = registry.find_by_kernel_object(kNormalObject);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(resolved->load_id, 0u);
  EXPECT_EQ(resolved->symbol, 10u);
  EXPECT_EQ(resolved->kernel_name, "kernel");
  EXPECT_EQ(resolved->variant_object("fallback"), kVariantObject);
}

TEST_F(SidecarRegistryTest, RepeatedSymbolObservationPreservesPublishedObjectFacts) {
  auto &registry = SidecarRegistry::instance();
  registry.record_load(1, 20, one_sidecar());
  registry.record_symbol(1, "kernel.kd", 10);
  registry.note_kernel_object(10, kNormalObject, 48);

  // Lookup-by-name and symbol iteration can both report the same symbol after
  // KERNEL_OBJECT has already been queried.
  registry.record_symbol(1, "kernel.kd", 10);

  EXPECT_EQ(registry.kernel_name_for_object(kNormalObject), "kernel");
  EXPECT_EQ(registry.private_segment_size_for_object(kNormalObject), 48u);
  const auto resolved = registry.find_by_kernel_object(kNormalObject);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(resolved->variant_object("fallback"), kVariantObject);
}

TEST_F(SidecarRegistryTest, ErasingOlderExecutableKeepsReplacementReverseMappings) {
  auto &registry = SidecarRegistry::instance();
  registry.record_load(1, 20, one_sidecar("old"));
  registry.record_symbol(1, "old.kd", 10);
  registry.note_kernel_object(10, kNormalObject, 16);

  // Model allocator reuse: a newer executable publishes the same descriptor
  // address before teardown for the older executable reaches the hook.
  registry.record_load(2, 21, one_sidecar("new"));
  registry.record_symbol(2, "new.kd", 11);
  registry.note_kernel_object(11, kNormalObject, 64);
  registry.erase_executable(1);

  EXPECT_EQ(registry.kernel_name_for_object(kNormalObject), "new");
  EXPECT_EQ(registry.private_segment_size_for_object(kNormalObject), 64u);
  const auto resolved = registry.find_by_kernel_object(kNormalObject);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(resolved->executable, 2u);
  EXPECT_EQ(resolved->kernel_name, "new");
}

TEST_F(SidecarRegistryTest, ReusedSymbolHandleDropsFactsOwnedByOldExecutable) {
  auto &registry = SidecarRegistry::instance();
  registry.record_load(1, 20, one_sidecar("old"));
  registry.record_symbol(1, "old.kd", 10);
  registry.note_kernel_object(10, kNormalObject, 16);

  registry.record_load(2, 21, one_sidecar("new"));
  registry.record_symbol(2, "new.kd", 10);

  EXPECT_FALSE(registry.kernel_name_for_object(kNormalObject).has_value());
  EXPECT_FALSE(registry.find_by_kernel_object(kNormalObject).has_value());

  registry.note_kernel_object(10, kNormalObject + 0x10000, 64);
  EXPECT_EQ(registry.kernel_name_for_object(kNormalObject + 0x10000), "new");
  EXPECT_EQ(registry.private_segment_size_for_object(kNormalObject + 0x10000), 64u);
}

TEST_F(SidecarRegistryTest, RejectsVariantAddressOverflow) {
  auto metadata = one_sidecar();
  metadata.front().variant_descriptor_vaddr = std::numeric_limits<uint64_t>::max();

  auto &registry = SidecarRegistry::instance();
  registry.record_load(1, 20, std::move(metadata));
  registry.record_symbol(1, "kernel.kd", 10);
  registry.note_kernel_object(10, kNormalObject, 32);

  EXPECT_FALSE(registry.find_by_kernel_object(kNormalObject).has_value());
}

TEST_F(SidecarRegistryTest, ClearsUnresolvedLoadRecords) {
  auto metadata = one_sidecar(std::string(256, 'k'));
  metadata.front().variant_name = std::string(256, 'v');

  auto &registry = SidecarRegistry::instance();
  registry.record_load(1, 0, std::move(metadata));
  registry.clear();

  EXPECT_FALSE(registry.find_by_kernel_object(kNormalObject).has_value());
}

} // namespace
