// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/hooks/sidecar_registry.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <string>
#include <thread>
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

// Thread-safety is the core contract of this registry: the load/symbol/object
// mutation path runs on the ROCR loader thread while the dispatch/lookup path
// runs on the doorbell and scanner threads, and executable teardown races both.
// This stress test drives all of those entry points concurrently so
// ThreadSanitizer can flag any missing lock or torn read. It asserts the
// process does not crash or trip TSan rather than a specific interleaving
// outcome, since the whole point is that no interleaving is unsafe.
TEST_F(SidecarRegistryTest, ConcurrentMutationAndLookupIsRaceFree) {
  auto &registry = SidecarRegistry::instance();
  constexpr int kExecutables = 8;
  constexpr int kIterations = 500;

  std::atomic<bool> start{false};
  std::vector<std::thread> threads;

  // Writers: each owns a distinct executable id and repeatedly loads, records a
  // symbol, publishes a kernel object, then erases and starts over.
  for (int e = 0; e < kExecutables; ++e) {
    threads.emplace_back([&, e] {
      const uint64_t executable = static_cast<uint64_t>(e) + 1;
      const uint64_t symbol = 100 + executable;
      const uint64_t object = kLoadBase + (executable << 16);
      while (!start.load(std::memory_order_acquire)) {
      }
      for (int i = 0; i < kIterations; ++i) {
        registry.record_load(executable, executable << 8,
                             one_sidecar("kernel" + std::to_string(e)));
        registry.record_symbol(executable, "kernel" + std::to_string(e) + ".kd", symbol);
        registry.note_kernel_object(symbol, object, static_cast<uint32_t>(i & 0xFF));
        (void)registry.find_by_kernel_object(object);
        registry.erase_executable(executable);
      }
    });
  }

  // Readers: hammer the lookup paths across the whole object-id space so they
  // observe partially-published state from the writers above.
  for (int r = 0; r < 4; ++r) {
    threads.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) {
      }
      for (int i = 0; i < kIterations * kExecutables; ++i) {
        const uint64_t executable = static_cast<uint64_t>(i % kExecutables) + 1;
        const uint64_t object = kLoadBase + (executable << 16);
        (void)registry.find_by_kernel_object(object);
        (void)registry.kernel_name_for_object(object);
        (void)registry.private_segment_size_for_object(object);
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto &thread : threads)
    thread.join();

  SUCCEED();
}

} // namespace
