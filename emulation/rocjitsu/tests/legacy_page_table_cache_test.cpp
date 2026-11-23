// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/kmd/linux/legacy_gpu_vm.h"
#include "rocjitsu/vm/amdgpu/legacy_address_space.h"
#include "rocjitsu/vm/amdgpu/legacy_page_table.h"
#include "util/diagnostic.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <optional>
#include <pthread.h>
#include <string_view>
#include <thread>
#include <unistd.h>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

TEST(LegacyPageTableCacheStateTest, RegistrationRequiresMutationSerialization) {
  GpuMemory memory("page_cache_registration");
  LegacyAddressSpace space(memory);
  LegacyPageTable pages;
  util::DistributedSharedMutex page_mutex;
  auto state = std::make_shared<LegacyPageTableCacheState>();
  util::StringDiagnostic diagnostics;
  EXPECT_TRUE(
      space.register_process(7, &pages, &page_mutex, nullptr, {}, state, diagnostics.emitter())
          .failed());
  EXPECT_EQ(diagnostics.message(), "cached page-table policies require a shared request mutex");
  auto request_mutex = std::make_shared<util::DistributedSharedMutex>();
  EXPECT_TRUE(
      space.register_process(7, &pages, &page_mutex, nullptr, request_mutex, state).succeeded());
}

TEST(LegacyPageTableCacheStateTest, FailedRegistrationPreservesMappingAndCanBeRetried) {
  GpuMemory memory("page_cache_retry");
  LegacyAddressSpace space(memory);
  KfdProcess original(7);
  KfdProcess replacement(7);
  constexpr uint64_t kAddress = 0x40000000;
  std::array<uint8_t, kLegacyPageSize> first_page{};
  std::array<uint8_t, kLegacyPageSize> second_page{};
  first_page[0] = 0x37;
  second_page[0] = 0x58;
  original.map_pages(kAddress, first_page.data(), first_page.size(), Mtype::RW);
  replacement.map_pages(kAddress, second_page.data(), second_page.size(), Mtype::RW);
  ASSERT_TRUE(
      space.register_process(7, &original.page_table_, &original.page_table_mutex_).succeeded());
  EXPECT_TRUE(space
                  .register_process(7, &replacement.page_table_, &replacement.page_table_mutex_,
                                    nullptr, {}, replacement.page_table_cache_state())
                  .failed());
  EXPECT_EQ(space.read8(kAddress, 7), first_page[0]);
  ASSERT_TRUE(space
                  .register_process(7, &replacement.page_table_, &replacement.page_table_mutex_,
                                    nullptr, replacement.page_table_request_mutex(),
                                    replacement.page_table_cache_state())
                  .succeeded());
  EXPECT_EQ(space.read8(kAddress, 7), second_page[0]);
}

TEST(LegacyPageTableCacheStateTest, AdapterDoesNotInstallFailedRegistration) {
  GpuMemory memory("page_cache_adapter");
  GpuVm gpu_vm;
  LegacyGpuVmAdapter adapter(gpu_vm, &memory);
  KfdProcess process(7);
  util::StringDiagnostic diagnostic;
  auto registration = [&] {
    LegacyAddressSpaceRegistration result{};
    result.page_table = &process.page_table_;
    result.page_table_mutex = &process.page_table_mutex_;
    result.page_table_cache_state = process.page_table_cache_state();
    return result;
  };
  unsigned diagnostic_count = 0;
  auto report = [&](std::string_view message) {
    ++diagnostic_count;
    EXPECT_EQ(adapter.address_space(7), nullptr);
    diagnostic(message);
  };
  EXPECT_FALSE(
      adapter.register_address_space(7, registration(), {}, util::DiagnosticEmitter(report)));
  EXPECT_EQ(diagnostic_count, 1u);
  EXPECT_EQ(diagnostic.message(), "cached page-table policies require a shared request mutex");
  EXPECT_EQ(gpu_vm.active_address_spaces(), 0u);
  EXPECT_EQ(adapter.address_space(7), nullptr);
  auto valid = registration();
  valid.request_mutex = process.page_table_request_mutex();
  const AddressSpaceHandle handle = adapter.register_address_space(7, std::move(valid));
  ASSERT_TRUE(handle);
  EXPECT_EQ(gpu_vm.active_address_spaces(), 1u);
  EXPECT_TRUE(adapter.unregister_address_space(handle));
}

TEST(LegacyPageTableCacheStateTest, InvalidatesOldCopiesAndAdmitsPublishedGeneration) {
  LegacyPageTableCacheState state;
  const uint64_t old_generation = state.generation();
  ASSERT_TRUE(state.try_acquire(old_generation));
  state.release();
  state.invalidate_and_wait();
  EXPECT_FALSE(state.try_acquire(old_generation));
  ASSERT_TRUE(state.try_acquire(state.generation()));
  state.release();
}

TEST(LegacyPageTableCacheStateTest, NoOpMutationRestoresLockFreeHits) {
  EXPECT_EXIT(([] {
                alarm(10);
                GpuMemory memory("page_cache_noop");
                LegacyAddressSpace space(memory);
                KfdProcess process(7);
                constexpr uint64_t kAddress = 0x40000000;
                std::array<uint8_t, kLegacyPageSize> page{};
                page[0] = 0x37;
                process.map_pages(kAddress, page.data(), page.size(), Mtype::RW);
                if (space
                        .register_process(7, &process.page_table_, &process.page_table_mutex_,
                                          process.page_table_generation(),
                                          process.page_table_request_mutex(),
                                          process.page_table_cache_state())
                        .failed())
                  _exit(4);
                if (space.read8(kAddress, 7) != page[0])
                  _exit(1);
                const uint64_t generation = *process.page_table_generation();
                process.set_page_mtype(kAddress, page.size(), Mtype::RW);
                if (*process.page_table_generation() != generation ||
                    space.read8(kAddress, 7) != page[0])
                  _exit(2);

                // The lookup after the no-op must restore admission. Holding
                // the table lock on another thread distinguishes a copied-PTE
                // hit from a fallback that still takes the reader locks.
                std::atomic<bool> locked{false};
                std::atomic<bool> read_done{false};
                std::thread locker([&] {
                  std::unique_lock lock(process.page_table_mutex_);
                  locked.store(true);
                  while (!read_done.load())
                    std::this_thread::yield();
                });
                while (!locked.load())
                  std::this_thread::yield();
                if (space.read8(kAddress, 7) != page[0])
                  _exit(3);
                read_done.store(true);
                locker.join();
                alarm(0);
                _exit(0);
              }()),
              ::testing::ExitedWithCode(0), "");
}

TEST(LegacyPageTableCacheStateTest, InvalidationDrainsEveryNestedReader) {
  EXPECT_EXIT(([] {
                alarm(10);
                constexpr size_t kReaders = 32;
                LegacyPageTableCacheState state;
                const uint64_t generation = state.generation();
                std::atomic<size_t> entered{0};
                std::atomic<size_t> partially_released{0};
                std::atomic<unsigned> phase{0};
                std::atomic<bool> writer_done{false};
                std::array<std::atomic<bool>, kReaders> release_last{};
                std::array<std::atomic<bool>, kReaders> left{};
                std::vector<std::thread> readers;
                for (size_t index = 0; index < kReaders; ++index) {
                  readers.emplace_back([&, index] {
                    if (!state.try_acquire(generation) || !state.try_acquire(generation))
                      _exit(1);
                    entered.fetch_add(1);
                    while (phase.load() == 0)
                      std::this_thread::yield();
                    state.release();
                    partially_released.fetch_add(1);
                    while (!release_last[index].load())
                      std::this_thread::yield();
                    state.release();
                    left[index].store(true);
                  });
                }
                while (entered.load() != kReaders)
                  std::this_thread::yield();
                std::thread writer([&] {
                  state.invalidate_and_wait();
                  writer_done.store(true);
                });
                while (state.generation() == generation)
                  std::this_thread::yield();
                if (state.try_acquire(generation) || writer_done.load())
                  _exit(2);
                phase.store(1);
                while (partially_released.load() != kReaders)
                  std::this_thread::yield();
                // Release readers individually: a writer must drain every admission,
                // including collisions and the second admission on each thread.
                for (size_t index = 0; index < kReaders; ++index) {
                  if (writer_done.load())
                    _exit(3);
                  release_last[index].store(true);
                  while (!left[index].load())
                    std::this_thread::yield();
                }
                for (auto &reader : readers)
                  reader.join();
                writer.join();
                if (!writer_done.load() || state.try_acquire(generation))
                  _exit(4);
                alarm(0);
                _exit(0);
              }()),
              ::testing::ExitedWithCode(0), "");
}

TEST(LegacyPageTableCacheStateTest, ConcurrentAdmissionsObserveOnlyPublishedCopies) {
  EXPECT_EXIT(([] {
                // TSan plus concurrent CTest jobs can exceed ten seconds without a deadlock.
                alarm(60);
                LegacyPageTableCacheState state;
                std::atomic<uint64_t> published{state.generation()};
                uint64_t payload = published.load();
                std::atomic<bool> stop{false};
                std::atomic<size_t> started{0};
                std::atomic<size_t> reads{0};
                std::vector<std::thread> readers;
                for (size_t index = 0; index < 16; ++index) {
                  readers.emplace_back([&] {
                    started.fetch_add(1);
                    while (!stop.load()) {
                      const uint64_t copy_generation = published.load();
                      if (!state.try_acquire(copy_generation))
                        continue;
                      if (payload != copy_generation)
                        _exit(1);
                      reads.fetch_add(1, std::memory_order_relaxed);
                      state.release();
                    }
                  });
                }
                while (started.load() != readers.size() || reads.load() == 0)
                  std::this_thread::yield();
                for (size_t mutation = 0; mutation < 2000; ++mutation) {
                  state.invalidate_and_wait();
                  // Real PTE copies are published under the mutation-side locks only
                  // after the old admissions drain. This publication models that
                  // boundary, independently of the cache-state generation itself.
                  payload = state.generation();
                  published.store(payload);
                  std::this_thread::yield();
                }
                stop.store(true);
                for (auto &reader : readers)
                  reader.join();
                state.invalidate_and_wait();
                if (state.try_acquire(published.load()))
                  _exit(2);
                alarm(0);
                _exit(0);
              }()),
              ::testing::ExitedWithCode(0), "");
}

// glibc carves every thread's static TLS out of its stack, so rocJITsu's TLS
// decides whether the preloaded application can start a small-stack thread.
TEST(LegacyLocalExtentTableTest, PublishMergesOverlappingPiecesOfOneRun) {
  LegacyLocalExtentTable table;
  auto *host = reinterpret_cast<uint8_t *>(uintptr_t{0x7000'0000});
  const auto piece = [&](uint64_t begin, uint64_t end) {
    return LegacyLocalExtent{.begin = begin, .end = end, .host_begin = host + (begin - 0x10000)};
  };
  table.publish(piece(0x20000, 0x40000), 5);
  table.publish(piece(0x10000, 0x28000), 5);
  table.publish(piece(0x60000, 0x70000), 5);
  auto merged = table.find(0x3f000, 5);
  ASSERT_TRUE(merged);
  EXPECT_EQ(merged->begin, 0x10000u);
  EXPECT_EQ(merged->end, 0x40000u);
  EXPECT_EQ(merged->host(0x3f000), host + 0x2f000);
  EXPECT_FALSE(table.find(0x50000, 5));
  EXPECT_TRUE(table.find(0x60000, 5));
  // A newer generation discards every older extent; an older one is ignored.
  EXPECT_FALSE(table.find(0x10000, 6));
  table.publish(piece(0x60000, 0x61000), 6);
  table.publish(piece(0x10000, 0x40000), 5);
  EXPECT_FALSE(table.find(0x10000, 6));
  EXPECT_FALSE(table.find(0x60000, 5));
  EXPECT_TRUE(table.find(0x60000, 6));
}

class LegacyLocalExtentTest : public ::testing::Test {
protected:
  static constexpr uint32_t kVmid = 7;
  static constexpr uint64_t kBase = 0x4000'0000;

  void SetUp() override {
    ASSERT_TRUE(space_
                    .register_process(kVmid, &process_.page_table_, &process_.page_table_mutex_,
                                      nullptr, process_.page_table_request_mutex(),
                                      process_.page_table_cache_state())
                    .succeeded());
  }

  std::optional<LegacyLocalExtent> extent(uint64_t address) {
    return space_.local_extent(address, kVmid, process_.page_table_request_mutex(),
                               process_.page_table_cache_state(),
                               process_.page_table_cache_state()->generation());
  }

  uint8_t *pages(size_t count) {
    storage_.emplace_back(count * kLegacyPageSize + kLegacyPageSize);
    auto address = reinterpret_cast<uintptr_t>(storage_.back().data());
    return reinterpret_cast<uint8_t *>((address + kLegacyPageSize - 1) & ~(kLegacyPageSize - 1));
  }

  GpuMemory memory_{"local_extent"};
  LegacyAddressSpace space_{memory_};
  KfdProcess process_{kVmid};
  std::vector<std::vector<uint8_t>> storage_;
};

TEST_F(LegacyLocalExtentTest, CoversOneContiguousSealedRun) {
  uint8_t *host = pages(4);
  ASSERT_EQ(process_.map_pages(kBase, host, 4 * kLegacyPageSize, Mtype::NT,
                               LegacyHostExtentOwner::DriverSealedRam),
            KfdProcess::PageTableMutationResult::Applied);
  const auto found = extent(kBase + 2 * kLegacyPageSize + 8);
  ASSERT_TRUE(found);
  EXPECT_EQ(found->begin, kBase);
  EXPECT_EQ(found->end, kBase + 4 * kLegacyPageSize);
  EXPECT_EQ(found->host(kBase + 5), host + 5);
  EXPECT_EQ(found->mtype, Mtype::NT);
  EXPECT_EQ(found->owner, LegacyHostExtentOwner::DriverSealedRam);
  EXPECT_FALSE(extent(kBase + 4 * kLegacyPageSize));
}

TEST_F(LegacyLocalExtentTest, SplitsAtHostGapsMtypeAndOwnerChanges) {
  uint8_t *first = pages(2);
  uint8_t *second = pages(1);
  uint8_t *third = pages(2);
  constexpr auto kSealed = LegacyHostExtentOwner::DriverSealedRam;
  process_.map_pages(kBase, first, 2 * kLegacyPageSize, Mtype::RW, kSealed);
  process_.map_pages(kBase + 2 * kLegacyPageSize, second, kLegacyPageSize, Mtype::RW, kSealed);
  process_.map_pages(kBase + 3 * kLegacyPageSize, third, kLegacyPageSize, Mtype::UC, kSealed);
  process_.map_pages(kBase + 4 * kLegacyPageSize, third + kLegacyPageSize, kLegacyPageSize,
                     Mtype::UC, LegacyHostExtentOwner::Application);
  const auto run = [&](unsigned page) { return extent(kBase + page * kLegacyPageSize); };
  ASSERT_TRUE(run(0) && run(2) && run(3) && run(4));
  EXPECT_EQ(run(1)->end, kBase + 2 * kLegacyPageSize);
  EXPECT_EQ(run(2)->begin, kBase + 2 * kLegacyPageSize);
  EXPECT_EQ(run(2)->end, kBase + 3 * kLegacyPageSize);
  EXPECT_EQ(run(3)->end, kBase + 4 * kLegacyPageSize);
  EXPECT_EQ(run(3)->mtype, Mtype::UC);
  EXPECT_EQ(run(4)->owner, LegacyHostExtentOwner::Application);
}

TEST_F(LegacyLocalExtentTest, MutationRetiresExtentsOfOlderGenerations) {
  uint8_t *host = pages(2);
  uint8_t *replacement = pages(1);
  process_.map_pages(kBase, host, 2 * kLegacyPageSize, Mtype::RW,
                     LegacyHostExtentOwner::DriverSealedRam);
  auto state = process_.page_table_cache_state();
  const uint64_t old_generation = state->generation();
  ASSERT_TRUE(extent(kBase));
  ASSERT_EQ(process_.remap_page_host_ptrs(kBase + kLegacyPageSize, host + kLegacyPageSize,
                                          replacement, kLegacyPageSize),
            KfdProcess::PageTableMutationResult::Applied);
  EXPECT_NE(state->generation(), old_generation);
  // Extents are answered per generation; a retired one can no longer be admitted.
  EXPECT_FALSE(state->try_acquire(old_generation));
  const auto fresh = extent(kBase + kLegacyPageSize);
  ASSERT_TRUE(fresh);
  EXPECT_FALSE(state->local_extents().find(kBase, old_generation));
  EXPECT_EQ(fresh->begin, kBase + kLegacyPageSize);
  EXPECT_EQ(fresh->host(kBase + kLegacyPageSize), replacement);
  ASSERT_EQ(process_.unmap_pages(kBase, 2 * kLegacyPageSize),
            KfdProcess::PageTableMutationResult::Applied);
  EXPECT_FALSE(extent(kBase));
}

TEST_F(LegacyLocalExtentTest, OtherThreadsReuseThePublishedExtent) {
  uint8_t *host = pages(3);
  process_.map_pages(kBase, host, 3 * kLegacyPageSize, Mtype::RW,
                     LegacyHostExtentOwner::DriverSealedRam);
  const uint64_t generation = process_.page_table_cache_state()->generation();
  ASSERT_TRUE(extent(kBase));
  EXPECT_TRUE(process_.page_table_cache_state()->local_extents().find(kBase + 9000, generation));
  std::optional<LegacyLocalExtent> seen;
  std::thread([&] { seen = extent(kBase + 2 * kLegacyPageSize); }).join();
  ASSERT_TRUE(seen);
  EXPECT_EQ(seen->begin, kBase);
}

TEST_F(LegacyLocalExtentTest, WideBoxBeyondTheOldCoverageCapIsLocal) {
  constexpr size_t kPages = 32 * 1024; // 128 MiB, past the old 512 KiB coverage cap.
  uint8_t *host = pages(kPages);
  ASSERT_EQ(process_.map_pages(kBase, host, kPages * kLegacyPageSize, Mtype::RW,
                               LegacyHostExtentOwner::DriverSealedRam),
            KfdProcess::PageTableMutationResult::Applied);
  const uint64_t end = kBase + uint64_t{kPages} * kLegacyPageSize;
  const MemoryIngressSpan span{kBase, kPages * kLegacyPageSize, VmAccessKind::Write};
  EXPECT_TRUE(space_.local_ingress_mapped({&span, 1}, kBase, end, kVmid,
                                          process_.page_table_request_mutex(),
                                          process_.page_table_cache_state()));
  const auto last = extent(end - 4);
  ASSERT_TRUE(last);
  EXPECT_EQ(last->host(end - 4), host + kPages * kLegacyPageSize - 4);
  // A hole inside the box is never reported as mapped.
  process_.unmap_pages(kBase + 7 * kLegacyPageSize, kLegacyPageSize);
  EXPECT_FALSE(space_.local_ingress_mapped({&span, 1}, kBase, end, kVmid,
                                           process_.page_table_request_mutex(),
                                           process_.page_table_cache_state()));
}

TEST(LegacyAddressSpaceTest, StaticTlsLeavesRoomForASmallThreadStack) {
  pthread_attr_t attr;
  ASSERT_EQ(pthread_attr_init(&attr), 0);
  ASSERT_EQ(pthread_attr_setstacksize(&attr, 64 * 1024), 0);
  pthread_t thread{};
  ASSERT_EQ(pthread_create(&thread, &attr, [](void *) -> void * { return nullptr; }, nullptr), 0);
  EXPECT_EQ(pthread_join(thread, nullptr), 0);
  EXPECT_EQ(pthread_attr_destroy(&attr), 0);
}

} // namespace
} // namespace rocjitsu::amdgpu
