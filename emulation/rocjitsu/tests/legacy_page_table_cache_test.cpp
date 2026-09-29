// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/legacy_page_table.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <thread>
#include <unistd.h>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

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
                alarm(10);
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

} // namespace
} // namespace rocjitsu::amdgpu
