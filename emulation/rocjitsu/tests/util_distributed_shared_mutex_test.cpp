// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "util/distributed_shared_mutex.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <barrier>
#include <future>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

namespace {

TEST(DistributedSharedMutexTest, CoherenceDomainWritersUnderOuterLocks) {
  // Model eight XCD caches beneath two coordinator locks. Exercise both
  // writer acquisition paths without exhausting TSan's smallest lock table.
  constexpr size_t kCaches = 8;
  std::mutex atomic, coherence;
  std::lock_guard atomic_lock(atomic);
  std::lock_guard coherence_lock(coherence);
  std::array<util::DistributedSharedMutex, kCaches> mutexes;
  std::array<std::unique_lock<util::DistributedSharedMutex>, kCaches> writers;
  for (size_t i = 0; i < kCaches; ++i) {
    if (i < kCaches / 2)
      writers[i] = std::unique_lock(mutexes[i]);
    else
      writers[i] = std::unique_lock(mutexes[i], std::try_to_lock);
    ASSERT_TRUE(writers[i].owns_lock());
  }

  std::jthread reader([&] {
    for (auto &mutex : mutexes) {
      std::shared_lock lock(mutex, std::try_to_lock);
      EXPECT_FALSE(lock.owns_lock());
    }
  });
  reader.join();
}

TEST(DistributedSharedMutexTest, FailedWriterTryReleasesEveryAcquiredShard) {
  util::DistributedSharedMutex mutex;
  std::promise<void> held;
  std::promise<void> release;
  auto released = release.get_future();
  std::jthread reader([&] {
    std::shared_lock lock(mutex);
    held.set_value();
    released.wait();
  });
  held.get_future().wait();
  EXPECT_FALSE(mutex.try_lock());
  release.set_value();
  reader.join();
  ASSERT_TRUE(mutex.try_lock());
  mutex.unlock();
  ASSERT_TRUE(mutex.try_lock_shared());
  mutex.unlock_shared();
}

TEST(DistributedSharedMutexTest, WriterExcludesReadersAndOtherWriters) {
  util::DistributedSharedMutex mutex;
  std::unique_lock writer(mutex);
  std::jthread contender([&] {
    const bool read_acquired = mutex.try_lock_shared();
    EXPECT_FALSE(read_acquired);
    if (read_acquired)
      mutex.unlock_shared();
    const bool write_acquired = mutex.try_lock();
    EXPECT_FALSE(write_acquired);
    if (write_acquired)
      mutex.unlock();
  });
  contender.join();
}

TEST(DistributedSharedMutexTest, ConcurrentReadersObserveCompleteWriterUpdates) {
  util::DistributedSharedMutex mutex;
  constexpr unsigned kReaders = 16;
  constexpr unsigned kWriters = 3;
  constexpr unsigned kIterations = 1000;
  std::barrier start(kReaders + kWriters);
  unsigned first = 0;
  unsigned second = 0;
  std::atomic<bool> inconsistent = false;
  std::vector<std::jthread> threads;
  for (unsigned i = 0; i < kReaders; ++i) {
    threads.emplace_back([&] {
      start.arrive_and_wait();
      for (unsigned j = 0; j < kIterations; ++j) {
        std::shared_lock lock(mutex);
        if (first != second)
          inconsistent.store(true, std::memory_order_relaxed);
      }
    });
  }
  for (unsigned i = 0; i < kWriters; ++i) {
    threads.emplace_back([&] {
      start.arrive_and_wait();
      for (unsigned j = 0; j < kIterations; ++j) {
        std::unique_lock lock(mutex);
        ++first;
        std::this_thread::yield();
        ++second;
      }
    });
  }
  threads.clear();
  EXPECT_FALSE(inconsistent.load());
  EXPECT_EQ(first, kWriters * kIterations);
  EXPECT_EQ(second, first);
}

} // namespace
