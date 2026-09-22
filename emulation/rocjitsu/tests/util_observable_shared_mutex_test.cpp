// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "util/observable_shared_mutex.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using util::ObservableSharedMutex;

// Spin rather than sleep: the count is the thing under test, so waiting on it
// is the point. A deadline keeps a broken implementation from hanging the suite.
[[nodiscard]] bool wait_for_blocked_writers(ObservableSharedMutex &mutex, uint64_t expected) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (mutex.blocked_writers() != expected) {
    if (std::chrono::steady_clock::now() > deadline)
      return false;
    std::this_thread::yield();
  }
  return true;
}

TEST(ObservableSharedMutexTest, ReportsNoWaitersWhenIdle) {
  ObservableSharedMutex mutex;
  EXPECT_EQ(mutex.blocked_writers(), 0u);
}

TEST(ObservableSharedMutexTest, AnUncontendedWriterIsNotLeftCounted) {
  ObservableSharedMutex mutex;
  {
    auto guard = mutex.lock_exclusive();
    // Counted only while it waits, and this one never did.
    EXPECT_EQ(mutex.blocked_writers(), 0u);
  }
  EXPECT_EQ(mutex.blocked_writers(), 0u);
}

TEST(ObservableSharedMutexTest, ReadersAreNotCounted) {
  ObservableSharedMutex mutex;
  auto first = mutex.lock_shared();
  auto second = mutex.lock_shared();
  EXPECT_EQ(mutex.blocked_writers(), 0u)
      << "shared ownership is the uncontended case the count deliberately ignores";
}

// The property the whole class exists for: a writer held off by a reader can be
// waited for, so exclusion becomes something a test observes rather than infers
// from elapsed time.
TEST(ObservableSharedMutexTest, AWriterBlockedByAReaderIsCountedUntilItAcquires) {
  ObservableSharedMutex mutex;
  std::atomic<bool> acquired = false;

  // The writer outlives the reader's scope on purpose. An ASSERT_* below returns
  // from the function, and the reader must have released by the time the
  // writer is joined -- otherwise a failing assertion hangs on a thread that
  // can never acquire, instead of reporting what went wrong.
  std::jthread writer;
  {
    auto reader = mutex.lock_shared();
    writer = std::jthread([&] {
      auto guard = mutex.lock_exclusive();
      acquired.store(true, std::memory_order_release);
    });

    ASSERT_TRUE(wait_for_blocked_writers(mutex, 1)) << "the writer never registered as blocked";
    EXPECT_FALSE(acquired.load(std::memory_order_acquire))
        << "a writer acquired the lock while a reader still held it";
  }
  writer.join();
  EXPECT_TRUE(acquired.load(std::memory_order_acquire));
  EXPECT_EQ(mutex.blocked_writers(), 0u) << "the count outlived the wait";
}

TEST(ObservableSharedMutexTest, EveryBlockedWriterIsCounted) {
  ObservableSharedMutex mutex;
  constexpr uint64_t kWriters = 3;

  std::vector<std::jthread> writers;
  {
    auto reader = mutex.lock_shared();
    for (uint64_t i = 0; i < kWriters; ++i)
      writers.emplace_back([&] { auto guard = mutex.lock_exclusive(); });

    ASSERT_TRUE(wait_for_blocked_writers(mutex, kWriters));
  }
  for (auto &writer : writers)
    writer.join();
  EXPECT_EQ(mutex.blocked_writers(), 0u);
}

TEST(ObservableSharedMutexTest, ExclusiveOwnershipHoldsOffAReader) {
  ObservableSharedMutex mutex;
  std::atomic<bool> read_entered = false;

  std::jthread reader;
  {
    auto guard = mutex.lock_exclusive();
    reader = std::jthread([&] {
      auto shared = mutex.lock_shared();
      read_entered.store(true, std::memory_order_release);
    });

    // Nothing to wait on here -- readers are uncounted by design -- so this is the
    // one place elapsed time is the only evidence available. A false pass would
    // mean the reader was merely slow, which the join below then contradicts.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(read_entered.load(std::memory_order_acquire))
        << "a reader entered while the lock was held exclusively";
  }
  reader.join();
  EXPECT_TRUE(read_entered.load(std::memory_order_acquire));
}

// Acquiring can throw -- std::shared_timed_mutex::lock() reports system errors
// that way -- and a throw past the count would leave a writer that does not
// exist. Nothing lowers it again, so a later wait for it to fall hangs and a
// wait for it to rise passes without anything having blocked.
TEST(ObservableSharedMutexTest, TheWaitingCountUnwindsWhenAcquisitionThrows) {
  ObservableSharedMutex mutex;
  ASSERT_EQ(mutex.blocked_writers(), 0u);

  try {
    const ObservableSharedMutex::ExclusiveGuard::CountedWait counted(mutex);
    EXPECT_EQ(mutex.blocked_writers(), 1u) << "the wait was never counted";
    throw std::runtime_error("acquisition failed");
  } catch (const std::runtime_error &) {
  }

  EXPECT_EQ(mutex.blocked_writers(), 0u)
      << "a throw past the count leaves a phantom writer that nothing ever clears";
}

} // namespace
