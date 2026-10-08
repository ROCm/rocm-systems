// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/l2_maintenance_mutex.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <future>
#include <latch>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

namespace rocjitsu::amdgpu {

class L2MaintenanceMutexTestAccess {
public:
  static constexpr size_t reader_shards() { return L2MaintenanceMutex::kReaderShards; }
  static size_t reader_shard() { return L2MaintenanceMutex::reader_shard(); }
  template <typename Hooks> static void lock_shared(L2MaintenanceMutex &mutex, Hooks hooks) {
    mutex.lock_shared_impl(hooks);
  }
  template <typename Hooks> static bool try_lock_shared(L2MaintenanceMutex &mutex, Hooks hooks) {
    return mutex.try_lock_shared_impl(hooks);
  }
  template <typename Hooks> static void lock(L2MaintenanceMutex &mutex, Hooks hooks) {
    mutex.lock_impl(hooks);
  }
  static std::atomic<unsigned> &reader_count(L2MaintenanceMutex &mutex, size_t shard) {
    return mutex.readers_[shard].count;
  }
};

} // namespace rocjitsu::amdgpu

namespace {
using MutexTestAccess = rocjitsu::amdgpu::L2MaintenanceMutexTestAccess;

TEST(L2MaintenanceMutexTest, ReaderRechecksAdmissionAfterWriterEnters) {
  for (bool blocking : {false, true}) {
    SCOPED_TRACE(blocking);
    rocjitsu::amdgpu::L2MaintenanceMutex mutex;
    std::latch checked_open(1), resume_reader(1);
    std::promise<bool> reader_outcome;
    auto outcome = reader_outcome.get_future().share();
    struct Hooks {
      std::latch &checked_open, &resume_reader;
      std::promise<bool> &outcome;
      bool first = true;
      void before_reader_increment() {
        if (first) {
          first = false;
          checked_open.count_down();
          resume_reader.wait();
        }
      }
      void reader_retry() { outcome.set_value(false); }
    };
    std::jthread reader([&] {
      Hooks hooks{checked_open, resume_reader, reader_outcome};
      if (blocking) {
        MutexTestAccess::lock_shared(mutex, hooks);
        // If the recheck was removed, entry precedes the retry notification.
        if (outcome.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
          reader_outcome.set_value(true);
        mutex.unlock_shared();
      } else {
        const bool acquired = MutexTestAccess::try_lock_shared(mutex, hooks);
        reader_outcome.set_value(acquired);
        if (acquired)
          mutex.unlock_shared();
      }
    });
    checked_open.wait();
    std::unique_lock writer(mutex);
    resume_reader.count_down();
    // A handshake reports either retry/rejection or premature entry. No sleep
    // is used to infer that the reader reached its admission recheck.
    EXPECT_FALSE(outcome.get());
    writer.unlock();
    reader.join();
  }
}

TEST(L2MaintenanceMutexTest, WriterWaitsAgainAfterNonzeroCounterWake) {
  rocjitsu::amdgpu::L2MaintenanceMutex mutex;
  // Pin one real counter at two owners without relying on thread-ID collisions.
  auto &counter = MutexTestAccess::reader_count(mutex, MutexTestAccess::reader_shard());
  counter.store(2);
  std::latch waiting_for_two(1), resume_writer(1);
  std::promise<bool> writer_outcome;
  auto outcome = writer_outcome.get_future().share();
  struct Hooks {
    std::latch &waiting_for_two, &resume_writer;
    std::promise<bool> &outcome;
    void before_writer_wait(unsigned active) {
      if (active == 2) {
        waiting_for_two.count_down();
        resume_writer.wait();
      } else {
        EXPECT_EQ(active, 1u);
        outcome.set_value(false);
      }
    }
  };
  std::jthread writer([&] {
    MutexTestAccess::lock(mutex, Hooks{waiting_for_two, resume_writer, writer_outcome});
    if (outcome.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
      writer_outcome.set_value(true);
    mutex.unlock();
  });
  waiting_for_two.wait();
  counter.store(1);
  counter.notify_one();
  resume_writer.count_down();
  // wait(2) returns on the changed value, but one reader still owns the shard.
  // The next handshake must be another wait, not entry into the writer section.
  EXPECT_FALSE(outcome.get());
  counter.store(0);
  counter.notify_one();
  writer.join();
}

TEST(L2MaintenanceMutexTest, CoherenceDomainWritersUnderOuterLocks) {
  // Model eight XCD caches beneath two coordinator locks. Exercise both
  // writer acquisition paths without a separate mutex acquisition for each reader counter.
  constexpr size_t kCaches = 8;
  std::mutex atomic, coherence;
  std::lock_guard atomic_lock(atomic);
  std::lock_guard coherence_lock(coherence);
  std::array<rocjitsu::amdgpu::L2MaintenanceMutex, kCaches> mutexes;
  std::array<std::unique_lock<rocjitsu::amdgpu::L2MaintenanceMutex>, kCaches> writers;
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

TEST(L2MaintenanceMutexTest, FailedWriterTryReopensReaderAdmission) {
  rocjitsu::amdgpu::L2MaintenanceMutex mutex;
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
  // Verify failure cleanup before any later exclusive unlock can reopen it.
  const bool new_reader = mutex.try_lock_shared();
  EXPECT_TRUE(new_reader);
  if (new_reader)
    mutex.unlock_shared();
  release.set_value();
  reader.join();
  ASSERT_TRUE(mutex.try_lock());
  mutex.unlock();
  ASSERT_TRUE(mutex.try_lock_shared());
  mutex.unlock_shared();
}

TEST(L2MaintenanceMutexTest, WriterExcludesReadersAndOtherWriters) {
  rocjitsu::amdgpu::L2MaintenanceMutex mutex;
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

TEST(L2MaintenanceMutexTest, ConcurrentReadersObserveCompleteWriterUpdates) {
  rocjitsu::amdgpu::L2MaintenanceMutex mutex;
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

TEST(L2MaintenanceMutexTest, PendingWriterClosesAdmissionAndWakesAfterLastReader) {
  rocjitsu::amdgpu::L2MaintenanceMutex mutex;
  std::shared_lock existing_reader(mutex);
  std::promise<void> writer_acquired;
  auto acquired = writer_acquired.get_future();
  std::jthread writer([&] {
    std::unique_lock lock(mutex);
    writer_acquired.set_value();
  });
  bool rejected_new_reader = false;
  std::jthread contender([&] {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
      std::shared_lock candidate(mutex, std::try_to_lock);
      if (!candidate.owns_lock()) {
        rejected_new_reader = true;
        return;
      }
      std::this_thread::yield();
    }
  });
  contender.join();
  EXPECT_TRUE(rejected_new_reader);
  EXPECT_EQ(acquired.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
  existing_reader.unlock();
  EXPECT_EQ(acquired.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  writer.join();
  ASSERT_TRUE(mutex.try_lock_shared());
  mutex.unlock_shared();
}

TEST(L2MaintenanceMutexTest, CollidingReadersOverlapAndWriterWaitsForTheLastOwner) {
  rocjitsu::amdgpu::L2MaintenanceMutex mutex;
  constexpr size_t kReaders = MutexTestAccess::reader_shards() + 1;
  std::array<size_t, kReaders> slots{};
  std::mutex state_mutex;
  std::condition_variable held_changed;
  size_t held = 0;
  std::promise<void> release_others, release_last;
  const auto others_released = release_others.get_future().share();
  const auto last_released = release_last.get_future().share();
  size_t last_reader = 0;
  std::vector<std::jthread> readers;
  for (size_t i = 0; i < kReaders; ++i) {
    readers.emplace_back([&, i] {
      slots[i] = MutexTestAccess::reader_shard();
      std::shared_lock lock(mutex);
      {
        std::lock_guard state_lock(state_mutex);
        ++held;
      }
      held_changed.notify_one();
      others_released.wait();
      if (i == last_reader)
        last_released.wait();
    });
  }
  bool all_held;
  {
    std::unique_lock state_lock(state_mutex);
    all_held = held_changed.wait_for(state_lock, std::chrono::seconds(5),
                                     [&] { return held == kReaders; });
  }
  EXPECT_TRUE(all_held);
  if (!all_held) {
    release_others.set_value();
    release_last.set_value();
    readers.clear();
    return;
  }
  // One more live reader than shards guarantees a collision with the actual policy.
  // Keep one colliding owner last, after every other owner of its slot drains.
  bool found_collision = false;
  for (size_t i = 0; i < kReaders && !found_collision; ++i) {
    for (size_t j = i + 1; j < kReaders; ++j) {
      if (slots[i] == slots[j]) {
        last_reader = i;
        found_collision = true;
        break;
      }
    }
  }
  EXPECT_TRUE(found_collision);
  std::promise<void> writer_acquired;
  auto acquired = writer_acquired.get_future();
  std::jthread writer([&] {
    std::unique_lock lock(mutex);
    writer_acquired.set_value();
  });
  bool admission_closed = false;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    std::shared_lock contender(mutex, std::try_to_lock);
    if (!contender.owns_lock()) {
      admission_closed = true;
      break;
    }
    std::this_thread::yield();
  }
  EXPECT_TRUE(admission_closed);
  release_others.set_value();
  for (size_t i = 0; i < kReaders; ++i)
    if (i != last_reader)
      readers[i].join();
  EXPECT_EQ(acquired.wait_for(std::chrono::milliseconds(10)), std::future_status::timeout);
  release_last.set_value();
  readers.clear();
  EXPECT_EQ(acquired.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  writer.join();
}

TEST(L2MaintenanceMutexTest, ExclusiveUnlockWakesBlockingReaders) {
  rocjitsu::amdgpu::L2MaintenanceMutex mutex;
  std::unique_lock writer(mutex);
  constexpr size_t kReaders = 8;
  std::latch started(kReaders);
  std::atomic<size_t> admitted = 0;
  std::vector<std::jthread> readers;
  for (size_t i = 0; i < kReaders; ++i) {
    readers.emplace_back([&] {
      started.count_down();
      std::shared_lock lock(mutex);
      admitted.fetch_add(1, std::memory_order_relaxed);
    });
  }
  started.wait();
  EXPECT_EQ(admitted.load(std::memory_order_relaxed), 0u);
  writer.unlock();
  readers.clear();
  EXPECT_EQ(admitted.load(std::memory_order_relaxed), kReaders);
}

} // namespace
