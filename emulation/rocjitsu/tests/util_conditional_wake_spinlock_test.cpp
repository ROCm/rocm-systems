// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "util/conditional_wake_spinlock.h"

#include <gtest/gtest.h>

#if defined(__linux__)
#include "rocjitsu/kmd/linux/host_futex.h"
#include "rocjitsu/kmd/linux/libc_passthrough.h"

#include <cerrno>
#include <csignal>
#include <cstdarg>
#include <pthread.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <latch>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;

#if defined(__linux__)
using NativeLock = util::ConditionalWakeSpinlock<rocjitsu::HostFutexParking<>>;
#else
using NativeLock = util::ConditionalWakeSpinlock<void>;
#endif

// Long enough for a blocked thread to finish spinning and sleep in the kernel.
constexpr auto kParkDelay = 50ms;

// Only the test-runner thread accesses this sticky failure state.
bool detached_waiters = false;

class ConditionalWakeSpinlockTest : public testing::Test {
protected:
  void SetUp() override {
    if (detached_waiters) {
      GTEST_SKIP() << "A previous lock test left detached waiters";
    }
#if defined(__linux__)
    // Resolution is a single-threaded bootstrap operation, not a lock slow path.
    rocjitsu::libc_passthrough().resolve();
    ASSERT_TRUE(rocjitsu::libc_passthrough().ready());
#endif
  }
};

template <typename Predicate> bool eventually(Predicate predicate) {
  const auto deadline = std::chrono::steady_clock::now() + 30s;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() > deadline)
      return false;
    std::this_thread::sleep_for(1ms);
  }
  return true;
}

// A lost wakeup can leave threads blocked; detach them so the failure is
// reported instead of hanging in join().
void join_or_detach(std::vector<std::thread> &threads, bool completed) {
  if (!completed)
    detached_waiters = true;
  for (auto &thread : threads) {
    if (completed)
      thread.join();
    else
      thread.detach();
  }
}

TEST_F(ConditionalWakeSpinlockTest, FailedTryLocksPreserveWaiterProgress) {
  // Shared state must outlive any waiter detached after a test timeout.
  static NativeLock lock;
  static uint64_t counter = 0;
  static std::array<uint64_t, 8> published{};
  static std::atomic<bool> inconsistent{false};
  static std::atomic<uint64_t> finished{0};
  constexpr uint64_t kWaiters = 6;
  constexpr uint64_t kRounds = 32;
  counter = 0;
  published.fill(0);
  inconsistent = false;
  for (uint64_t round = 0; round < kRounds; ++round) {
    // A detached failure-path waiter must also keep its readiness latch alive.
    auto ready = std::make_shared<std::latch>(kWaiters);
    finished = 0;
    std::vector<std::thread> waiters;
    lock.lock();
    for (uint64_t index = 0; index < kWaiters; ++index) {
      waiters.emplace_back([ready] {
        ready->count_down();
        {
          std::lock_guard guard(lock);
          if (!std::ranges::all_of(published, [](uint64_t value) { return value == counter; }))
            inconsistent.store(true, std::memory_order_relaxed);
          published.fill(++counter);
        }
        finished.fetch_add(1);
      });
    }
    ready->wait();
    // Give contenders an opportunity to park before failed probes and the
    // owner's release. Every waiter must eventually receive ownership.
    std::this_thread::sleep_for(1ms);
    for (unsigned probe = 0; probe < 256; ++probe)
      EXPECT_FALSE(lock.try_lock());
    lock.unlock();
    const bool completed = eventually([] { return finished.load() == kWaiters; });
    EXPECT_TRUE(completed);
    join_or_detach(waiters, completed);
    if (!completed)
      return;
    EXPECT_EQ(counter, (round + 1) * kWaiters);
  }
  EXPECT_FALSE(inconsistent.load(std::memory_order_relaxed));
  EXPECT_TRUE(std::ranges::all_of(published, [](uint64_t value) { return value == counter; }));
}

TEST_F(ConditionalWakeSpinlockTest, ParkedWaiterAcquiresAfterFailedTryLocks) {
  // Shared state must outlive any waiter detached after a test timeout.
  static NativeLock lock;
  static std::atomic<bool> acquired{false};
  acquired = false;
  lock.lock();
  std::vector<std::thread> threads;
  threads.emplace_back([] {
    lock.lock();
    acquired = true;
    lock.unlock();
  });
  std::this_thread::sleep_for(kParkDelay);
  for (int attempt = 0; attempt < 16; ++attempt)
    EXPECT_FALSE(lock.try_lock());
  lock.unlock();
  const bool completed = eventually([] { return acquired.load(); });
  EXPECT_TRUE(completed);
  join_or_detach(threads, completed);
}

TEST_F(ConditionalWakeSpinlockTest, ParkedWaitersAcquireThroughNewcomerHandoffs) {
  // Shared state must outlive any waiter detached after a test timeout.
  static NativeLock lock;
  static std::atomic<unsigned> parked_done{0};
  static std::atomic<unsigned> newcomers_done{0};
  static std::atomic<bool> stop_newcomers{false};
  constexpr unsigned kParked = 6;
  constexpr unsigned kNewcomers = 2;
  parked_done = 0;
  newcomers_done = 0;
  stop_newcomers = false;
  lock.lock();
  std::vector<std::thread> parked;
  for (unsigned i = 0; i < kParked; ++i) {
    parked.emplace_back([] {
      lock.lock();
      std::this_thread::sleep_for(1ms);
      lock.unlock();
      parked_done.fetch_add(1);
    });
  }
  std::this_thread::sleep_for(kParkDelay);
  // Newcomers keep taking the uncontended path between wakeups, so a woken
  // waiter often finds the lock held again.
  std::vector<std::thread> newcomers;
  for (unsigned i = 0; i < kNewcomers; ++i) {
    newcomers.emplace_back([] {
      while (!stop_newcomers.load(std::memory_order_relaxed)) {
        if (lock.try_lock())
          lock.unlock();
        std::lock_guard guard(lock);
      }
      newcomers_done.fetch_add(1);
    });
  }
  lock.unlock();
  const bool completed = eventually([] { return parked_done.load() == kParked; });
  EXPECT_TRUE(completed);
  stop_newcomers = true;
  // A newcomer can also be stranded behind a lost wakeup.
  const bool newcomers_completed = eventually([] { return newcomers_done.load() == kNewcomers; });
  EXPECT_TRUE(newcomers_completed);
  join_or_detach(newcomers, newcomers_completed);
  join_or_detach(parked, completed);
}

TEST_F(ConditionalWakeSpinlockTest, MutualExclusionAcrossParkingAndFailedTryLocks) {
  // Shared state must outlive any waiter detached after a test timeout.
  static NativeLock lock;
  static std::atomic<unsigned> inside{0};
  static std::atomic<unsigned> finished{0};
  static std::atomic<bool> overlap{false};
  static uint64_t counter = 0;
  constexpr unsigned kThreads = 8;
  constexpr unsigned kIterations = 20000;
  inside = 0;
  finished = 0;
  overlap = false;
  counter = 0;
  std::vector<std::thread> threads;
  for (unsigned t = 0; t < kThreads; ++t) {
    threads.emplace_back([t] {
      for (unsigned i = 0; i < kIterations; ++i) {
        if ((i + t) % 3 == 0) {
          if (!lock.try_lock())
            lock.lock();
        } else {
          lock.lock();
        }
        if (inside.fetch_add(1, std::memory_order_relaxed) != 0)
          overlap = true;
        ++counter;
        // Occasional long holds force the other threads to park.
        if (i % 4096 == t)
          std::this_thread::sleep_for(200us);
        inside.fetch_sub(1, std::memory_order_relaxed);
        lock.unlock();
      }
      finished.fetch_add(1);
    });
  }
  const bool completed = eventually([] { return finished.load() == kThreads; });
  EXPECT_TRUE(completed);
  join_or_detach(threads, completed);
  if (completed) {
    EXPECT_FALSE(overlap.load());
    EXPECT_EQ(counter, uint64_t{kThreads} * kIterations);
  }
}

#if defined(__linux__)
// The provider is deliberately distinct from the process's syscall symbol.
// Bypassing either policy call loses the corresponding recorded operation.
struct RecordingSyscalls {
  struct Trace {
    std::atomic<unsigned> waits{0};
    std::atomic<unsigned> wakes{0};
    std::atomic<unsigned> errors{0};
    std::atomic<bool> invalid{false};
    std::atomic<bool> blocked{false};
    std::atomic<bool> resume{false};
    std::atomic<uint32_t *> word{nullptr};
    std::array<long, 9> timeouts{};
  };

  static Trace &trace() {
    static Trace value;
    return value;
  }

  static long invoke(long number, ...) {
    auto &record = trace();
    if (number != SYS_futex) {
      record.invalid = true;
      errno = ENOSYS;
      return -1;
    }
    va_list args;
    va_start(args, number);
    auto *word = va_arg(args, uint32_t *);
    const int operation = va_arg(args, int);
    if (operation == FUTEX_WAIT_PRIVATE) {
      const uint32_t expected = va_arg(args, uint32_t);
      const auto *timeout = va_arg(args, const timespec *);
      const auto *second_word = va_arg(args, void *);
      const int bitset = va_arg(args, int);
      va_end(args);
      const auto index = record.waits.fetch_add(1);
      auto *unset = static_cast<uint32_t *>(nullptr);
      record.word.compare_exchange_strong(unset, word);
      if (word != record.word.load() || expected != 1 || timeout == nullptr ||
          second_word != nullptr || bitset != 0) {
        record.invalid = true;
      }
      if (timeout != nullptr) {
        if (timeout->tv_sec != 0 || timeout->tv_nsec <= 0 || timeout->tv_nsec > 2'000'000)
          record.invalid = true;
        if (index < record.timeouts.size())
          record.timeouts[index] = timeout->tv_nsec;
      }
      // Keep the waiter registered while its owner releases. This makes the
      // wake check deterministic without relying on scheduling delays.
      if (index + 1 == record.timeouts.size()) {
        record.blocked.store(true, std::memory_order_release);
        while (!record.resume.load(std::memory_order_acquire))
          std::this_thread::yield();
      }
      constexpr std::array errors{EINTR, EAGAIN, ETIMEDOUT};
      record.errors.fetch_or(1u << (index % errors.size()));
      errno = errors[index % errors.size()];
      return -1;
    }
    if (operation == FUTEX_WAKE_PRIVATE) {
      const int count = va_arg(args, int);
      const auto *timeout = va_arg(args, void *);
      const auto *second_word = va_arg(args, void *);
      const int bitset = va_arg(args, int);
      va_end(args);
      if (word != record.word.load() || count != 1 || timeout != nullptr ||
          second_word != nullptr || bitset != 0)
        record.invalid = true;
      record.wakes.fetch_add(1);
      return 0;
    }
    va_end(args);
    record.invalid = true;
    errno = ENOSYS;
    return -1;
  }

  static auto get() { return &invoke; }
};

TEST_F(ConditionalWakeSpinlockTest, ParkingUsesProvidedHostWaitAndWake) {
  using Lock = util::ConditionalWakeSpinlock<rocjitsu::HostFutexParking<RecordingSyscalls>>;
  // Static state also survives a failure that detaches a waiter.
  static Lock lock;
  static std::atomic<bool> completed{false};
  static uint64_t published = 0;
  static uint64_t observed = 0;
  auto &record = RecordingSyscalls::trace();
  record.waits = 0;
  record.wakes = 0;
  record.errors = 0;
  record.invalid = false;
  record.blocked = false;
  record.resume = false;
  record.word = nullptr;
  record.timeouts.fill(0);
  completed = false;
  published = 0;
  observed = 0;
  lock.lock();
  std::vector<std::thread> threads;
  threads.emplace_back([] {
    lock.lock();
    observed = published;
    lock.unlock();
    completed.store(true, std::memory_order_release);
  });
  const bool parked = eventually([&record] { return record.blocked.load(); });
  EXPECT_TRUE(parked);
  EXPECT_FALSE(completed.load());
  for (unsigned probe = 0; probe < 16; ++probe)
    EXPECT_FALSE(lock.try_lock());
  published = 0x0123456789abcdef;
  lock.unlock();
  // Even when an expectation fails, release the gate and the original owner
  // before waiting for cleanup, so a routing regression reports a failure.
  record.resume.store(true, std::memory_order_release);
  const bool done = eventually([] { return completed.load(std::memory_order_acquire); });
  EXPECT_TRUE(done);
  join_or_detach(threads, done);
  if (!done)
    return;
  EXPECT_EQ(observed, published);
  EXPECT_EQ(record.waits.load(), 9u);
  EXPECT_EQ(record.wakes.load(), 1u);
  EXPECT_EQ(record.errors.load(), 7u);
  EXPECT_FALSE(record.invalid.load());
  constexpr std::array<long, 9> expected_timeouts{
      50'000, 100'000, 200'000, 400'000, 800'000, 1'600'000, 2'000'000, 2'000'000, 2'000'000};
  EXPECT_EQ(record.timeouts, expected_timeouts);
}

struct NullSyscalls {
  inline static std::atomic<unsigned> lookups{0};

  static auto get() -> long (*)(long, ...) {
    lookups.fetch_add(1);
    return nullptr;
  }
};

TEST_F(ConditionalWakeSpinlockTest, UnresolvedHostProviderKeepsProgressAndPublication) {
  using Lock = util::ConditionalWakeSpinlock<rocjitsu::HostFutexParking<NullSyscalls>>;
  static Lock lock;
  static std::atomic<bool> completed{false};
  static uint64_t published = 0;
  static uint64_t observed = 0;
  NullSyscalls::lookups = 0;
  completed = false;
  published = 0;
  observed = 0;
  lock.lock();
  std::vector<std::thread> threads;
  threads.emplace_back([] {
    lock.lock();
    observed = published;
    lock.unlock();
    completed.store(true, std::memory_order_release);
  });
  EXPECT_TRUE(eventually([] { return NullSyscalls::lookups.load() != 0; }));
  EXPECT_FALSE(completed.load());
  published = 0xfedcba9876543210;
  lock.unlock();
  const bool done = eventually([] { return completed.load(std::memory_order_acquire); });
  EXPECT_TRUE(done);
  join_or_detach(threads, done);
  if (done) {
    EXPECT_EQ(observed, published);
  }
}

TEST_F(ConditionalWakeSpinlockTest, SignalInterruptedParksKeepExclusionAndProgress) {
  // Shared state must outlive any waiter detached after a test timeout.
  static NativeLock lock;
  static std::atomic<unsigned> inside{0};
  static std::atomic<unsigned> finished{0};
  static std::atomic<bool> overlap{false};
  static std::atomic<bool> stop_signals{false};
  static std::atomic<bool> release_waiters{false};
  constexpr unsigned kThreads = 4;
  constexpr unsigned kRounds = 200;
  inside = 0;
  finished = 0;
  overlap = false;
  stop_signals = false;
  release_waiters = false;
  // Without SA_RESTART, a signal that arrives during a parked wait ends it
  // early with EINTR.
  struct sigaction action {};
  struct sigaction previous {};
  action.sa_handler = [](int) {};
  sigemptyset(&action.sa_mask);
  ASSERT_EQ(sigaction(SIGUSR1, &action, &previous), 0);
  std::vector<std::thread> threads;
  for (unsigned t = 0; t < kThreads; ++t) {
    threads.emplace_back([] {
      for (unsigned round = 0; round < kRounds; ++round) {
        std::lock_guard guard(lock);
        if (inside.fetch_add(1, std::memory_order_relaxed) != 0)
          overlap = true;
        // Periodic long holds make the other threads park.
        if (round % 8 == 0)
          std::this_thread::sleep_for(2ms);
        inside.fetch_sub(1, std::memory_order_relaxed);
      }
      finished.fetch_add(1);
      // Stay alive until signals stop so none targets an exited thread.
      while (!release_waiters.load())
        std::this_thread::sleep_for(1ms);
    });
  }
  std::vector<pthread_t> handles;
  for (auto &thread : threads)
    handles.push_back(thread.native_handle());
  std::thread signaler([handles] {
    while (!stop_signals.load()) {
      for (pthread_t handle : handles)
        pthread_kill(handle, SIGUSR1);
      std::this_thread::sleep_for(100us);
    }
  });
  const bool completed = eventually([] { return finished.load() == kThreads; });
  stop_signals = true;
  signaler.join();
  release_waiters = true;
  EXPECT_TRUE(completed);
  join_or_detach(threads, completed);
  sigaction(SIGUSR1, &previous, nullptr);
  EXPECT_FALSE(overlap.load());
}
#endif

} // namespace
