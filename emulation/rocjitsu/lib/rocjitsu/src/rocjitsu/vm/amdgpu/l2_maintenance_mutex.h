// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file l2_maintenance_mutex.h
/// @brief Reader admission and exclusion for device cache maintenance.
///
/// Readers hash their thread IDs into 128 cache-line-separated counters;
/// collisions allow concurrent readers. A mutex serializes writers, which close
/// one admission gate and wait for all counters to drain.
///
/// Sequentially consistent admission orders each reader's increment and gate
/// recheck against the writer's gate close and counter scan: a reader is either
/// counted by the writer or retries without accessing cache state. Reader
/// decrements publish accesses to the writer; reopening the gate publishes
/// maintenance to subsequent readers. Failed exclusive try_lock reopens the gate.
///
/// Ownership is nonrecursive and must be released by the acquiring thread.
/// TSan annotations expose one shared/exclusive lock without changing the atomic
/// protocol. Device maintenance nests multiple L2 locks beneath coordinator
/// locks; annotating every counter would exhaust TSan's held-lock tracker.
/// Wakeup ordering is documented at release_reader().

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>

#if defined(__SANITIZE_THREAD__)
#define ROCJITSU_L2_MUTEX_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define ROCJITSU_L2_MUTEX_TSAN 1
#endif
#endif
#ifdef ROCJITSU_L2_MUTEX_TSAN
#include <sanitizer/tsan_interface.h>
#endif

namespace rocjitsu::amdgpu {

/// @brief Exclude L2 maintenance while distributing cache access over 128 counters.
class L2MaintenanceMutex {
public:
#ifdef ROCJITSU_L2_MUTEX_TSAN
  L2MaintenanceMutex() { __tsan_mutex_create(this, 0); }
  ~L2MaintenanceMutex() { __tsan_mutex_destroy(this, 0); }
#endif

  void lock_shared() { lock_shared_impl(NoopHooks{}); }
  bool try_lock_shared() { return try_lock_shared_impl(NoopHooks{}); }

  void unlock_shared() {
#ifdef ROCJITSU_L2_MUTEX_TSAN
    __tsan_mutex_pre_unlock(this, __tsan_mutex_read_lock);
#endif
    release_reader(readers_[reader_shard()].count);
#ifdef ROCJITSU_L2_MUTEX_TSAN
    __tsan_mutex_post_unlock(this, __tsan_mutex_read_lock);
#endif
  }

  void lock() { lock_impl(NoopHooks{}); }

  bool try_lock() {
#ifdef ROCJITSU_L2_MUTEX_TSAN
    __tsan_mutex_pre_lock(this, __tsan_mutex_try_lock);
#endif
    bool acquired = writers_.try_lock();
    if (acquired) {
      writer_pending_.store(1, std::memory_order_seq_cst);
      for (auto &reader : readers_) {
        if (reader.count.load(std::memory_order_seq_cst) != 0) {
          reopen_admission();
          acquired = false;
          break;
        }
      }
    }
#ifdef ROCJITSU_L2_MUTEX_TSAN
    __tsan_mutex_post_lock(
        this, __tsan_mutex_try_lock | (acquired ? 0 : __tsan_mutex_try_lock_failed), 0);
#endif
    return acquired;
  }

  void unlock() {
#ifdef ROCJITSU_L2_MUTEX_TSAN
    __tsan_mutex_pre_unlock(this, 0);
#endif
    reopen_admission();
#ifdef ROCJITSU_L2_MUTEX_TSAN
    __tsan_mutex_post_unlock(this, 0);
#endif
  }

private:
  friend class L2MaintenanceMutexTestAccess;

  // Compile-time hooks let tests control interleavings in the actual protocol.
  // Normal acquisitions inline empty hooks without storing any test state.
  struct NoopHooks {
    void before_reader_increment() const {}
    void reader_retry() const {}
    void before_writer_wait(unsigned) const {}
  };

  template <typename Hooks> void lock_shared_impl(Hooks hooks) {
#ifdef ROCJITSU_L2_MUTEX_TSAN
    __tsan_mutex_pre_lock(this, __tsan_mutex_read_lock);
#endif
    auto &counter = readers_[reader_shard()].count;
    for (;;) {
      while (writer_pending_.load(std::memory_order_acquire))
        writer_pending_.wait(1, std::memory_order_acquire);
      // The counter increment, gate recheck, writer gate store, and writer
      // counter loads form one total order. Either the writer observes an
      // admitted reader, or that reader observes the closed gate and retries.
      hooks.before_reader_increment();
      counter.fetch_add(1, std::memory_order_seq_cst);
      if (!writer_pending_.load(std::memory_order_seq_cst)) {
#ifdef ROCJITSU_L2_MUTEX_TSAN
        __tsan_mutex_post_lock(this, __tsan_mutex_read_lock, 0);
#endif
        return;
      }
      release_reader(counter);
      hooks.reader_retry();
    }
  }

  template <typename Hooks> bool try_lock_shared_impl(Hooks hooks) {
#ifdef ROCJITSU_L2_MUTEX_TSAN
    constexpr unsigned flags = __tsan_mutex_read_lock | __tsan_mutex_try_lock;
    __tsan_mutex_pre_lock(this, flags);
#endif
    bool acquired = false;
    if (!writer_pending_.load(std::memory_order_acquire)) {
      auto &counter = readers_[reader_shard()].count;
      hooks.before_reader_increment();
      counter.fetch_add(1, std::memory_order_seq_cst);
      acquired = !writer_pending_.load(std::memory_order_seq_cst);
      if (!acquired)
        release_reader(counter);
    }
#ifdef ROCJITSU_L2_MUTEX_TSAN
    __tsan_mutex_post_lock(this, flags | (acquired ? 0 : __tsan_mutex_try_lock_failed), 0);
#endif
    return acquired;
  }

  template <typename Hooks> void lock_impl(Hooks hooks) {
#ifdef ROCJITSU_L2_MUTEX_TSAN
    __tsan_mutex_pre_lock(this, 0);
#endif
    writers_.lock();
    writer_pending_.store(1, std::memory_order_seq_cst);
    for (auto &reader : readers_) {
      unsigned active = reader.count.load(std::memory_order_seq_cst);
      while (active != 0) {
        hooks.before_writer_wait(active);
        reader.count.wait(active, std::memory_order_acquire);
        active = reader.count.load(std::memory_order_seq_cst);
      }
    }
#ifdef ROCJITSU_L2_MUTEX_TSAN
    __tsan_mutex_post_lock(this, 0, 0);
#endif
  }

  struct alignas(64) Reader {
    std::atomic<unsigned> count{0};
  };
  static constexpr std::size_t kReaderShards = 128;
  std::array<Reader, kReaderShards> readers_;
  // A word-sized gate permits direct futex waiting on Linux rather than the
  // shared proxy used for byte-sized atomics by some standard libraries.
  alignas(64) std::atomic<uint32_t> writer_pending_{0};
  alignas(64) std::mutex writers_;

  static std::size_t reader_shard() {
    // Thread identity keeps inline acquisition and release consistent across
    // translation units and shared libraries, without shared TLS registration.
    static thread_local const std::size_t shard =
        std::hash<std::thread::id>{}(std::this_thread::get_id()) % kReaderShards;
    return shard;
  }

  void reopen_admission() {
    writer_pending_.store(0, std::memory_order_release);
    writer_pending_.notify_all();
    writers_.unlock();
  }

  void release_reader(std::atomic<unsigned> &counter) {
    // The decrement and gate check join the same sequentially consistent order
    // as writer admission and scanning. If a writer observed this reader, its
    // closed gate is visible here, so the last reader cannot miss its wakeup.
    if (counter.fetch_sub(1, std::memory_order_seq_cst) == 1 &&
        writer_pending_.load(std::memory_order_seq_cst))
      counter.notify_one();
  }
};

} // namespace rocjitsu::amdgpu

#ifdef ROCJITSU_L2_MUTEX_TSAN
#undef ROCJITSU_L2_MUTEX_TSAN
#endif
