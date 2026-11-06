/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

// Bounds on the waits and retries a replay performs on behalf of the archive.

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace hrr {

// Default bounds. Each is one "wait": a single 1 µs sleep (the yield phase
// before it is not counted). A sleep costs about 50-60 µs on Linux, so the
constexpr uint64_t kDefaultMaxSeqWaits      = 5000000;

// A hipEventQuery/hipStreamQuery replay spins until the recorded work is done,
// which is what a synchronize does and a synchronize has no count either: the
constexpr uint64_t kDefaultMaxQueryAttempts = UINT64_MAX;

// Caps on the resources an archive can make replay consume.
constexpr uint64_t kDefaultMaxThreads   = 256;
constexpr uint64_t kDefaultMaxEvents    = 50000000;
constexpr uint64_t kDefaultMaxHostBytes = 16ull << 30;  // 16 GiB
constexpr uint64_t kDefaultMaxWallSecs  = 3600;         // 1 hour
constexpr uint64_t kMaxWallSecs         = 31536000;     // 365 days, the largest accepted value

// Number of yields before a wait starts sleeping.
constexpr int kSpinYields = 1000;

enum class TurnResult {
  Turn,     // next_seq reached the event's sequence number
  Fatal,    // another thread set the fatal flag
  Stalled,  // next_seq made no progress for max_waits waits
};

// The first sequence number that occurs more than once in `ids`, if any. Two
// events with the same number would both see next_seq equal to it and both
inline bool find_duplicate_seq(std::vector<uint64_t> ids, uint64_t* dup) {
  std::sort(ids.begin(), ids.end());
  const auto it = std::adjacent_find(ids.begin(), ids.end());
  if (it == ids.end()) return false;
  if (dup) *dup = *it;
  return true;
}

// The first sequence number missing from `ids` between its smallest and
// largest value, if any. Capture hands out a number only to an event it
inline bool find_seq_gap(std::vector<uint64_t> ids, uint64_t* missing) {
  std::sort(ids.begin(), ids.end());
  for (size_t i = 1; i < ids.size(); ++i) {
    if (ids[i] > ids[i - 1] + 1) {
      if (missing) *missing = ids[i - 1] + 1;
      return true;
    }
  }
  return false;
}

// Wait until `next_seq` equals `seq`.
inline TurnResult wait_for_turn(const std::atomic<uint64_t>& next_seq,
                                uint64_t seq,
                                const std::atomic<bool>& fatal,
                                uint64_t max_waits,
                                uint64_t* observed = nullptr,
                                const std::atomic<int>* busy = nullptr) {
  int spin = 0;
  uint64_t waits = 0;
  uint64_t last = next_seq.load(std::memory_order_acquire);
  for (;;) {
    uint64_t cur = next_seq.load(std::memory_order_acquire);
    if (cur == seq) {
      if (observed) *observed = cur;
      return TurnResult::Turn;
    }
    if (fatal.load(std::memory_order_acquire)) {
      if (observed) *observed = cur;
      return TurnResult::Fatal;
    }
    if (cur > seq) {  // the turn has passed: a duplicate or out-of-order number
      if (observed) *observed = cur;
      return TurnResult::Stalled;
    }
    if (cur != last) {  // someone else made progress: not stalled
      last = cur;
      waits = 0;
    }
    if (++spin < kSpinYields) {
      std::this_thread::yield();
    } else {
      if (busy && busy->load(std::memory_order_acquire) > 0) {
        waits = 0;  // a call is running: the turn will move when it returns
      } else if (++waits > max_waits) {
        if (observed) *observed = cur;
        return TurnResult::Stalled;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
  }
}

// Parse an unsigned integer option value. Requires the first character to be a
// digit: strtoull skips leading whitespace (space, tab, newline) and accepts a
inline bool parse_u64(const char* text, uint64_t* out) {
  if (!text || *text < '0' || *text > '9') return false;
  char* end = nullptr;
  errno = 0;
  const unsigned long long v = std::strtoull(text, &end, 0);
  if (errno == ERANGE || !end || *end != '\0') return false;
  *out = static_cast<uint64_t>(v);
  return true;
}

// Call `attempt` until it returns true or max_attempts calls have been made.
template <typename Attempt>
bool retry_bounded(uint64_t max_attempts, Attempt&& attempt) {
  int spin = 0;
  for (uint64_t n = 0; n < max_attempts; ++n) {
    if (attempt()) return true;
    if (++spin < kSpinYields)
      std::this_thread::yield();
    else
      std::this_thread::sleep_for(std::chrono::microseconds(1));
  }
  return false;
}

// Refuse a recorded thread count above the cap. Returns true if `recorded`
// threads may be created; otherwise false and *message says how many were
inline bool check_thread_cap(size_t recorded, uint64_t cap, std::string* message) {
  if (recorded <= cap) return true;
  if (message)
    *message = "archive records " + std::to_string(recorded) +
               " threads, more than the cap of " + std::to_string(cap) +
               " (--max-threads)";
  return false;
}

// A running total of bytes against a cap. try_charge() is all-or-nothing: it
// adds `n` only if the total stays within the cap, with no overflow for any
class ByteBudget {
 public:
  explicit ByteBudget(uint64_t cap = kDefaultMaxHostBytes) : cap_(cap) {}
  void set_cap(uint64_t cap) { cap_.store(cap, std::memory_order_relaxed); }
  uint64_t cap() const { return cap_.load(std::memory_order_relaxed); }
  uint64_t used() const { return used_.load(std::memory_order_relaxed); }

  bool try_charge(uint64_t n) {
    const uint64_t cap = cap_.load(std::memory_order_relaxed);
    uint64_t cur = used_.load(std::memory_order_relaxed);
    do {
      if (n > cap || cur > cap - n) return false;
    } while (!used_.compare_exchange_weak(cur, cur + n, std::memory_order_relaxed));
    return true;
  }
  void release(uint64_t n) {
    uint64_t cur = used_.load(std::memory_order_relaxed);
    while (!used_.compare_exchange_weak(cur, cur > n ? cur - n : 0,
                                        std::memory_order_relaxed)) {}
  }

 private:
  std::atomic<uint64_t> cap_;
  std::atomic<uint64_t> used_{0};
};

// Holds a charge against a ByteBudget for the life of the object, for buffers
// that are freed again (read-back buffers). ok() is false if the charge was
class ScopedCharge {
 public:
  ScopedCharge(ByteBudget& b, uint64_t n) : b_(b), n_(n), ok_(b.try_charge(n)) {}
  ~ScopedCharge() { if (ok_) b_.release(n_); }
  ScopedCharge(const ScopedCharge&) = delete;
  ScopedCharge& operator=(const ScopedCharge&) = delete;
  bool ok() const { return ok_; }

 private:
  ByteBudget& b_;
  uint64_t n_;
  bool ok_;
};

// Calls `on_expire` once, from its own thread, if it is still alive `limit`
// after construction. Destroying it first cancels the call. A zero `limit`
class WallClockWatchdog {
 public:
  WallClockWatchdog(std::chrono::milliseconds limit, std::function<void()> on_expire)
      : on_expire_(std::move(on_expire)) {
    if (limit.count() == 0) return;
    thread_ = std::thread([this, limit] {
      std::unique_lock<std::mutex> lk(m_);
      if (!cv_.wait_for(lk, limit, [this] { return cancel_; }))
        on_expire_();
    });
  }
  ~WallClockWatchdog() {
    {
      std::lock_guard<std::mutex> lk(m_);
      cancel_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
  }
  WallClockWatchdog(const WallClockWatchdog&) = delete;
  WallClockWatchdog& operator=(const WallClockWatchdog&) = delete;

 private:
  std::function<void()> on_expire_;
  std::mutex m_;
  std::condition_variable cv_;
  bool cancel_ = false;
  std::thread thread_;
};

}  // namespace hrr
