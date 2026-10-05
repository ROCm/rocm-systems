/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

// Bounds on the waits and retries a replay performs on behalf of the archive.
//
// An archive is input from another principal. Anything the replay waits for or
// repeats because the archive says so needs an upper bound, or a malformed
// archive turns into a hang. This header is header-only and has no HIP
// dependency so the bounds can be unit tested without a GPU.

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

namespace hrr {

// Default bounds. Each is one "wait": a single 1 µs sleep (the yield phase
// before it is not counted). A sleep costs about 50-60 µs on Linux, so the
// defaults allow a few minutes of waiting with no progress, far more than a
// recorded run needs, and still end a malformed archive.
constexpr uint64_t kDefaultMaxSeqWaits      = 5000000;
constexpr uint64_t kDefaultMaxQueryAttempts = 5000000;

// Caps on the resources an archive can make replay consume.
// Each is set with a --max-* option; 0 turns the wall-clock watchdog off and
// makes every other cap refuse everything, so there is no "unlimited" value to
// pass by accident. The defaults hold on a developer machine: a recorded run
// has a handful of threads and a bounded working set, and a replay that needs
// more says so with the option that raises it.
//
//   threads     one operating-system thread is created per recorded thread id
//               in --multi-thread mode.
//   events      events held in memory after the archive is loaded.
//   host bytes  host memory the replay holds on behalf of the archive: blob
//               and code-object caches, pinned and registered host buffers,
//               graph landing buffers and D2H read-back buffers.
//   wall time   seconds from the start of the replay.
constexpr uint64_t kDefaultMaxThreads   = 256;
constexpr uint64_t kDefaultMaxEvents    = 50000000;
constexpr uint64_t kDefaultMaxHostBytes = 16ull << 30;  // 16 GiB
constexpr uint64_t kDefaultMaxWallSecs  = 3600;         // 1 hour

// Number of yields before a wait starts sleeping.
constexpr int kSpinYields = 1000;

enum class TurnResult {
  Turn,     // next_seq reached the event's sequence number
  Fatal,    // another thread set the fatal flag
  Stalled,  // next_seq made no progress for max_waits waits
};

// Wait until `next_seq` equals `seq`.
//
// A recorded stream hands the turn from one event to the next by writing
// seq + 1 into next_seq. A gap in the recorded sequence numbers (the turn is
// never handed to `seq`) leaves this wait with nothing to wait for; a
// duplicate (next_seq has already passed `seq`) can never become current, and
// is reported at once. For a gap the wait counts sleeps while next_seq does
// not move, restarts the count whenever it does, and returns Stalled after
// max_waits of them. *observed receives the value next_seq had when the wait
// ended, so the caller can name it in a diagnostic.
//
// An ordered event keeps next_seq where it is for as long as its HIP call
// runs, which can be minutes for a recorded synchronize or a long kernel.
// `busy`, when given, counts the threads that hold a turn and are inside such
// a call; no wait is counted while it is non-zero, so a slow legitimate call
// is never mistaken for a missing event.
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

// Parse an unsigned integer option value. Rejects an empty string, a leading
// sign or space (strtoull would accept "-1" as the largest value, silently
// turning a cap off), trailing characters and out-of-range values. Accepts
// decimal, and 0x hex or 0 octal as strtoull does with base 0.
inline bool parse_u64(const char* text, uint64_t* out) {
  if (!text || !*text || *text == '-' || *text == '+' || *text == ' ') return false;
  char* end = nullptr;
  errno = 0;
  const unsigned long long v = std::strtoull(text, &end, 0);
  if (errno == ERANGE || !end || *end != '\0') return false;
  *out = static_cast<uint64_t>(v);
  return true;
}

// Call `attempt` until it returns true or max_attempts calls have been made.
// Returns true if `attempt` succeeded, false if the bound was reached. A
// max_attempts of 0 makes no call.
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
// recorded and what the cap is. Called before any replay thread starts, so the
// process never exceeds the cap.
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
// `n`. Thread safe.
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
// refused; nothing is held then.
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
// disables it. The callback must end the process (the replay may be blocked in
// a HIP call that no flag can interrupt), for example by printing and calling
// _exit.
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
