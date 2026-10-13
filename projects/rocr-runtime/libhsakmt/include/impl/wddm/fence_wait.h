/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <thread>
#include <utility>

namespace wsl {
namespace thunk {

// CPU wait for a queue's monitored fence, used where the platform has no
// atomics and the CPU retires a completion signal itself.
//
// The wait sits on the dispatch critical path: until it returns, the AqlToPm4
// thread translates nothing else and the signal's waiter stays asleep. A fixed
// coarse poll interval is therefore a latency floor under every signalled
// dispatch. The schedule yields for a short window, which covers short kernels,
// then sleeps for an interval that doubles up to max_sleep. Once the interval is
// capped, a long kernel is noticed within max_sleep of retiring and costs one
// wakeup per max_sleep: the wakeup rate is bounded, but the number of wakeups
// grows with the length of the wait.
//
// The defaults are the schedule ComputeQueue ships. Whether the fault timeout
// is enforced is decided per wait, because the runtime can disable it.
struct FenceWaitPolicy {
  std::chrono::nanoseconds yield_for = std::chrono::microseconds(200);
  std::chrono::nanoseconds min_sleep = std::chrono::microseconds(20);
  std::chrono::nanoseconds max_sleep = std::chrono::microseconds(250);
  std::chrono::nanoseconds timeout = std::chrono::seconds(30);
  bool enforce_timeout = true;
};

enum class FenceWaitResult { kReached, kTimedOut };

// Waits until load() >= target. The fence is loaded before every timeout check,
// so a value published during the final sleep still counts as reached. now()
// returns monotonic time as a std::chrono duration; now, yield and sleep are
// parameters so the schedule can be replayed deterministically on the host.
template <typename LoadFn, typename NowFn, typename YieldFn, typename SleepFn>
FenceWaitResult WaitForFence(uint64_t target, const FenceWaitPolicy& policy, LoadFn&& load,
                             NowFn&& now, YieldFn&& yield, SleepFn&& sleep) {
  const auto start = now();
  std::chrono::nanoseconds interval = policy.min_sleep;
  while (load() < target) {
    const auto elapsed = now() - start;
    if (policy.enforce_timeout && elapsed >= policy.timeout) return FenceWaitResult::kTimedOut;
    if (elapsed < policy.yield_for) {
      yield();
    } else {
      sleep(interval);
      interval = std::min(interval * 2, policy.max_sleep);
    }
  }
  return FenceWaitResult::kReached;
}

// The same wait on the steady clock, yielding and sleeping the calling thread.
template <typename LoadFn>
FenceWaitResult WaitForFence(uint64_t target, const FenceWaitPolicy& policy, LoadFn&& load) {
  return WaitForFence(
      target, policy, std::forward<LoadFn>(load),
      [] { return std::chrono::steady_clock::now().time_since_epoch(); },
      [] { std::this_thread::yield(); },
      [](std::chrono::nanoseconds interval) { std::this_thread::sleep_for(interval); });
}

}  // namespace thunk
}  // namespace wsl
