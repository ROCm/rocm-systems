/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>

namespace wsl {
namespace thunk {

// CPU wait for a queue's monitored fence, used where the platform has no
// atomics and the CPU retires a completion signal itself.
//
// The wait sits on the dispatch critical path: until it returns, the AqlToPm4
// thread translates nothing else and the signal's waiter stays asleep. A fixed
// coarse poll interval is therefore a latency floor under every signalled
// dispatch. The schedule yields for a short window, which covers short kernels,
// then sleeps for an interval that doubles up to max_sleep, so a long kernel
// costs a bounded number of wakeups and is noticed within max_sleep.
struct FenceWaitPolicy {
  std::chrono::nanoseconds spin_for;
  std::chrono::nanoseconds min_sleep;
  std::chrono::nanoseconds max_sleep;
  std::chrono::nanoseconds timeout;
  bool enforce_timeout;
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
    if (elapsed < policy.spin_for) {
      yield();
    } else {
      sleep(interval);
      interval = std::min(interval * 2, policy.max_sleep);
    }
  }
  return FenceWaitResult::kReached;
}

}  // namespace thunk
}  // namespace wsl
