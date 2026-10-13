/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// WaitForFence: the no-platform-atomics completion wait, replayed on a fake
// clock. The fence, the clock, yield and sleep are all simulated, so every
// case runs instantly and deterministically with no GPU and no Windows SDK.

#include "impl/wddm/fence_wait.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using namespace std::chrono_literals;
using wsl::thunk::FenceWaitPolicy;
using wsl::thunk::FenceWaitResult;
using wsl::thunk::WaitForFence;

int failures = 0;

void Failed(const char* expr, int line) {
  std::fprintf(stderr, "fence_wait_test:%d: FAILED: %s\n", line, expr);
  ++failures;
}

#define CHECK(expr) ((expr) ? (void)0 : Failed(#expr, __LINE__))

constexpr FenceWaitPolicy kPolicy{200us, 20us, 250us, 30000ms, true};

// A monitored fence that jumps to `value` once the fake clock reaches
// `retire_at`. A yield costs 1us of fake time; a sleep costs exactly its length.
struct FakeFence {
  uint64_t value = 0;
  uint64_t retire_value = 0;
  std::chrono::nanoseconds retire_at = std::chrono::nanoseconds::max();
  std::chrono::nanoseconds clock{0};
  int yields = 0;
  std::vector<std::chrono::nanoseconds> sleeps;

  FenceWaitResult Wait(uint64_t target, const FenceWaitPolicy& policy = kPolicy) {
    return WaitForFence(
        target, policy,
        [this] {
          if (clock >= retire_at) value = retire_value;
          return value;
        },
        [this] { return clock; },
        [this] {
          ++yields;
          clock += 1us;
        },
        [this](std::chrono::nanoseconds d) {
          sleeps.push_back(d);
          clock += d;
        });
  }
};

void AlreadyReachedReturnsWithoutWaiting() {
  FakeFence f;
  f.value = 7;
  CHECK(f.Wait(7) == FenceWaitResult::kReached);
  CHECK(f.Wait(3) == FenceWaitResult::kReached);
  CHECK(f.yields == 0);
  CHECK(f.sleeps.empty());
}

// A kernel that retires inside the spin window is noticed within one yield and
// never pays a sleep.
void ShortKernelOnlyYields() {
  FakeFence f;
  f.retire_value = 1;
  f.retire_at = 50us;
  CHECK(f.Wait(1) == FenceWaitResult::kReached);
  CHECK(f.sleeps.empty());
  CHECK(f.clock - f.retire_at <= 1us);
}

// Past the spin window the interval starts at min_sleep and doubles up to
// max_sleep, so a long kernel is noticed within max_sleep of retiring - not
// within the 50 ms that used to be the floor.
void LongKernelBacksOffToTheCap() {
  FakeFence f;
  f.retire_value = 1;
  f.retire_at = 10ms;
  CHECK(f.Wait(1) == FenceWaitResult::kReached);
  CHECK(f.sleeps.size() >= 5);
  CHECK(f.sleeps[0] == 20us);
  CHECK(f.sleeps[1] == 40us);
  CHECK(f.sleeps[2] == 80us);
  CHECK(f.sleeps[3] == 160us);
  CHECK(f.sleeps[4] == 250us);
  for (auto d : f.sleeps) CHECK(d <= kPolicy.max_sleep);
  CHECK(f.clock - f.retire_at <= kPolicy.max_sleep);
  CHECK(f.sleeps.size() < 50);  // about 40 wakeups for 10 ms, not 500 at 20 us
}

// Progress on the fence is not enough: only the target value ends the wait.
void WaitsForTheTargetNotForProgress() {
  FakeFence f;
  f.value = 1;
  f.retire_value = 2;
  f.retire_at = 1ms;
  CHECK(f.Wait(3, {200us, 20us, 250us, 5ms, true}) == FenceWaitResult::kTimedOut);
  CHECK(f.value == 2);
}

// The regression the submission-fence target fixes: 1000 merged dispatches
// retire with the fence at 25, so a wait for 1000 can only end as a fault.
void UnreachableTargetTimesOutAsAFault() {
  FakeFence f;
  f.retire_value = 25;
  f.retire_at = 1ms;
  CHECK(f.Wait(1000) == FenceWaitResult::kTimedOut);
  CHECK(f.clock >= kPolicy.timeout);
  CHECK(f.clock < kPolicy.timeout + kPolicy.max_sleep + 1us);

  FakeFence g;
  g.retire_value = 25;
  g.retire_at = 1ms;
  CHECK(g.Wait(25) == FenceWaitResult::kReached);
}

// With the timeout disabled (long-running kernels) the wait outlives the fault
// deadline and still returns as soon as the fence arrives.
void DisabledTimeoutKeepsWaiting() {
  FakeFence f;
  f.retire_value = 1;
  f.retire_at = 2 * kPolicy.timeout;
  FenceWaitPolicy policy = kPolicy;
  policy.enforce_timeout = false;
  CHECK(f.Wait(1, policy) == FenceWaitResult::kReached);
  CHECK(f.clock >= f.retire_at);
  CHECK(f.clock - f.retire_at <= kPolicy.max_sleep);
}

// A fence published during the final sleep before the deadline counts as
// reached rather than as a fault.
void ValueArrivingAtTheDeadlineIsReached() {
  FakeFence f;
  f.retire_value = 1;
  f.retire_at = 5ms;
  CHECK(f.Wait(1, {200us, 20us, 250us, 5ms, true}) == FenceWaitResult::kReached);
}

// Fence values are compared as plain 64-bit ordinals, matching every other
// sync_addr comparison; at one submission per nanosecond they wrap after
// roughly 584 years.
void LargeOrdinals() {
  FakeFence f;
  f.value = UINT64_C(0xfffffffffffffff0);
  CHECK(f.Wait(UINT64_C(0xffffffffffffffef)) == FenceWaitResult::kReached);
  f.retire_value = UINT64_C(0xfffffffffffffff1);
  f.retire_at = 10us;
  CHECK(f.Wait(UINT64_C(0xfffffffffffffff1)) == FenceWaitResult::kReached);
}

}  // namespace

int main() {
  AlreadyReachedReturnsWithoutWaiting();
  ShortKernelOnlyYields();
  LongKernelBacksOffToTheCap();
  WaitsForTheTargetNotForProgress();
  UnreachableTargetTimesOutAsAFault();
  DisabledTimeoutKeepsWaiting();
  ValueArrivingAtTheDeadlineIsReached();
  LargeOrdinals();

  if (failures != 0) {
    std::fprintf(stderr, "fence_wait_test: %d check(s) failed\n", failures);
    return 1;
  }

  std::printf("fence_wait_test: all checks passed\n");
  return 0;
}
