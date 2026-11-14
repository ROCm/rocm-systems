/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// LaunchCompletionEvent: ncclCollConfig_t::launchCompletionEvent (NCCL 2.32). HIP has no launch-completion
// launch attribute, so RCCL records the caller's event on the collective's stream right before the kernel
// launch (ncclLaunchOneRank, ncclLaunchKernel), and a group takes at most one such event per communicator.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

#include "common/ProcessIsolatedTestRunner.hpp"
#include "common/ResourceGuards.hpp"

namespace RcclUnitTesting {
namespace {

// launchCompletionEvent is appended right after userProfilerTag. It is written by offset so this file also
// builds against the 2.31 header, whose library has no such member and never reads it.
constexpr size_t kEventOffset = offsetof(ncclCollConfig_t, userProfilerTag) + sizeof(uint64_t);
static_assert(sizeof(ncclCollConfig_t) == kEventOffset ||
                sizeof(ncclCollConfig_t) == kEventOffset + sizeof(hipEvent_t),
              "launchCompletionEvent must directly follow userProfilerTag");
constexpr size_t kCount = 4096;

// NCCL_COLLCONFIG_INITIALIZER with `event` in the launchCompletionEvent slot.
struct EventConfig {
  alignas(ncclCollConfig_t) unsigned char bytes[kEventOffset + sizeof(hipEvent_t)];
  explicit EventConfig(hipEvent_t event) {
    ncclCollConfig_t init = NCCL_COLLCONFIG_INITIALIZER;
    std::memcpy(bytes, &init, sizeof(init));
    std::memcpy(bytes + kEventOffset, &event, sizeof(event));
  }
  const ncclCollConfig_t* get() const { return reinterpret_cast<const ncclCollConfig_t*>(bytes); }
};

struct Rank {
  RCCLTestGuards::NcclCommAutoGuard comm{nullptr};
  RCCLTestGuards::HipStreamAutoGuard stream{nullptr};
  RCCLTestGuards::DeviceBufferAutoGuard buf{nullptr};
  RCCLTestGuards::HipEventAutoGuard event{nullptr};
  ncclRedOp_t op = ncclSum;
};

// One communicator per GPU, each with a stream, a buffer, a timing-disabled event and a PreMulSum op.
// PreMulSum makes even one rank launch a kernel: ncclSum there is a device copy, and copies record no event.
void setUpRanks(std::vector<Rank>& ranks, int nRanks) {
  std::vector<ncclComm_t> comms(nRanks, nullptr);
  std::vector<int> devs(nRanks);
  std::iota(devs.begin(), devs.end(), 0);
  ASSERT_EQ(ncclCommInitAll(comms.data(), nRanks, devs.data()), ncclSuccess);
  ranks.resize(nRanks);
  float scalar = 1.0f;
  for (int r = 0; r < nRanks; ++r) {
    Rank& k = ranks[r];
    k.comm.set(comms[r]);
    ASSERT_EQ(hipSetDevice(r), hipSuccess);
    ASSERT_EQ(hipStreamCreateWithFlags(k.stream.ptr(), hipStreamNonBlocking), hipSuccess);
    ASSERT_EQ(hipEventCreateWithFlags(k.event.ptr(), hipEventDisableTiming), hipSuccess);
    ASSERT_EQ(hipMalloc(k.buf.ptr(), kCount * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMemset(k.buf.get(), 0, kCount * sizeof(float)), hipSuccess);
    ASSERT_EQ(ncclRedOpCreatePreMulSum(&k.op, &scalar, ncclFloat, ncclScalarHostImmediate, k.comm.get()),
              ncclSuccess);
  }
}

// `calls` rounds of one in-place allreduce per rank in one group, each config carrying the rank's event or
// none. Returns the first failing call's result, else what ncclGroupEnd reports.
ncclResult_t groupedAllReduce(std::vector<Rank>& ranks, bool withEvent, int calls) {
  ncclResult_t res = ncclGroupStart();
  for (int i = 0; i < calls; ++i) {
    for (Rank& k : ranks) {
      EventConfig config(withEvent ? k.event.get() : nullptr);
      if (res == ncclSuccess)
        res = ncclAllReduceConfig(k.buf.get(), k.buf.get(), kCount, ncclFloat, k.op, k.comm.get(),
                                  k.stream.get(), config.get());
    }
  }
  ncclResult_t end = ncclGroupEnd();
  return res != ncclSuccess ? res : end;
}

// Host function that holds its stream until *open is set. Bounded, so a lost release cannot wedge HIP's
// callback thread for good.
void holdUntilOpen(void* open) {
  for (int ms = 0; ms < 60000 && !static_cast<std::atomic<bool>*>(open)->load(); ++ms)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

// Runs the allreduce while every stream is held. With eventInConfig NCCL must record each rank's event on
// that rank's stream; without it the test records the event there itself, which is what the control checks.
void runHeldAllReduce(int nRanks, bool eventInConfig) {
  std::vector<Rank> ranks;
  ASSERT_NO_FATAL_FAILURE(setUpRanks(ranks, nRanks));
  // Warm-up, so connection setup and kernel loading do not happen while the streams are held. Two configured
  // collectives without an event in one group are valid on any version.
  ASSERT_EQ(groupedAllReduce(ranks, /*withEvent=*/false, /*calls=*/2), ncclSuccess);
  for (Rank& k : ranks) {
    ASSERT_EQ(hipStreamSynchronize(k.stream.get()), hipSuccess);
    ASSERT_EQ(hipEventQuery(k.event.get()), hipSuccess) << "an event never recorded reports complete";
  }
  {
    // One flag opens every stream at once: HIP may run all host functions on a single thread.
    std::atomic<bool> open{false};
    auto release = RCCLTestGuards::makeScopeGuard([&]() {
      open = true;
      for (Rank& k : ranks) (void)hipStreamSynchronize(k.stream.get());
    });
    for (Rank& k : ranks) ASSERT_EQ(hipLaunchHostFunc(k.stream.get(), holdUntilOpen, &open), hipSuccess);
    ASSERT_EQ(groupedAllReduce(ranks, eventInConfig, /*calls=*/1), ncclSuccess);
    for (Rank& k : ranks) {
      if (!eventInConfig) {
        ASSERT_EQ(hipEventRecord(k.event.get(), k.stream.get()), hipSuccess);
      }
      EXPECT_EQ(hipEventQuery(k.event.get()), hipErrorNotReady)
        << nRanks << " rank(s): the event was not recorded behind the work queued on the collective's stream";
    }
  }
  for (Rank& k : ranks) EXPECT_EQ(hipEventQuery(k.event.get()), hipSuccess) << nRanks << " rank(s)";
}

int gpuCount() {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess ? n : 0;
}

} // namespace

// One rank takes ncclLaunchOneRank; two ranks, when present, take the planner and ncclLaunchKernel.
TEST(LaunchCompletionEvent, RecordedOnCollectiveStream) {
  RUN_ISOLATED_TEST("LaunchCompletionEvent.RecordedOnCollectiveStream", []() {
    if (gpuCount() < 1) GTEST_SKIP() << "requires at least one GPU";
    for (int n = 1; n <= std::min(gpuCount(), 2); ++n) ASSERT_NO_FATAL_FAILURE(runHeldAllReduce(n, true));
  });
}

TEST(LaunchCompletionEvent, SecondEventInGroupIsInvalidUsage) {
  RUN_ISOLATED_TEST("LaunchCompletionEvent.SecondEventInGroupIsInvalidUsage", []() {
    if (gpuCount() < 1) GTEST_SKIP() << "requires at least one GPU";
    std::vector<Rank> ranks;
    ASSERT_NO_FATAL_FAILURE(setUpRanks(ranks, 1));
    EXPECT_EQ(groupedAllReduce(ranks, /*withEvent=*/true, /*calls=*/2), ncclInvalidUsage)
      << "a group takes at most one launch-completion event per communicator";
    // The rejection must leave the communicator usable.
    EXPECT_EQ(groupedAllReduce(ranks, /*withEvent=*/true, /*calls=*/1), ncclSuccess);
    EXPECT_EQ(hipStreamSynchronize(ranks[0].stream.get()), hipSuccess);
  });
}

// The same held-stream flow with no event in the config: an event the test records itself after the
// collective must stay pending until the streams are released.
TEST(LaunchCompletionEvent, ControlTestRecordedEvent) {
  RUN_ISOLATED_TEST("LaunchCompletionEvent.ControlTestRecordedEvent", []() {
    if (gpuCount() < 1) GTEST_SKIP() << "requires at least one GPU";
    for (int n = 1; n <= std::min(gpuCount(), 2); ++n) ASSERT_NO_FATAL_FAILURE(runHeldAllReduce(n, false));
  });
}

} // namespace RcclUnitTesting
