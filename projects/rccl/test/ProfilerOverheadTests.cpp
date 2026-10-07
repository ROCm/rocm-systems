/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <unistd.h>
#include <atomic>

#include "comm.h"
#include "common/ProcessIsolatedTestRunner.hpp"

namespace {
constexpr int kRanks = 2;
constexpr size_t kBytes = 1 << 20;
std::atomic<int> kernelChStops{0}, kernelPhases{0};
constexpr auto noop = [](void*) { return ncclSuccess; };

// Two ranks in this process: issue(comm, buf, stream) per rank in one group, then check(comms) before teardown.
template <typename Issue, typename Check> void runTwoRanks(Issue issue, Check check) {
  ncclComm_t comms[kRanks];
  hipStream_t streams[kRanks];
  char* bufs[kRanks];
  ASSERT_EQ(ncclCommInitAll(comms, kRanks, nullptr), ncclSuccess);
  for (int r = 0; r < kRanks; ++r) {
    ASSERT_EQ(hipSetDevice(r), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&streams[r]), hipSuccess);
    ASSERT_EQ(hipMalloc(&bufs[r], 4 * kBytes), hipSuccess);
  }
  ASSERT_EQ(ncclGroupStart(), ncclSuccess);
  for (int r = 0; r < kRanks; ++r) ASSERT_EQ(issue(comms[r], bufs[r], streams[r]), ncclSuccess);
  ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
  for (int r = 0; r < kRanks; ++r) ASSERT_EQ(hipStreamSynchronize(streams[r]), hipSuccess);
  check(comms);
  for (int r = 0; r < kRanks; ++r) ASSERT_EQ(ncclCommDestroy(comms[r]), ncclSuccess);
  for (int r = 0; r < kRanks; ++r) ASSERT_EQ(hipStreamDestroy(streams[r]), hipSuccess);
  for (int r = 0; r < kRanks; ++r) ASSERT_EQ(hipFree(bufs[r]), hipSuccess);
}
} // namespace

// Loaded in-process via NCCL_PROFILER_PLUGIN=STATIC_PLUGIN and exported in test/CMakeLists.txt. Not
// const: HIP makes a const global an implicit __constant__, which then fails to link to host code.
extern "C" __attribute__((visibility("default"))) ncclProfiler_v7_t ncclProfiler_v7 = {
    "KernelPhaseCount",
    [](void** context, uint64_t, int* eActivationMask, const char*, int, int, int, ncclDebugLogger_t) {
      *context = &kernelChStops;
      *eActivationMask = ncclProfileKernelCh | ncclProfileKernelPhase;
      return ncclSuccess;
    },
    [](void* context, void** eHandle, ncclProfilerEventDescr_v7_t* eDescr) {
      *eHandle = context;
      if (eDescr->type == ncclProfileKernelPhase) ++kernelPhases;
      return ncclSuccess;
    }, noop,
    [](void*, ncclProfilerEventState_v7_t eState, ncclProfilerEventStateArgs_v7_t*) {
      if (eState == ncclProfilerKernelChStop) ++kernelChStops;
      return ncclSuccess;
    }, noop};

namespace RcclUnitTesting {
// KernelPhase times symmetric kernels' barriers; regular kernels used to stamp it with KernelCh on.
TEST(ProfilerOverhead, RegularKernelsEmitNoKernelPhase) {
  if (int n = 0; hipGetDeviceCount(&n) != hipSuccess || n < kRanks) GTEST_SKIP() << "requires 2 GPUs";
  using Config = ProcessIsolatedTestRunner::TestConfig;
  RUN_ISOLATED_TESTS(Config("ProfilerOverhead.RegularKernelsEmitNoKernelPhase", []() {
    // Wait before teardown: ncclCommDestroy drops pending events. Each channel's phase events precede its KernelCh
    // stop, but each comm drains on its own profiler thread, so wait until the stops have settled, not just begun.
    ASSERT_NO_FATAL_FAILURE(runTwoRanks([](ncclComm_t comm, char* buf, hipStream_t stream) {
      return ncclAllReduce(buf, buf, 1024, ncclFloat, ncclSum, comm, stream);
    }, [](ncclComm_t*) {
      for (int i = 0, last = -1, quiet = 0; i < 1000 && quiet < 20; ++i) {
        usleep(10000);
        const int now = kernelChStops;
        quiet = (now >= kRanks && now == last) ? quiet + 1 : 0;
        last = now;
      }
    }));
    ASSERT_GE(kernelChStops.load(), kRanks) << "KernelCh events missing, so KernelPhase events could be too";
    EXPECT_EQ(kernelPhases.load(), 0) << "KernelPhase events reported for a non-symmetric kernel";
  }).withEnvironment({{"NCCL_PROFILER_PLUGIN", "STATIC_PLUGIN"}}).withNumGpus(kRanks));
}

// Bcast work has no profiling bit; the kernel used to read bit 15 of the root's sendbuff instead.
TEST(ProfilerOverhead, NoPluginBroadcastPublishesNoKernelCh) {
#ifdef ENABLE_WARP_SPEED
  GTEST_SKIP() << "WarpSpeed layout: the misread bit is sendbuff bit 47, which no user pointer sets";
#endif
  if (int n = 0; hipGetDeviceCount(&n) != hipSuccess || n < kRanks) GTEST_SKIP() << "requires 2 GPUs";
  using Config = ProcessIsolatedTestRunner::TestConfig;
  RUN_ISOLATED_TESTS(Config("ProfilerOverhead.NoPluginBroadcastPublishesNoKernelCh", []() {
    // Two roots in one group, so the broadcasts are batched as Bcast work (NCCL_ALLGATHERV_ENABLE).
    ASSERT_NO_FATAL_FAILURE(runTwoRanks([](ncclComm_t comm, char* buf, hipStream_t stream) {
      char* send = buf + (~reinterpret_cast<uintptr_t>(buf) & 0x8000);
      ncclResult_t res = ncclBroadcast(send, buf + 2 * kBytes, kBytes, ncclInt8, 0, comm, stream);
      return res != ncclSuccess ? res : ncclBroadcast(send, buf + 3 * kBytes, kBytes, ncclInt8, 1, comm, stream);
    }, [](ncclComm_t* comms) {
      uint64_t published = 0;
      for (int r = 0; r < kRanks; ++r) {
        for (int c = 0; c < MAXCHANNELS; ++c) {
          for (int s = 0; s < MAX_PROFILER_EVENTS_PER_CHANNEL; ++s)
            published |= comms[r]->profiler.workStarted[c].data[s].counter;
        }
      }
      EXPECT_EQ(published, 0u) << "KernelCh work-start counters published with no profiler plugin loaded";
    }));
  }).withEnvironment({{"NCCL_PROFILER_PLUGIN", "none"}, {"NCCL_ALLGATHERV_ENABLE", "1"}}).withNumGpus(kRanks));
}
} // namespace RcclUnitTesting
