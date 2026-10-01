/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ProfilerApiEventRank: group/collective/P2P API and kernel-launch event descriptors must carry
// the rank of the communicator they were issued on, not rank 0 (NVIDIA/nccl#2300).

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <atomic>

#include "common/ProcessIsolatedTestRunner.hpp"
#include "nccl_common.h"
#include "nccl_profiler.h"

namespace {

constexpr uint64_t kApiEvents =
    ncclProfileGroupApi | ncclProfileCollApi | ncclProfileP2pApi | ncclProfileKernelLaunch;

// Event types seen from a non-zero rank's context, and those reported with a rank other than
// the one that context was created for.
std::atomic<uint64_t> seenOnNonZeroRank{0}, wrongRank{0};

ncclResult_t rankStartEvent(void* context, void** eHandle, ncclProfilerEventDescr_v7_t* eDescr) {
  const int rank = *static_cast<int*>(context);
  const uint64_t type = eDescr->type & kApiEvents;
  *eHandle = context;
  if (rank != 0) seenOnNonZeroRank |= type;
  if (eDescr->rank != rank) wrongRank |= type;
  return ncclSuccess;
}

} // namespace

// Loaded in-process via NCCL_PROFILER_PLUGIN=STATIC_PLUGIN; exported in test/CMakeLists.txt.
// Not const: HIP makes a const global an implicit __constant__, and the device pass then fails
// to link against these host-only callbacks.
extern "C" __attribute__((visibility("default"))) ncclProfiler_v7_t ncclProfiler_v7 = {
    "RankCheck",
    [](void** context, uint64_t, int* eActivationMask, const char*, int, int, int rank, ncclDebugLogger_t) {
      *context = new int(rank);
      *eActivationMask = static_cast<int>(kApiEvents);
      return ncclSuccess;
    },
    rankStartEvent, [](void*) { return ncclSuccess; },
    [](void*, ncclProfilerEventState_v7_t, ncclProfilerEventStateArgs_v7_t*) { return ncclSuccess; },
    [](void* context) { delete static_cast<int*>(context); return ncclSuccess; }};

namespace RcclUnitTesting {

// Isolated because the profiler plugin is loaded once per process, at the first communicator.
TEST(ProfilerApiEventRank, ApiEventsCarryCommRank) {
  if (int n = 0; hipGetDeviceCount(&n) != hipSuccess || n < 2) GTEST_SKIP() << "requires at least 2 GPUs";
  RUN_ISOLATED_TEST_WITH_ENV("ProfilerApiEventRank.ApiEventsCarryCommRank", []() {
    constexpr int kRanks = 2;
    constexpr size_t kCount = 1024;
    ncclComm_t comms[kRanks];
    hipStream_t streams[kRanks];
    float* bufs[kRanks];
    ASSERT_EQ(ncclCommInitAll(comms, kRanks, nullptr), ncclSuccess);
    for (int r = 0; r < kRanks; ++r) {
      ASSERT_EQ(hipSetDevice(r), hipSuccess);
      ASSERT_EQ(hipStreamCreate(&streams[r]), hipSuccess);
      ASSERT_EQ(hipMalloc(&bufs[r], 3 * kCount * sizeof(float)), hipSuccess);
    }
    // Highest rank first, so the outermost group API event is opened on rank 1's communicator.
    ASSERT_EQ(ncclGroupStart(), ncclSuccess);
    for (int r = kRanks - 1; r >= 0; --r) {
      const int peer = kRanks - 1 - r;
      ASSERT_EQ(ncclAllReduce(bufs[r], bufs[r], kCount, ncclFloat, ncclSum, comms[r], streams[r]), ncclSuccess);
      ASSERT_EQ(ncclSend(bufs[r] + kCount, kCount, ncclFloat, peer, comms[r], streams[r]), ncclSuccess);
      ASSERT_EQ(ncclRecv(bufs[r] + 2 * kCount, kCount, ncclFloat, peer, comms[r], streams[r]), ncclSuccess);
    }
    ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
    for (int r = 0; r < kRanks; ++r) {
      ASSERT_EQ(hipStreamSynchronize(streams[r]), hipSuccess);
      ASSERT_EQ(ncclCommDestroy(comms[r]), ncclSuccess);
      ASSERT_EQ(hipStreamDestroy(streams[r]), hipSuccess);
      ASSERT_EQ(hipFree(bufs[r]), hipSuccess);
    }

    // Without this the rank check below would also pass for events that never arrived.
    ASSERT_EQ(seenOnNonZeroRank.load(), kApiEvents) << "API event types seen from rank 1 (bitmask)";
    EXPECT_EQ(wrongRank.load(), 0u) << "API event types reported with the wrong rank (bitmask)";
  }, {{"NCCL_PROFILER_PLUGIN", "STATIC_PLUGIN"}});
}

} // namespace RcclUnitTesting
