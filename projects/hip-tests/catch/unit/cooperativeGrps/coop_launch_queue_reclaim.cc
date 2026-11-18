/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Regression test for a NULL gpu_queue_ dereference in the cooperative-launch path.
//
// Under dynamic HW queues, an idle stream releases its HW queue back to the shared pool and sets
// gpu_queue_ = nullptr (VirtualGPU::ReleaseHwQueue). The cooperative branch of
// VirtualGPU::submitKernel used to flush the fence (releaseGpuMemoryFence ->
// dispatchBarrierPacket) before reacquiring a queue, dereferencing the null pointer and crashing
// inside libamdhip64.
//
// The reclaim only happens under queue pressure, so the test creates more streams than the default
// HW queue cap, lets their work drain (so the queues go idle and get reclaimed), then issues two
// back-to-back cooperative launches per stream. Without the fix this segfaults, usually on the
// first iteration; with the fix it runs to completion. A segfault takes the whole test process
// down, so this case fails loudly when the fix is absent.

#include <hip/hip_cooperative_groups.h>
#include <hip_test_common.hh>

#include <chrono>
#include <thread>
#include <vector>

static __global__ void bump(int* out) {
  if (threadIdx.x == 0) {
    atomicAdd(out, 1);
  }
}

HIP_TEST_CASE(Unit_hipLaunchCooperativeKernel_QueueReclaim) {
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  if (!props.cooperativeLaunch) {
    HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported);
  }

  // More streams than the default HW queue cap so the pool is under pressure and idle streams get
  // their queues reclaimed.
  constexpr int kNumStreams = 8;
  constexpr int kLaunchesPerStream = 2;
  constexpr int kIterations = 20;

  int* out = nullptr;
  HIP_CHECK(hipMalloc(&out, sizeof(int)));
  HIP_CHECK(hipMemset(out, 0, sizeof(int)));

  std::vector<hipStream_t> streams(kNumStreams);
  for (auto& s : streams) {
    HIP_CHECK(hipStreamCreate(&s));
  }

  void* args[] = {&out};
  int expected = 0;
  for (int it = 0; it < kIterations; ++it) {
    // Give every stream some work so each holds a HW queue.
    for (int i = 0; i < kNumStreams; ++i) {
      hipLaunchKernelGGL(bump, dim3(1), dim3(64), 0, streams[i], out);
      HIP_CHECK(hipGetLastError());
      ++expected;
    }

    // Let the work drain while the host issues no synchronization, so the idle streams release
    // their HW queues back to the pool (gpu_queue_ -> nullptr).
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Synchronize one stream, matching the original reproducer.
    HIP_CHECK(hipStreamSynchronize(streams[0]));

    // Back-to-back cooperative launches with no intervening op. The first launch leaves a pending
    // dispatch + retained external signal, guaranteeing the second launch's fence flush hits the
    // reclaimed (null) queue on the pre-fix runtime.
    for (int i = 0; i < kNumStreams; ++i) {
      for (int n = 0; n < kLaunchesPerStream; ++n) {
        HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<const void*>(&bump), dim3(1),
                                             dim3(64), args, 0, streams[i]));
        ++expected;
      }
    }
    HIP_CHECK(hipDeviceSynchronize());
  }

  int host_out = 0;
  HIP_CHECK(hipMemcpy(&host_out, out, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(host_out == expected);

  for (auto& s : streams) {
    HIP_CHECK(hipStreamDestroy(s));
  }
  HIP_CHECK(hipFree(out));
}
