/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
Testcase Scenarios :
Unit_hipStreamSynchronize_ConcurrentSubmitFromOtherThread - While one thread is blocked in
hipStreamSynchronize on a stream, another thread must still be able to call hipStreamQuery and
hipStreamWaitEvent on that same stream. The kernel being waited on only exits once the second
thread has made those calls, so a runtime that holds a stream-wide lock across the host wait in
hipStreamSynchronize turns this into a deadlock (ROCM-32517). A device-side watchdog bounds the
failure so a regression fails the test instead of hanging the GPU. The test is also run with
several additional threads synchronizing the same stream, which exercises the case where a thread
finds the synchronize of another thread still in flight.
*/

#include <hip_test_common.hh>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace {

// Timer with a known rate: wall_clock64() runs at hipDeviceAttributeWallClockRate on AMD, while
// clock64() follows the varying core clock there. NVIDIA has no wall_clock64().
__device__ inline long long readTimer() {
#if HT_NVIDIA
  return clock64();
#else
  return wall_clock64();
#endif
}

// Spins until the host sets release[0], or until the watchdog expires. On expiry it records the
// fact in timedOut[0] and exits, so a runtime bug cannot hang the GPU forever.
__global__ void spinUntilReleased(volatile unsigned int* release, volatile unsigned int* timedOut,
                                  long long watchdogTicks) {
  const long long start = readTimer();
  while (*release == 0) {
    if (readTimer() - start > watchdogTicks) {
      *timedOut = 1;
      return;
    }
  }
}

// Releases the spinning kernel on scope exit, so every path out of the worker thread (including an
// early return from HIP_CHECK_THREAD) lets the kernel finish.
struct ReleaseOnExit {
  volatile unsigned int* release;
  ~ReleaseOnExit() { *release = 1; }
};

using Clock = std::chrono::steady_clock;

long long elapsedUs(Clock::time_point since) {
  return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - since).count();
}

}  // namespace

HIP_TEST_CASE(Unit_hipStreamSynchronize_ConcurrentSubmitFromOtherThread) {
  constexpr size_t kBytes = 10 * sizeof(float);
  constexpr int kWatchdogSeconds = 10;
  // The calls under test only enqueue work or read a snapshot. A second is orders of magnitude above
  // their normal cost, and orders of magnitude below the watchdog.
  constexpr long long kMaxCallUs = 1'000'000;

  const bool useNullStream = GENERATE(true, false);
  // Threads that call hipStreamSynchronize on the same stream in addition to the main thread
  const int extraSyncers = GENERATE(0, 3);
  INFO("Waiting stream: " << (useNullStream ? "null stream" : "created stream")
                          << ", extra synchronizing threads: " << extraSyncers);

  // Earlier tests may leave another device selected, while the threads below start on device 0
  HIP_CHECK(hipSetDevice(0));

  int ticksPerMs = 0;
#if HT_NVIDIA
  HIP_CHECK(hipDeviceGetAttribute(&ticksPerMs, hipDeviceAttributeClockRate, 0));
#else
  HIP_CHECK(hipDeviceGetAttribute(&ticksPerMs, hipDeviceAttributeWallClockRate, 0));
#endif
  REQUIRE(ticksPerMs > 0);
  const long long watchdogTicks = static_cast<long long>(ticksPerMs) * 1000 * kWatchdogSeconds;

  hipStream_t stream = nullptr;
  if (!useNullStream) {
    HIP_CHECK(hipStreamCreate(&stream));
  }
  hipStream_t side{nullptr};
  HIP_CHECK(hipStreamCreateWithFlags(&side, hipStreamNonBlocking));

  // [0] = release the kernel, [1] = the watchdog fired
  unsigned int* flagsHost = nullptr;
  unsigned int* flagsDev = nullptr;
  HIP_CHECK(hipHostMalloc(&flagsHost, 2 * sizeof(unsigned int), hipHostMallocCoherent));
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&flagsDev), flagsHost, 0));
  flagsHost[0] = 0;
  flagsHost[1] = 0;

  float* pinned = nullptr;
  float* devBuf = nullptr;
  HIP_CHECK(hipHostMalloc(&pinned, kBytes));
  HIP_CHECK(hipMalloc(&devBuf, kBytes));

  hipEvent_t sideReady{nullptr};
  HIP_CHECK(hipEventCreateWithFlags(&sideReady, hipEventDisableTiming));
  // Created here rather than in the waiter thread, so an early return there doesn't leak it
  hipEvent_t copied{nullptr};
  HIP_CHECK(hipEventCreateWithFlags(&copied, hipEventDisableTiming));

  spinUntilReleased<<<1, 1, 0, stream>>>(flagsDev, flagsDev + 1, watchdogTicks);
  HIP_CHECK(hipGetLastError());

  // Cross-stream dependency, as ProcessGroupGloo does: the side stream waits on an event recorded
  // behind the kernel. This leaves a small, not yet submitted batch on `stream`.
  HIP_CHECK(hipEventRecord(sideReady, stream));
  HIP_CHECK(hipStreamWaitEvent(side, sideReady, 0));

  std::atomic<bool> aboutToSync{false};
  std::atomic<bool> syncReturned{false};
  std::atomic<long long> maxQueryUs{-1};
  std::atomic<long long> waitUs{-1};

  // Two independent threads, so that each call is measured while the main thread is still blocked
  // in hipStreamSynchronize, rather than one call queuing up behind the other.
  //
  // The querier keeps calling hipStreamQuery until the synchronize returns, so a call lands inside
  // the synchronize no matter how the threads get scheduled.
  std::thread querier([&] {
    while (!aboutToSync) {
      std::this_thread::yield();
    }
    while (!syncReturned) {
      const auto t0 = Clock::now();
      // The kernel keeps running until the waiter releases it, so hipErrorNotReady is expected.
      const hipError_t queryStatus = hipStreamQuery(stream);
      const long long us = elapsedUs(t0);
      if (us > maxQueryUs) {
        maxQueryUs = us;
      }
      REQUIRE_THREAD(queryStatus == hipErrorNotReady || queryStatus == hipSuccess);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });

  std::thread waiter([&] {
    ReleaseOnExit releaseKernel{flagsHost};

    while (!aboutToSync) {
      std::this_thread::yield();
    }
    // The main thread sets the flag right before it calls hipStreamSynchronize, so this leaves it
    // ample time to be blocked inside the call. This is a best-effort ordering: on a heavily
    // loaded host the calls below may be enqueued first. The test then passes without covering the
    // contention, but it can't fail spuriously.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    HIP_CHECK_THREAD(hipMemcpyAsync(devBuf, pinned, kBytes, hipMemcpyHostToDevice, side));
    HIP_CHECK_THREAD(hipEventRecord(copied, side));

    const auto t0 = Clock::now();
    HIP_CHECK_THREAD(hipStreamWaitEvent(stream, copied, 0));
    waitUs = elapsedUs(t0);
  });

  std::atomic<int> extraSyncFailures{0};
  std::vector<std::thread> extraSyncThreads;
  for (int i = 0; i < extraSyncers; ++i) {
    extraSyncThreads.emplace_back([&] {
      while (!aboutToSync) {
        std::this_thread::yield();
      }
      if (hipStreamSynchronize(stream) != hipSuccess) {
        ++extraSyncFailures;
      }
    });
  }

  aboutToSync = true;
  const hipError_t syncStatus = hipStreamSynchronize(stream);
  syncReturned = true;
  querier.join();
  waiter.join();
  for (auto& t : extraSyncThreads) {
    t.join();
  }

  // Collect the results and release everything before asserting, so that a failing assertion does
  // not leak the streams and buffers for the rest of the process.
  const hipError_t deviceSyncStatus = hipDeviceSynchronize();
  const long long queryUsResult = maxQueryUs.load();
  const long long waitUsResult = waitUs.load();
  const bool watchdogFired = flagsHost[1] != 0;

  HIP_CHECK(hipEventDestroy(copied));
  HIP_CHECK(hipEventDestroy(sideReady));
  HIP_CHECK(hipFree(devBuf));
  HIP_CHECK(hipHostFree(pinned));
  HIP_CHECK(hipHostFree(flagsHost));
  HIP_CHECK(hipStreamDestroy(side));
  if (!useNullStream) {
    HIP_CHECK(hipStreamDestroy(stream));
  }

  HIP_CHECK_THREAD_FINALIZE();
  INFO("Slowest hipStreamQuery took " << queryUsResult << " us, hipStreamWaitEvent took "
                                      << waitUsResult
                                      << " us while another thread was in hipStreamSynchronize");
  HIP_CHECK(syncStatus);
  REQUIRE(extraSyncFailures.load() == 0);
  HIP_CHECK(deviceSyncStatus);
  REQUIRE(queryUsResult >= 0);
  REQUIRE(waitUsResult >= 0);
  REQUIRE(queryUsResult < kMaxCallUs);
  REQUIRE(waitUsResult < kMaxCallUs);
  // The kernel only exits once the waiter released it after its calls returned.
  REQUIRE_FALSE(watchdogFired);
}
