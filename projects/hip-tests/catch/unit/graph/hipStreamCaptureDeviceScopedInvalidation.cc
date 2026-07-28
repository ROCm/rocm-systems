/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipStreamBeginCapture hipStreamBeginCapture
 * @{
 * @ingroup GraphTest
 *
 * Device-scoped invalidation tests for stream capture.
 *
 * These tests verify that HIP's stream capture invalidation follows
 * device-scoped behavior.
 *
 * Key principle: Invalidation is DEVICE-SCOPED, not thread-scoped or mode-scoped.
 *   - Sync APIs invalidate ALL captures on the SAME device (any thread, any mode)
 *   - Sync APIs do NOT invalidate captures on DIFFERENT devices
 *   - Capture mode (Global/ThreadLocal/Relaxed) is irrelevant for cross-thread invalidation
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#include <future>
#include <thread>

namespace {

static __global__ void addKernel(int* dst, const int* src, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

constexpr int kN = 256;

}  // anonymous namespace

// ================================================================================================
/**
 * Test: ThreadLocal capture on GPU 0, sync API from another thread on SAME GPU 0.
 *
 * Thread A: ThreadLocal capture on GPU 0
 * Thread B: hipMemcpy on GPU 0 (same device)
 *
 * Expected: Thread A's capture is INVALIDATED despite being ThreadLocal mode.
 * Device-scoped invalidation means same-device sync APIs invalidate ALL captures on that device.
 */
TEST_CASE("ThreadLocal_SameDevice_SyncInvalidates", "[graph][capture][multithreaded]") {
  hipStream_t streamA = nullptr;
  int *dA = nullptr, *dB = nullptr, *hB = nullptr;
  hipGraph_t graphA = nullptr;
  hipError_t captureErrA = hipSuccess;
  hipError_t endCaptureErrA = hipSuccess;
  hipError_t syncErrB = hipSuccess;

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipStreamCreate(&streamA));
  HIP_CHECK(hipMalloc(&dA, kN * sizeof(int)));
  HIP_CHECK(hipMalloc(&dB, kN * sizeof(int)));
  hB = (int*)malloc(kN * sizeof(int));
  memset(hB, 0, kN * sizeof(int));

  std::promise<void> captureStarted, syncDone;
  auto captureReady = captureStarted.get_future();
  auto syncReady = syncDone.get_future();

  std::thread threadA([&]() {
    HIP_CHECK_THREAD(hipSetDevice(0));
    captureErrA = hipStreamBeginCapture(streamA, hipStreamCaptureModeThreadLocal);
    captureStarted.set_value();
    syncReady.wait();

    if (captureErrA == hipSuccess) {
      addKernel<<<(kN + 63) / 64, 64, 0, streamA>>>(dA, dA, kN);
      endCaptureErrA = hipStreamEndCapture(streamA, &graphA);
    }
  });

  std::thread threadB([&]() {
    HIP_CHECK_THREAD(hipSetDevice(0));  // Same device as Thread A
    captureReady.wait();

    // This sync API is on the same device as Thread A's capture.
    // Device-scoped invalidation: should invalidate Thread A's capture.
    syncErrB = hipMemcpy(dB, hB, kN * sizeof(int), hipMemcpyHostToDevice);

    syncDone.set_value();
  });

  threadA.join();
  threadB.join();

  INFO("captureErrA=" << captureErrA << " endCaptureErrA=" << endCaptureErrA
                      << " syncErrB=" << syncErrB);

  // Thread B's sync should either:
  // 1. Return an error (hipErrorStreamCaptureImplicit), OR
  // 2. Succeed but invalidate Thread A's capture
  bool syncBlocked = (syncErrB != hipSuccess);
  bool captureInvalidated = (endCaptureErrA != hipSuccess);

  // At least one must be true: either sync was blocked or capture was invalidated
  REQUIRE((syncBlocked || captureInvalidated));

  // If sync succeeded, capture MUST be invalidated (device-scoped semantics)
  if (syncErrB == hipSuccess) {
    REQUIRE(endCaptureErrA != hipSuccess);
  }

  if (graphA) HIP_CHECK(hipGraphDestroy(graphA));
  HIP_CHECK(hipFree(dA));
  HIP_CHECK(hipFree(dB));
  free(hB);
  HIP_CHECK(hipStreamDestroy(streamA));
}

// ================================================================================================
/**
 * Test: Global capture on GPU 0, sync API from another thread on SAME GPU 0.
 *
 * Thread A: Global capture on GPU 0
 * Thread B: hipMemcpy on GPU 0 (same device)
 *
 * Expected: Thread A's capture is INVALIDATED.
 * Same-device sync APIs invalidate ALL captures, regardless of mode.
 */
TEST_CASE("Global_SameDevice_SyncInvalidates", "[graph][capture][multithreaded]") {
  hipStream_t streamA = nullptr;
  int *dA = nullptr, *dB = nullptr, *hB = nullptr;
  hipGraph_t graphA = nullptr;
  hipError_t captureErrA = hipSuccess;
  hipError_t endCaptureErrA = hipSuccess;
  hipError_t syncErrB = hipSuccess;

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipStreamCreate(&streamA));
  HIP_CHECK(hipMalloc(&dA, kN * sizeof(int)));
  HIP_CHECK(hipMalloc(&dB, kN * sizeof(int)));
  hB = (int*)malloc(kN * sizeof(int));
  memset(hB, 0, kN * sizeof(int));

  std::promise<void> captureStarted, syncDone;
  auto captureReady = captureStarted.get_future();
  auto syncReady = syncDone.get_future();

  std::thread threadA([&]() {
    HIP_CHECK_THREAD(hipSetDevice(0));
    captureErrA = hipStreamBeginCapture(streamA, hipStreamCaptureModeGlobal);
    captureStarted.set_value();
    syncReady.wait();

    if (captureErrA == hipSuccess) {
      addKernel<<<(kN + 63) / 64, 64, 0, streamA>>>(dA, dA, kN);
      endCaptureErrA = hipStreamEndCapture(streamA, &graphA);
    }
  });

  std::thread threadB([&]() {
    HIP_CHECK_THREAD(hipSetDevice(0));  // Same device as Thread A
    captureReady.wait();

    syncErrB = hipMemcpy(dB, hB, kN * sizeof(int), hipMemcpyHostToDevice);

    syncDone.set_value();
  });

  threadA.join();
  threadB.join();

  INFO("captureErrA=" << captureErrA << " endCaptureErrA=" << endCaptureErrA
                      << " syncErrB=" << syncErrB);

  bool syncBlocked = (syncErrB != hipSuccess);
  bool captureInvalidated = (endCaptureErrA != hipSuccess);

  REQUIRE((syncBlocked || captureInvalidated));

  if (syncErrB == hipSuccess) {
    REQUIRE(endCaptureErrA != hipSuccess);
  }

  if (graphA) HIP_CHECK(hipGraphDestroy(graphA));
  HIP_CHECK(hipFree(dA));
  HIP_CHECK(hipFree(dB));
  free(hB);
  HIP_CHECK(hipStreamDestroy(streamA));
}

// ================================================================================================
/**
 * Test: Relaxed capture on GPU 0, sync API from another thread on SAME GPU 0.
 *
 * Thread A: Relaxed capture on GPU 0
 * Thread B: hipMemcpy on GPU 0 (same device)
 *
 * Expected: Thread A's capture is INVALIDATED.
 * Relaxed mode doesn't exempt from device-scoped invalidation.
 */
TEST_CASE("Relaxed_SameDevice_SyncInvalidates", "[graph][capture][multithreaded]") {
  hipStream_t streamA = nullptr;
  int *dA = nullptr, *dB = nullptr, *hB = nullptr;
  hipGraph_t graphA = nullptr;
  hipError_t captureErrA = hipSuccess;
  hipError_t endCaptureErrA = hipSuccess;
  hipError_t syncErrB = hipSuccess;

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipStreamCreate(&streamA));
  HIP_CHECK(hipMalloc(&dA, kN * sizeof(int)));
  HIP_CHECK(hipMalloc(&dB, kN * sizeof(int)));
  hB = (int*)malloc(kN * sizeof(int));
  memset(hB, 0, kN * sizeof(int));

  std::promise<void> captureStarted, syncDone;
  auto captureReady = captureStarted.get_future();
  auto syncReady = syncDone.get_future();

  std::thread threadA([&]() {
    HIP_CHECK_THREAD(hipSetDevice(0));
    captureErrA = hipStreamBeginCapture(streamA, hipStreamCaptureModeRelaxed);
    captureStarted.set_value();
    syncReady.wait();

    if (captureErrA == hipSuccess) {
      addKernel<<<(kN + 63) / 64, 64, 0, streamA>>>(dA, dA, kN);
      endCaptureErrA = hipStreamEndCapture(streamA, &graphA);
    }
  });

  std::thread threadB([&]() {
    HIP_CHECK_THREAD(hipSetDevice(0));  // Same device
    captureReady.wait();

    syncErrB = hipMemcpy(dB, hB, kN * sizeof(int), hipMemcpyHostToDevice);

    syncDone.set_value();
  });

  threadA.join();
  threadB.join();

  INFO("captureErrA=" << captureErrA << " endCaptureErrA=" << endCaptureErrA
                      << " syncErrB=" << syncErrB);

  bool syncBlocked = (syncErrB != hipSuccess);
  bool captureInvalidated = (endCaptureErrA != hipSuccess);

  REQUIRE((syncBlocked || captureInvalidated));

  if (syncErrB == hipSuccess) {
    REQUIRE(endCaptureErrA != hipSuccess);
  }

  if (graphA) HIP_CHECK(hipGraphDestroy(graphA));
  HIP_CHECK(hipFree(dA));
  HIP_CHECK(hipFree(dB));
  free(hB);
  HIP_CHECK(hipStreamDestroy(streamA));
}

// ================================================================================================
/**
 * Test: Global capture on GPU 0, sync API from another thread on DIFFERENT GPU 1.
 *
 * Thread A: Global capture on GPU 0
 * Thread B: hipMemcpy on GPU 1 (different device)
 *
 * Expected: Thread A's capture is NOT invalidated.
 * Different-device sync APIs do not trigger cross-device invalidation.
 */
TEST_CASE("Global_DifferentDevice_SyncDoesNotInvalidate",
          "[graph][capture][multithreaded][multi_device]") {
  const int deviceCount = HipTest::getDeviceCount();
  if (deviceCount < 2) {
    HIP_SKIP_TEST(
        "Global_DifferentDevice_SyncDoesNotInvalidate requires at least 2 GPUs - skipping.");
    return;
  }

  hipStream_t streamA = nullptr;
  int *dA = nullptr, *dB = nullptr, *hB = nullptr;
  hipGraph_t graphA = nullptr;
  hipError_t captureErrA = hipSuccess;
  hipError_t endCaptureErrA = hipSuccess;
  hipError_t syncErrB = hipSuccess;

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipStreamCreate(&streamA));
  HIP_CHECK(hipMalloc(&dA, kN * sizeof(int)));

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipMalloc(&dB, kN * sizeof(int)));
  hB = (int*)malloc(kN * sizeof(int));
  memset(hB, 0, kN * sizeof(int));

  std::promise<void> captureStarted, syncDone;
  auto captureReady = captureStarted.get_future();
  auto syncReady = syncDone.get_future();

  std::thread threadA([&]() {
    HIP_CHECK_THREAD(hipSetDevice(0));
    captureErrA = hipStreamBeginCapture(streamA, hipStreamCaptureModeGlobal);
    captureStarted.set_value();
    syncReady.wait();

    if (captureErrA == hipSuccess) {
      addKernel<<<(kN + 63) / 64, 64, 0, streamA>>>(dA, dA, kN);
      endCaptureErrA = hipStreamEndCapture(streamA, &graphA);
    }
  });

  std::thread threadB([&]() {
    HIP_CHECK_THREAD(hipSetDevice(1));  // DIFFERENT device from Thread A
    captureReady.wait();

    // This sync API is on a different device.
    // Should NOT invalidate Thread A's capture (device-scoped).
    syncErrB = hipMemcpy(dB, hB, kN * sizeof(int), hipMemcpyHostToDevice);

    syncDone.set_value();
  });

  threadA.join();
  threadB.join();

  INFO("captureErrA=" << captureErrA << " endCaptureErrA=" << endCaptureErrA
                      << " syncErrB=" << syncErrB);

  // Thread B's sync should succeed (different device)
  REQUIRE(syncErrB == hipSuccess);

  // Thread A's capture should NOT be invalidated (different device)
  REQUIRE(captureErrA == hipSuccess);
  REQUIRE(endCaptureErrA == hipSuccess);
  REQUIRE(graphA != nullptr);

  HIP_CHECK(hipSetDevice(0));
  if (graphA) HIP_CHECK(hipGraphDestroy(graphA));
  HIP_CHECK(hipFree(dA));
  HIP_CHECK(hipStreamDestroy(streamA));

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipFree(dB));
  free(hB);
}

// ================================================================================================
/**
 * Test: ThreadLocal capture on GPU 0, sync API from another thread on DIFFERENT GPU 1.
 *
 * Thread A: ThreadLocal capture on GPU 0
 * Thread B: hipMemcpy on GPU 1 (different device)
 *
 * Expected: Thread A's capture is NOT invalidated.
 * This is the original ROCM-1945 scenario that was fixed.
 */
TEST_CASE("ThreadLocal_DifferentDevice_SyncDoesNotInvalidate",
          "[graph][capture][multithreaded][multi_device]") {
  const int deviceCount = HipTest::getDeviceCount();
  if (deviceCount < 2) {
    HIP_SKIP_TEST(
        "ThreadLocal_DifferentDevice_SyncDoesNotInvalidate requires at least 2 GPUs - skipping.");
    return;
  }

  hipStream_t streamA = nullptr;
  int *dA = nullptr, *dB = nullptr, *hB = nullptr;
  hipGraph_t graphA = nullptr;
  hipError_t captureErrA = hipSuccess;
  hipError_t endCaptureErrA = hipSuccess;
  hipError_t syncErrB = hipSuccess;

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipStreamCreate(&streamA));
  HIP_CHECK(hipMalloc(&dA, kN * sizeof(int)));

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipMalloc(&dB, kN * sizeof(int)));
  hB = (int*)malloc(kN * sizeof(int));
  memset(hB, 0, kN * sizeof(int));

  std::promise<void> captureStarted, syncDone;
  auto captureReady = captureStarted.get_future();
  auto syncReady = syncDone.get_future();

  std::thread threadA([&]() {
    HIP_CHECK_THREAD(hipSetDevice(0));
    captureErrA = hipStreamBeginCapture(streamA, hipStreamCaptureModeThreadLocal);
    captureStarted.set_value();
    syncReady.wait();

    if (captureErrA == hipSuccess) {
      addKernel<<<(kN + 63) / 64, 64, 0, streamA>>>(dA, dA, kN);
      endCaptureErrA = hipStreamEndCapture(streamA, &graphA);
    }
  });

  std::thread threadB([&]() {
    HIP_CHECK_THREAD(hipSetDevice(1));  // DIFFERENT device
    captureReady.wait();

    syncErrB = hipMemcpy(dB, hB, kN * sizeof(int), hipMemcpyHostToDevice);

    syncDone.set_value();
  });

  threadA.join();
  threadB.join();

  INFO("captureErrA=" << captureErrA << " endCaptureErrA=" << endCaptureErrA
                      << " syncErrB=" << syncErrB);

  REQUIRE(syncErrB == hipSuccess);
  REQUIRE(captureErrA == hipSuccess);
  REQUIRE(endCaptureErrA == hipSuccess);
  REQUIRE(graphA != nullptr);

  HIP_CHECK(hipSetDevice(0));
  if (graphA) HIP_CHECK(hipGraphDestroy(graphA));
  HIP_CHECK(hipFree(dA));
  HIP_CHECK(hipStreamDestroy(streamA));

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipFree(dB));
  free(hB);
}

// ================================================================================================
/**
 * Test: Relaxed capture on GPU 0, sync API from another thread on DIFFERENT GPU 1.
 *
 * Thread A: Relaxed capture on GPU 0
 * Thread B: hipMemcpy on GPU 1 (different device)
 *
 * Expected: Thread A's capture is NOT invalidated.
 */
TEST_CASE("Relaxed_DifferentDevice_SyncDoesNotInvalidate",
          "[graph][capture][multithreaded][multi_device]") {
  const int deviceCount = HipTest::getDeviceCount();
  if (deviceCount < 2) {
    HIP_SKIP_TEST(
        "Relaxed_DifferentDevice_SyncDoesNotInvalidate requires at least 2 GPUs - skipping.");
    return;
  }

  hipStream_t streamA = nullptr;
  int *dA = nullptr, *dB = nullptr, *hB = nullptr;
  hipGraph_t graphA = nullptr;
  hipError_t captureErrA = hipSuccess;
  hipError_t endCaptureErrA = hipSuccess;
  hipError_t syncErrB = hipSuccess;

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipStreamCreate(&streamA));
  HIP_CHECK(hipMalloc(&dA, kN * sizeof(int)));

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipMalloc(&dB, kN * sizeof(int)));
  hB = (int*)malloc(kN * sizeof(int));
  memset(hB, 0, kN * sizeof(int));

  std::promise<void> captureStarted, syncDone;
  auto captureReady = captureStarted.get_future();
  auto syncReady = syncDone.get_future();

  std::thread threadA([&]() {
    HIP_CHECK_THREAD(hipSetDevice(0));
    captureErrA = hipStreamBeginCapture(streamA, hipStreamCaptureModeRelaxed);
    captureStarted.set_value();
    syncReady.wait();

    if (captureErrA == hipSuccess) {
      addKernel<<<(kN + 63) / 64, 64, 0, streamA>>>(dA, dA, kN);
      endCaptureErrA = hipStreamEndCapture(streamA, &graphA);
    }
  });

  std::thread threadB([&]() {
    HIP_CHECK_THREAD(hipSetDevice(1));  // DIFFERENT device
    captureReady.wait();

    syncErrB = hipMemcpy(dB, hB, kN * sizeof(int), hipMemcpyHostToDevice);

    syncDone.set_value();
  });

  threadA.join();
  threadB.join();

  INFO("captureErrA=" << captureErrA << " endCaptureErrA=" << endCaptureErrA
                      << " syncErrB=" << syncErrB);

  REQUIRE(syncErrB == hipSuccess);
  REQUIRE(captureErrA == hipSuccess);
  REQUIRE(endCaptureErrA == hipSuccess);
  REQUIRE(graphA != nullptr);

  HIP_CHECK(hipSetDevice(0));
  if (graphA) HIP_CHECK(hipGraphDestroy(graphA));
  HIP_CHECK(hipFree(dA));
  HIP_CHECK(hipStreamDestroy(streamA));

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipFree(dB));
  free(hB);
}

// ================================================================================================
/**
 * Test: ThreadLocal capture on GPU 0, graphLaunch from another thread on GPU 1.
 *
 * Thread A: ThreadLocal capture on GPU 0, holds it open
 * Thread B: ThreadLocal capture on GPU 1, ends it, instantiates, then calls hipGraphLaunch
 *
 * Expected: Thread A's capture survives Thread B's graph launch (different device).
 *
 * This is the original ROCM-1945 regression scenario: hipGraphLaunch internally
 * triggers CHECK_STREAM_CAPTURING() on the null stream. Device-scoped invalidation
 * means it should only affect captures on the same device.
 */
TEST_CASE("ThreadLocal_GraphLaunch_DifferentDevice",
          "[graph][capture][multithreaded][multi_device]") {
  const int deviceCount = HipTest::getDeviceCount();
  if (deviceCount < 2) {
    HIP_SKIP_TEST(
        "ThreadLocal_GraphLaunch_DifferentDevice requires at least 2 GPUs - skipping.");
    return;
  }

  hipStream_t streamA = nullptr, streamB = nullptr;
  int* dA = nullptr;
  int* dB_src = nullptr;
  int* dB_dst = nullptr;
  hipGraph_t graphA = nullptr, graphB = nullptr;
  hipGraphExec_t execB = nullptr;
  hipError_t captureErrA = hipSuccess;
  hipError_t captureErrB = hipSuccess;
  hipError_t launchErrB = hipSuccess;

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipStreamCreate(&streamA));
  HIP_CHECK(hipMalloc(&dA, kN * sizeof(int)));

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipStreamCreate(&streamB));
  HIP_CHECK(hipMalloc(&dB_src, kN * sizeof(int)));
  HIP_CHECK(hipMalloc(&dB_dst, kN * sizeof(int)));
  HIP_CHECK(hipMemset(dB_src, 1, kN * sizeof(int)));
  HIP_CHECK(hipMemset(dB_dst, 0, kN * sizeof(int)));

  std::promise<void> captureStarted, launchDone;
  auto captureReady = captureStarted.get_future();
  auto launchReady = launchDone.get_future();

  std::thread threadA([&]() {
    HIP_CHECK_THREAD(hipSetDevice(0));
    captureErrA = hipStreamBeginCapture(streamA, hipStreamCaptureModeThreadLocal);

    captureStarted.set_value();  // signal B: A's capture is active
    launchReady.wait();          // wait for B: graph launch done

    if (captureErrA == hipSuccess) {
      addKernel<<<(kN + 63) / 64, 64, 0, streamA>>>(dA, dA, kN);
      captureErrA = hipStreamEndCapture(streamA, &graphA);
    }
  });

  std::thread threadB([&]() {
    HIP_CHECK_THREAD(hipSetDevice(1));
    captureReady.wait();  // wait for A's capture to start

    captureErrB = hipStreamBeginCapture(streamB, hipStreamCaptureModeThreadLocal);
    if (captureErrB == hipSuccess) {
      addKernel<<<(kN + 63) / 64, 64, 0, streamB>>>(dB_dst, dB_src, kN);
      captureErrB = hipStreamEndCapture(streamB, &graphB);
    }
    if (captureErrB == hipSuccess) {
      captureErrB = hipGraphInstantiate(&execB, graphB, nullptr, nullptr, 0);
    }
    if (captureErrB == hipSuccess) {
      // This hipGraphLaunch internally triggers CHECK_STREAM_CAPTURING on the null stream.
      // Device-scoped: it must NOT invalidate Thread A's capture (different device).
      launchErrB = hipGraphLaunch(execB, streamB);
      HIP_CHECK_THREAD(hipStreamSynchronize(streamB));
    }

    launchDone.set_value();  // signal A: graph launch complete
  });

  threadA.join();
  threadB.join();

  INFO("captureErrA=" << captureErrA << " captureErrB=" << captureErrB
                      << " launchErrB=" << launchErrB);
  REQUIRE(captureErrB == hipSuccess);
  REQUIRE(launchErrB == hipSuccess);
  REQUIRE(captureErrA == hipSuccess);

  HIP_CHECK(hipSetDevice(0));
  if (graphA) HIP_CHECK(hipGraphDestroy(graphA));
  HIP_CHECK(hipFree(dA));
  HIP_CHECK(hipStreamDestroy(streamA));

  HIP_CHECK(hipSetDevice(1));
  if (execB) HIP_CHECK(hipGraphExecDestroy(execB));
  if (graphB) HIP_CHECK(hipGraphDestroy(graphB));
  HIP_CHECK(hipFree(dB_src));
  HIP_CHECK(hipFree(dB_dst));
  HIP_CHECK(hipStreamDestroy(streamB));
}
