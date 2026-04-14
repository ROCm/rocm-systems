/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipEventQueryCaptureRestrictions
 * @{
 * @ingroup GraphTest
 *
 * Tests to verify hipEventQuery and hipEventSynchronize behavior with different
 * stream capture modes across multiple threads.
 *
 * This validates the fix for:
 * - hipEventQuery incorrectly returning hipErrorStreamCaptureUnsupported when
 *   the calling thread has switched to RELAXED or THREAD_LOCAL mode
 *
 * APIs tested:
 *   - hipEventQuery
 *   - hipEventSynchronize
 *
 * Capture mode behavior matrix:
 *
 * | Calling Thread Mode | T1 has GLOBAL Capture | Event Type | Expected Result |
 * |---------------------|----------------------|------------|-----------------|
 * | GLOBAL (default)    | Yes                  | Normal     | UNSUPPORTED     |
 * | GLOBAL (default)    | Yes                  | Captured   | CAPTURED_EVENT  |
 * | RELAXED             | Yes                  | Normal     | SUCCESS         |
 * | THREAD_LOCAL        | Yes                  | Normal     | SUCCESS         |
 * | GLOBAL (default)    | THREAD_LOCAL capture | Normal     | SUCCESS         |
 * | GLOBAL (default)    | THREAD_LOCAL capture | Captured   | CAPTURED_EVENT  |
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#include <atomic>
#include <chrono>
#include <thread>

// Simple kernel for testing
static __global__ void dummyKernel(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = data[idx] * 2.0f;
  }
}

// ============================================================================
// Synchronization primitives
// ============================================================================
static std::atomic<bool> g_thread1_capturing{false};
static std::atomic<bool> g_thread2_can_start{false};
static std::atomic<bool> g_thread2_done{false};
static std::atomic<hipError_t> g_thread2_result{hipSuccess};

// Shared resources for cross-thread testing
static hipStream_t g_capture_stream = nullptr;
static hipEvent_t g_normal_event = nullptr;      // Event recorded before capture
static hipEvent_t g_captured_event = nullptr;    // Event recorded during capture
static hipStream_t g_non_capture_stream = nullptr;

static void resetSyncState() {
  g_thread1_capturing = false;
  g_thread2_can_start = false;
  g_thread2_done = false;
  g_thread2_result = hipSuccess;
  g_capture_stream = nullptr;
  g_normal_event = nullptr;
  g_captured_event = nullptr;
  g_non_capture_stream = nullptr;
}

// ============================================================================
// Thread 1: Capture Thread Functions
// ============================================================================

static void thread1_global_capture_with_events() {
  float* d_data;
  HIP_CHECK(hipMalloc(&d_data, 1024 * sizeof(float)));

  // Create streams and events
  HIP_CHECK(hipStreamCreate(&g_capture_stream));
  HIP_CHECK(hipStreamCreate(&g_non_capture_stream));
  HIP_CHECK(hipEventCreate(&g_normal_event));
  HIP_CHECK(hipEventCreate(&g_captured_event));

  // Record normal event BEFORE capture
  dummyKernel<<<1, 256, 0, g_non_capture_stream>>>(d_data, 1024);
  HIP_CHECK(hipEventRecord(g_normal_event, g_non_capture_stream));
  HIP_CHECK(hipStreamSynchronize(g_non_capture_stream));

  // Start GLOBAL capture
  HIP_CHECK(hipStreamBeginCapture(g_capture_stream, hipStreamCaptureModeGlobal));

  // Launch kernel during capture
  dummyKernel<<<1, 256, 0, g_capture_stream>>>(d_data, 1024);

  // Record captured event DURING capture
  HIP_CHECK(hipEventRecord(g_captured_event, g_capture_stream));

  // Signal thread 2 to start
  g_thread1_capturing = true;
  g_thread2_can_start = true;

  // Wait for thread 2 to complete
  while (!g_thread2_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // End capture
  hipGraph_t graph;
  hipError_t endResult = hipStreamEndCapture(g_capture_stream, &graph);

  // Cleanup - capture may have been invalidated
  if (endResult == hipSuccess && graph != nullptr) {
    HIP_CHECK(hipGraphDestroy(graph));
  }
  HIP_CHECK(hipEventDestroy(g_captured_event));
  HIP_CHECK(hipEventDestroy(g_normal_event));
  HIP_CHECK(hipFree(d_data));
  HIP_CHECK(hipStreamDestroy(g_non_capture_stream));
  HIP_CHECK(hipStreamDestroy(g_capture_stream));
}

static void thread1_threadlocal_capture_with_events() {
  float* d_data;
  HIP_CHECK(hipMalloc(&d_data, 1024 * sizeof(float)));

  HIP_CHECK(hipStreamCreate(&g_capture_stream));
  HIP_CHECK(hipStreamCreate(&g_non_capture_stream));
  HIP_CHECK(hipEventCreate(&g_normal_event));
  HIP_CHECK(hipEventCreate(&g_captured_event));

  // Record normal event BEFORE capture
  dummyKernel<<<1, 256, 0, g_non_capture_stream>>>(d_data, 1024);
  HIP_CHECK(hipEventRecord(g_normal_event, g_non_capture_stream));
  HIP_CHECK(hipStreamSynchronize(g_non_capture_stream));

  // Start THREAD_LOCAL capture
  HIP_CHECK(hipStreamBeginCapture(g_capture_stream, hipStreamCaptureModeThreadLocal));

  dummyKernel<<<1, 256, 0, g_capture_stream>>>(d_data, 1024);
  HIP_CHECK(hipEventRecord(g_captured_event, g_capture_stream));

  g_thread1_capturing = true;
  g_thread2_can_start = true;

  while (!g_thread2_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  hipGraph_t graph;
  hipError_t endResult = hipStreamEndCapture(g_capture_stream, &graph);

  if (endResult == hipSuccess && graph != nullptr) {
    HIP_CHECK(hipGraphDestroy(graph));
  }
  HIP_CHECK(hipEventDestroy(g_captured_event));
  HIP_CHECK(hipEventDestroy(g_normal_event));
  HIP_CHECK(hipFree(d_data));
  HIP_CHECK(hipStreamDestroy(g_non_capture_stream));
  HIP_CHECK(hipStreamDestroy(g_capture_stream));
}

// ============================================================================
// Thread 2: hipEventQuery Functions
// ============================================================================

// Default mode - query normal event
static void thread2_default_mode_event_query_normal() {
  while (!g_thread2_can_start) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  g_thread2_result = hipEventQuery(g_normal_event);
  g_thread2_done = true;
}

// Default mode - query captured event
static void thread2_default_mode_event_query_captured() {
  while (!g_thread2_can_start) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  g_thread2_result = hipEventQuery(g_captured_event);
  g_thread2_done = true;
}

// RELAXED mode - query normal event
static void thread2_relaxed_mode_event_query_normal() {
  while (!g_thread2_can_start) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  hipStreamCaptureMode oldMode;
  hipStreamCaptureMode newMode = hipStreamCaptureModeRelaxed;
  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&newMode));
  oldMode = newMode;

  g_thread2_result = hipEventQuery(g_normal_event);

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&oldMode));
  g_thread2_done = true;
}

// THREAD_LOCAL mode - query normal event
static void thread2_threadlocal_mode_event_query_normal() {
  while (!g_thread2_can_start) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  hipStreamCaptureMode oldMode;
  hipStreamCaptureMode newMode = hipStreamCaptureModeThreadLocal;
  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&newMode));
  oldMode = newMode;

  g_thread2_result = hipEventQuery(g_normal_event);

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&oldMode));
  g_thread2_done = true;
}

// ============================================================================
// Thread 2: hipEventSynchronize Functions
// ============================================================================

// Default mode - sync normal event
static void thread2_default_mode_event_sync_normal() {
  while (!g_thread2_can_start) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  g_thread2_result = hipEventSynchronize(g_normal_event);
  g_thread2_done = true;
}

// Default mode - sync captured event
static void thread2_default_mode_event_sync_captured() {
  while (!g_thread2_can_start) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  g_thread2_result = hipEventSynchronize(g_captured_event);
  g_thread2_done = true;
}

// RELAXED mode - sync normal event
static void thread2_relaxed_mode_event_sync_normal() {
  while (!g_thread2_can_start) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  hipStreamCaptureMode oldMode;
  hipStreamCaptureMode newMode = hipStreamCaptureModeRelaxed;
  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&newMode));
  oldMode = newMode;

  g_thread2_result = hipEventSynchronize(g_normal_event);

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&oldMode));
  g_thread2_done = true;
}

// THREAD_LOCAL mode - sync normal event
static void thread2_threadlocal_mode_event_sync_normal() {
  while (!g_thread2_can_start) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  hipStreamCaptureMode oldMode;
  hipStreamCaptureMode newMode = hipStreamCaptureModeThreadLocal;
  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&newMode));
  oldMode = newMode;

  g_thread2_result = hipEventSynchronize(g_normal_event);

  HIP_CHECK(hipThreadExchangeStreamCaptureMode(&oldMode));
  g_thread2_done = true;
}

// ============================================================================
// Test Helper
// ============================================================================
static void runCrossThreadTest(void (*thread1Func)(), void (*thread2Func)(),
                               hipError_t expectedResult, const char* testName) {
  INFO("Running test: " << testName);
  resetSyncState();

  std::thread t1(thread1Func);
  std::thread t2(thread2Func);

  t1.join();
  t2.join();

  hipError_t result = g_thread2_result;
  INFO("Result: " << result << ", Expected: " << expectedResult);

  if (expectedResult == hipSuccess) {
    // Allow hipSuccess or hipErrorNotReady for success cases
    REQUIRE((result == hipSuccess || result == hipErrorNotReady));
  } else {
    REQUIRE(result == expectedResult);
  }
}

// ============================================================================
// TEST CASES: GLOBAL Capture - Default Mode (BLOCKED)
// ============================================================================

/**
 * Test: During GLOBAL capture, hipEventQuery on normal event from default-mode thread
 * Expected: hipErrorStreamCaptureUnsupported (blocked by cross-thread GLOBAL capture)
 */
TEST_CASE("Unit_hipEventQuery_GlobalCapture_DefaultMode_NormalEvent") {
  runCrossThreadTest(thread1_global_capture_with_events,
                     thread2_default_mode_event_query_normal,
                     hipErrorStreamCaptureUnsupported,
                     "GLOBAL capture, Default mode, hipEventQuery(normal event)");
}

/**
 * Test: During GLOBAL capture, hipEventQuery on captured event from default-mode thread
 * Expected: hipErrorCapturedEvent (event was recorded during capture)
 */
TEST_CASE("Unit_hipEventQuery_GlobalCapture_DefaultMode_CapturedEvent") {
  runCrossThreadTest(thread1_global_capture_with_events,
                     thread2_default_mode_event_query_captured,
                     hipErrorCapturedEvent,
                     "GLOBAL capture, Default mode, hipEventQuery(captured event)");
}

/**
 * Test: During GLOBAL capture, hipEventSynchronize on normal event from default-mode thread
 * Expected: hipErrorStreamCaptureUnsupported (blocked by cross-thread GLOBAL capture)
 */
TEST_CASE("Unit_hipEventSynchronize_GlobalCapture_DefaultMode_NormalEvent") {
  runCrossThreadTest(thread1_global_capture_with_events,
                     thread2_default_mode_event_sync_normal,
                     hipErrorStreamCaptureUnsupported,
                     "GLOBAL capture, Default mode, hipEventSynchronize(normal event)");
}

/**
 * Test: During GLOBAL capture, hipEventSynchronize on captured event from default-mode thread
 * Expected: hipErrorCapturedEvent (event was recorded during capture)
 */
TEST_CASE("Unit_hipEventSynchronize_GlobalCapture_DefaultMode_CapturedEvent") {
  runCrossThreadTest(thread1_global_capture_with_events,
                     thread2_default_mode_event_sync_captured,
                     hipErrorCapturedEvent,
                     "GLOBAL capture, Default mode, hipEventSynchronize(captured event)");
}

// ============================================================================
// TEST CASES: GLOBAL Capture - RELAXED Mode (ALLOWED)
// These tests validate the fix for the RCCL/PyTorch watchdog issue
// ============================================================================

/**
 * Test: During GLOBAL capture, hipEventQuery on normal event from RELAXED-mode thread
 * Expected: hipSuccess (RELAXED mode bypasses cross-thread GLOBAL capture check)
 *
 * This is the key fix scenario: RCCL watchdog thread switches to RELAXED mode
 * and should be able to query events even when another thread has GLOBAL capture active.
 */
TEST_CASE("Unit_hipEventQuery_GlobalCapture_RelaxedMode_NormalEvent") {
  runCrossThreadTest(thread1_global_capture_with_events,
                     thread2_relaxed_mode_event_query_normal,
                     hipSuccess,
                     "GLOBAL capture, RELAXED mode, hipEventQuery(normal event)");
}

/**
 * Test: During GLOBAL capture, hipEventSynchronize on normal event from RELAXED-mode thread
 * Expected: hipSuccess (RELAXED mode bypasses cross-thread GLOBAL capture check)
 */
TEST_CASE("Unit_hipEventSynchronize_GlobalCapture_RelaxedMode_NormalEvent") {
  runCrossThreadTest(thread1_global_capture_with_events,
                     thread2_relaxed_mode_event_sync_normal,
                     hipSuccess,
                     "GLOBAL capture, RELAXED mode, hipEventSynchronize(normal event)");
}

// ============================================================================
// TEST CASES: GLOBAL Capture - THREAD_LOCAL Mode (ALLOWED)
// ============================================================================

/**
 * Test: During GLOBAL capture, hipEventQuery on normal event from THREAD_LOCAL-mode thread
 * Expected: hipSuccess (THREAD_LOCAL mode skips cross-thread GLOBAL capture check)
 */
TEST_CASE("Unit_hipEventQuery_GlobalCapture_ThreadLocalMode_NormalEvent") {
  runCrossThreadTest(thread1_global_capture_with_events,
                     thread2_threadlocal_mode_event_query_normal,
                     hipSuccess,
                     "GLOBAL capture, THREAD_LOCAL mode, hipEventQuery(normal event)");
}

/**
 * Test: During GLOBAL capture, hipEventSynchronize on normal event from THREAD_LOCAL-mode thread
 * Expected: hipSuccess (THREAD_LOCAL mode skips cross-thread GLOBAL capture check)
 */
TEST_CASE("Unit_hipEventSynchronize_GlobalCapture_ThreadLocalMode_NormalEvent") {
  runCrossThreadTest(thread1_global_capture_with_events,
                     thread2_threadlocal_mode_event_sync_normal,
                     hipSuccess,
                     "GLOBAL capture, THREAD_LOCAL mode, hipEventSynchronize(normal event)");
}

// ============================================================================
// TEST CASES: THREAD_LOCAL Capture - Default Mode (ALLOWED)
// THREAD_LOCAL capture should NOT block other threads
// ============================================================================

/**
 * Test: During THREAD_LOCAL capture, hipEventQuery on normal event from default-mode thread
 * Expected: hipSuccess (THREAD_LOCAL capture does NOT block other threads)
 */
TEST_CASE("Unit_hipEventQuery_ThreadLocalCapture_DefaultMode_NormalEvent") {
  runCrossThreadTest(thread1_threadlocal_capture_with_events,
                     thread2_default_mode_event_query_normal,
                     hipSuccess,
                     "THREAD_LOCAL capture, Default mode, hipEventQuery(normal event)");
}

/**
 * Test: During THREAD_LOCAL capture, hipEventQuery on captured event from default-mode thread
 * Expected: hipErrorCapturedEvent (captured event cannot be queried regardless of mode)
 */
TEST_CASE("Unit_hipEventQuery_ThreadLocalCapture_DefaultMode_CapturedEvent") {
  runCrossThreadTest(thread1_threadlocal_capture_with_events,
                     thread2_default_mode_event_query_captured,
                     hipErrorCapturedEvent,
                     "THREAD_LOCAL capture, Default mode, hipEventQuery(captured event)");
}

/**
 * Test: During THREAD_LOCAL capture, hipEventSynchronize on normal event from default-mode thread
 * Expected: hipSuccess (THREAD_LOCAL capture does NOT block other threads)
 */
TEST_CASE("Unit_hipEventSynchronize_ThreadLocalCapture_DefaultMode_NormalEvent") {
  runCrossThreadTest(thread1_threadlocal_capture_with_events,
                     thread2_default_mode_event_sync_normal,
                     hipSuccess,
                     "THREAD_LOCAL capture, Default mode, hipEventSynchronize(normal event)");
}

/**
 * Test: During THREAD_LOCAL capture, hipEventSynchronize on captured event from default-mode thread
 * Expected: hipErrorCapturedEvent (captured event cannot be synchronized regardless of mode)
 */
TEST_CASE("Unit_hipEventSynchronize_ThreadLocalCapture_DefaultMode_CapturedEvent") {
  runCrossThreadTest(thread1_threadlocal_capture_with_events,
                     thread2_default_mode_event_sync_captured,
                     hipErrorCapturedEvent,
                     "THREAD_LOCAL capture, Default mode, hipEventSynchronize(captured event)");
}

// ============================================================================
// TEST CASES: Same Thread - Operations during capture
// ============================================================================

/**
 * Test: Same thread calls hipEventQuery on normal event while capturing in GLOBAL mode
 * Expected: hipErrorStreamCaptureUnsupported (same thread blocked in GLOBAL mode)
 */
TEST_CASE("Unit_hipEventQuery_SameThread_GlobalCapture") {
  hipStream_t captureStream;
  hipStream_t otherStream;
  hipEvent_t normalEvent;
  float* d_data;

  HIP_CHECK(hipStreamCreate(&captureStream));
  HIP_CHECK(hipStreamCreate(&otherStream));
  HIP_CHECK(hipEventCreate(&normalEvent));
  HIP_CHECK(hipMalloc(&d_data, 1024 * sizeof(float)));

  // Record event before capture
  dummyKernel<<<1, 256, 0, otherStream>>>(d_data, 1024);
  HIP_CHECK(hipEventRecord(normalEvent, otherStream));
  HIP_CHECK(hipStreamSynchronize(otherStream));

  // Start GLOBAL capture
  HIP_CHECK(hipStreamBeginCapture(captureStream, hipStreamCaptureModeGlobal));
  dummyKernel<<<1, 256, 0, captureStream>>>(d_data, 1024);

  // Try to query event while capturing - should fail
  hipError_t result = hipEventQuery(normalEvent);
  REQUIRE(result == hipErrorStreamCaptureUnsupported);

  // End capture (will be invalidated)
  hipGraph_t graph;
  hipError_t endResult = hipStreamEndCapture(captureStream, &graph);
  REQUIRE(endResult == hipErrorStreamCaptureInvalidated);

  HIP_CHECK(hipEventDestroy(normalEvent));
  HIP_CHECK(hipFree(d_data));
  HIP_CHECK(hipStreamDestroy(otherStream));
  HIP_CHECK(hipStreamDestroy(captureStream));
}

/**
 * Test: Same thread calls hipEventQuery on normal event while capturing in RELAXED mode
 * Expected: hipSuccess (RELAXED mode allows the operation)
 */
TEST_CASE("Unit_hipEventQuery_SameThread_RelaxedCapture") {
  hipStream_t captureStream;
  hipStream_t otherStream;
  hipEvent_t normalEvent;
  float* d_data;

  HIP_CHECK(hipStreamCreate(&captureStream));
  HIP_CHECK(hipStreamCreate(&otherStream));
  HIP_CHECK(hipEventCreate(&normalEvent));
  HIP_CHECK(hipMalloc(&d_data, 1024 * sizeof(float)));

  // Record event before capture
  dummyKernel<<<1, 256, 0, otherStream>>>(d_data, 1024);
  HIP_CHECK(hipEventRecord(normalEvent, otherStream));
  HIP_CHECK(hipStreamSynchronize(otherStream));

  // Start RELAXED capture
  HIP_CHECK(hipStreamBeginCapture(captureStream, hipStreamCaptureModeRelaxed));
  dummyKernel<<<1, 256, 0, captureStream>>>(d_data, 1024);

  // Query event while capturing in RELAXED mode - should succeed
  hipError_t result = hipEventQuery(normalEvent);
  REQUIRE((result == hipSuccess || result == hipErrorNotReady));

  // End capture successfully
  hipGraph_t graph;
  HIP_CHECK(hipStreamEndCapture(captureStream, &graph));
  HIP_CHECK(hipGraphDestroy(graph));

  HIP_CHECK(hipEventDestroy(normalEvent));
  HIP_CHECK(hipFree(d_data));
  HIP_CHECK(hipStreamDestroy(otherStream));
  HIP_CHECK(hipStreamDestroy(captureStream));
}

/**
 * @}
 */
