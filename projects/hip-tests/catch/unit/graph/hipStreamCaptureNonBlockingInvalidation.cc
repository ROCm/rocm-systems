/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipStreamCaptureNonBlockingInvalidation
 * @{
 * @ingroup GraphTest
 *
 * Tests to verify that legacy-stream sync operations (hipMemset, hipMemcpy) and
 * hipStreamSynchronize on a non-capturing stream do not invalidate Relaxed or
 * ThreadLocal captures on NonBlocking streams, matching CUDA semantics.
 *
 * Bug: ROCM-33595
 *
 * APIs tested:
 *   - hipMemset
 *   - hipMemcpy
 *   - hipStreamSynchronize
 *
 * CUDA semantics:
 *   - A NonBlocking stream has no implicit dependency on the legacy stream.
 *   - A Relaxed or ThreadLocal capture on a NonBlocking stream should not be
 *     affected by sync operations on the legacy stream or other streams.
 *   - Only Global-mode captures, or captures on blocking streams, should be
 *     invalidated by legacy-stream operations.
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#include <atomic>
#include <chrono>
#include <thread>

static __global__ void captureKernel(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = static_cast<float>(idx);
  }
}

// ============================================================================
// Helpers
// ============================================================================

static constexpr int kDataSize = 256;

struct CaptureState {
  std::atomic<bool> capture_active{false};
  std::atomic<bool> api_done{false};
  std::atomic<hipStreamCaptureStatus> capture_status{hipStreamCaptureStatusNone};
  std::atomic<hipError_t> end_capture_err{hipSuccess};
};

// Thread function: begin capture, signal, wait for API, check status, end capture.
static void captureThread(hipStreamCaptureMode mode, bool nonblocking,
                          CaptureState& state) {
  hipStream_t stream;
  if (nonblocking) {
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  } else {
    HIP_CHECK(hipStreamCreate(&stream));
  }

  float* d_data;
  HIP_CHECK(hipMalloc(&d_data, kDataSize * sizeof(float)));
  HIP_CHECK(hipStreamBeginCapture(stream, mode));
  captureKernel<<<1, kDataSize, 0, stream>>>(d_data, kDataSize);

  state.capture_active = true;

  while (!state.api_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  hipStreamCaptureStatus st;
  HIP_CHECK(hipStreamGetCaptureInfo(stream, &st, nullptr));
  state.capture_status = st;

  hipGraph_t graph;
  hipError_t err = hipStreamEndCapture(stream, &graph);
  state.end_capture_err = err;
  if (err == hipSuccess && graph != nullptr) {
    HIP_CHECK(hipGraphDestroy(graph));
  }

  HIP_CHECK(hipFree(d_data));
  HIP_CHECK(hipStreamDestroy(stream));
}

// ============================================================================
// A: hipMemset on legacy stream — cross-thread capture
// ============================================================================

static void testMemsetCrossThread(hipStreamCaptureMode mode, bool nonblocking,
                                  hipStreamCaptureStatus expectedStatus) {
  float* d_buf;
  HIP_CHECK(hipMalloc(&d_buf, kDataSize * sizeof(float)));

  CaptureState state;
  std::thread t(captureThread, mode, nonblocking, std::ref(state));

  while (!state.capture_active) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  (void)hipMemset(d_buf, 0, kDataSize * sizeof(float));
  (void)hipGetLastError();

  state.api_done = true;
  t.join();

  REQUIRE(state.capture_status == expectedStatus);
  HIP_CHECK(hipFree(d_buf));
}

HIP_TEST_CASE(Unit_hipMemset_LegacyStream_GlobalCapture_BlockingStream) {
  testMemsetCrossThread(hipStreamCaptureModeGlobal, false, hipStreamCaptureStatusInvalidated);
}

HIP_TEST_CASE(Unit_hipMemset_LegacyStream_GlobalCapture_NonBlockingStream) {
  testMemsetCrossThread(hipStreamCaptureModeGlobal, true, hipStreamCaptureStatusInvalidated);
}

HIP_TEST_CASE(Unit_hipMemset_LegacyStream_ThreadLocalCapture_BlockingStream) {
  testMemsetCrossThread(hipStreamCaptureModeThreadLocal, false, hipStreamCaptureStatusInvalidated);
}

HIP_TEST_CASE(Unit_hipMemset_LegacyStream_ThreadLocalCapture_NonBlockingStream) {
  testMemsetCrossThread(hipStreamCaptureModeThreadLocal, true, hipStreamCaptureStatusActive);
}

HIP_TEST_CASE(Unit_hipMemset_LegacyStream_RelaxedCapture_BlockingStream) {
  testMemsetCrossThread(hipStreamCaptureModeRelaxed, false, hipStreamCaptureStatusInvalidated);
}

HIP_TEST_CASE(Unit_hipMemset_LegacyStream_RelaxedCapture_NonBlockingStream) {
  testMemsetCrossThread(hipStreamCaptureModeRelaxed, true, hipStreamCaptureStatusActive);
}

// ============================================================================
// B: hipMemcpy on legacy stream — cross-thread capture
// ============================================================================

static void testMemcpyCrossThread(hipStreamCaptureMode mode, bool nonblocking,
                                  hipStreamCaptureStatus expectedStatus) {
  float* d_src;
  float* d_dst;
  HIP_CHECK(hipMalloc(&d_src, kDataSize * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_dst, kDataSize * sizeof(float)));

  CaptureState state;
  std::thread t(captureThread, mode, nonblocking, std::ref(state));

  while (!state.capture_active) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  (void)hipMemcpy(d_dst, d_src, kDataSize * sizeof(float), hipMemcpyDeviceToDevice);
  (void)hipGetLastError();

  state.api_done = true;
  t.join();

  REQUIRE(state.capture_status == expectedStatus);
  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));
}

HIP_TEST_CASE(Unit_hipMemcpy_LegacyStream_ThreadLocalCapture_NonBlockingStream) {
  testMemcpyCrossThread(hipStreamCaptureModeThreadLocal, true, hipStreamCaptureStatusActive);
}

HIP_TEST_CASE(Unit_hipMemcpy_LegacyStream_RelaxedCapture_NonBlockingStream) {
  testMemcpyCrossThread(hipStreamCaptureModeRelaxed, true, hipStreamCaptureStatusActive);
}

HIP_TEST_CASE(Unit_hipMemcpy_LegacyStream_GlobalCapture_BlockingStream) {
  testMemcpyCrossThread(hipStreamCaptureModeGlobal, false, hipStreamCaptureStatusInvalidated);
}

HIP_TEST_CASE(Unit_hipMemcpy_LegacyStream_RelaxedCapture_BlockingStream) {
  testMemcpyCrossThread(hipStreamCaptureModeRelaxed, false, hipStreamCaptureStatusInvalidated);
}

// ============================================================================
// B2: NonBlocking origin with blocking participant — legacy-stream ops must
//     still invalidate because the blocking participant has a legacy-stream
//     dependency (verified on CUDA Blackwell).
// ============================================================================

static void captureWithBlockingParticipant(hipStreamCaptureMode mode,
                                           std::atomic<bool>& active,
                                           std::atomic<bool>& done,
                                           std::atomic<hipStreamCaptureStatus>& status) {
  hipStream_t nbStream, bStream;
  HIP_CHECK(hipStreamCreateWithFlags(&nbStream, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreate(&bStream));

  float* d;
  HIP_CHECK(hipMalloc(&d, kDataSize * sizeof(float)));
  hipEvent_t forkEvent, joinEvent;
  HIP_CHECK(hipEventCreate(&forkEvent));
  HIP_CHECK(hipEventCreate(&joinEvent));

  HIP_CHECK(hipStreamBeginCapture(nbStream, mode));
  captureKernel<<<1, kDataSize, 0, nbStream>>>(d, kDataSize);

  // Fork: blocking stream joins the capture
  HIP_CHECK(hipEventRecord(forkEvent, nbStream));
  HIP_CHECK(hipStreamWaitEvent(bStream, forkEvent, 0));
  captureKernel<<<1, kDataSize, 0, bStream>>>(d, kDataSize);

  active = true;

  while (!done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  hipStreamCaptureStatus st;
  HIP_CHECK(hipStreamGetCaptureInfo(nbStream, &st, nullptr));
  status = st;

  // Only rejoin if capture is still active — on invalidated captures these
  // would return hipErrorStreamCaptureInvalidated and HIP_CHECK would assert
  // inside a worker thread where Catch2 assertions are not thread-safe.
  if (st == hipStreamCaptureStatusActive) {
    HIP_CHECK(hipEventRecord(joinEvent, bStream));
    HIP_CHECK(hipStreamWaitEvent(nbStream, joinEvent, 0));
  }

  hipGraph_t graph;
  hipError_t endErr = hipStreamEndCapture(nbStream, &graph);
  if (endErr == hipSuccess && graph != nullptr) {
    HIP_CHECK(hipGraphDestroy(graph));
  }

  HIP_CHECK(hipEventDestroy(joinEvent));
  HIP_CHECK(hipEventDestroy(forkEvent));
  HIP_CHECK(hipFree(d));
  HIP_CHECK(hipStreamDestroy(bStream));
  HIP_CHECK(hipStreamDestroy(nbStream));
}

static void testMemsetWithBlockingParticipant(hipStreamCaptureMode mode,
                                              hipStreamCaptureStatus expectedStatus) {
  float* d_buf;
  HIP_CHECK(hipMalloc(&d_buf, kDataSize * sizeof(float)));

  std::atomic<bool> active{false}, done{false};
  std::atomic<hipStreamCaptureStatus> status{hipStreamCaptureStatusNone};

  std::thread t(captureWithBlockingParticipant, mode,
                std::ref(active), std::ref(done), std::ref(status));

  while (!active) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  (void)hipMemset(d_buf, 0, kDataSize * sizeof(float));
  (void)hipGetLastError();

  done = true;
  t.join();

  REQUIRE(status == expectedStatus);
  HIP_CHECK(hipFree(d_buf));
}

HIP_TEST_CASE(Unit_hipMemset_NonBlockingOrigin_BlockingParticipant_Relaxed_Invalidated) {
  testMemsetWithBlockingParticipant(hipStreamCaptureModeRelaxed,
                                   hipStreamCaptureStatusInvalidated);
}

HIP_TEST_CASE(Unit_hipMemset_NonBlockingOrigin_BlockingParticipant_ThreadLocal_Invalidated) {
  testMemsetWithBlockingParticipant(hipStreamCaptureModeThreadLocal,
                                   hipStreamCaptureStatusInvalidated);
}

// ============================================================================
// C: hipStreamSynchronize on SAME capturing stream — always invalidated
// ============================================================================

HIP_TEST_CASE(Unit_hipStreamSync_SameStream_GlobalCapture_BlockingStream) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  float* d;
  HIP_CHECK(hipMalloc(&d, kDataSize * sizeof(float)));

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  captureKernel<<<1, kDataSize, 0, stream>>>(d, kDataSize);

  HIP_CHECK_ERROR(hipStreamSynchronize(stream), hipErrorStreamCaptureUnsupported);

  hipStreamCaptureStatus st;
  HIP_CHECK(hipStreamGetCaptureInfo(stream, &st, nullptr));
  REQUIRE(st == hipStreamCaptureStatusInvalidated);

  hipGraph_t graph;
  HIP_CHECK_ERROR(hipStreamEndCapture(stream, &graph), hipErrorStreamCaptureInvalidated);

  HIP_CHECK(hipFree(d));
  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Unit_hipStreamSync_SameStream_RelaxedCapture_NonBlockingStream) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  float* d;
  HIP_CHECK(hipMalloc(&d, kDataSize * sizeof(float)));

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
  captureKernel<<<1, kDataSize, 0, stream>>>(d, kDataSize);

  HIP_CHECK_ERROR(hipStreamSynchronize(stream), hipErrorStreamCaptureUnsupported);

  hipStreamCaptureStatus st;
  HIP_CHECK(hipStreamGetCaptureInfo(stream, &st, nullptr));
  REQUIRE(st == hipStreamCaptureStatusInvalidated);

  hipGraph_t graph;
  HIP_CHECK_ERROR(hipStreamEndCapture(stream, &graph), hipErrorStreamCaptureInvalidated);

  HIP_CHECK(hipFree(d));
  HIP_CHECK(hipStreamDestroy(stream));
}

// ============================================================================
// D: hipStreamSynchronize on DIFFERENT stream, SAME thread — capture unaffected
// ============================================================================

static void testSyncDiffStreamSameThread(hipStreamCaptureMode mode, bool cap_nonblocking,
                                         bool sync_nonblocking,
                                         hipStreamCaptureStatus expectedStatus) {
  hipStream_t cap_stream;
  if (cap_nonblocking) {
    HIP_CHECK(hipStreamCreateWithFlags(&cap_stream, hipStreamNonBlocking));
  } else {
    HIP_CHECK(hipStreamCreate(&cap_stream));
  }

  hipStream_t sync_stream;
  if (sync_nonblocking) {
    HIP_CHECK(hipStreamCreateWithFlags(&sync_stream, hipStreamNonBlocking));
  } else {
    HIP_CHECK(hipStreamCreate(&sync_stream));
  }

  float* d;
  HIP_CHECK(hipMalloc(&d, kDataSize * sizeof(float)));

  HIP_CHECK(hipStreamBeginCapture(cap_stream, mode));
  captureKernel<<<1, kDataSize, 0, cap_stream>>>(d, kDataSize);

  (void)hipStreamSynchronize(sync_stream);
  (void)hipGetLastError();

  hipStreamCaptureStatus st;
  HIP_CHECK(hipStreamGetCaptureInfo(cap_stream, &st, nullptr));
  REQUIRE(st == expectedStatus);

  hipGraph_t graph;
  hipError_t endErr = hipStreamEndCapture(cap_stream, &graph);
  if (endErr == hipSuccess && graph != nullptr) {
    HIP_CHECK(hipGraphDestroy(graph));
  }

  HIP_CHECK(hipFree(d));
  HIP_CHECK(hipStreamDestroy(sync_stream));
  HIP_CHECK(hipStreamDestroy(cap_stream));
}

// CUDA semantics for same-thread sync on a different stream:
//   Global/ThreadLocal capture → Invalidated (regardless of stream flags)
//   Relaxed capture → Active

// Blocking capture stream, NonBlocking sync stream
HIP_TEST_CASE(Unit_hipStreamSync_DiffStream_SameThread_GlobalCapture_Blocking_SyncNonBlocking) {
  testSyncDiffStreamSameThread(hipStreamCaptureModeGlobal, false, true, hipStreamCaptureStatusInvalidated);
}

HIP_TEST_CASE(Unit_hipStreamSync_DiffStream_SameThread_ThreadLocalCapture_Blocking_SyncNonBlocking) {
  testSyncDiffStreamSameThread(hipStreamCaptureModeThreadLocal, false, true, hipStreamCaptureStatusInvalidated);
}

HIP_TEST_CASE(Unit_hipStreamSync_DiffStream_SameThread_RelaxedCapture_Blocking_SyncNonBlocking) {
  testSyncDiffStreamSameThread(hipStreamCaptureModeRelaxed, false, true, hipStreamCaptureStatusActive);
}

// NonBlocking capture stream, NonBlocking sync stream
HIP_TEST_CASE(Unit_hipStreamSync_DiffStream_SameThread_GlobalCapture_NonBlocking_SyncNonBlocking) {
  testSyncDiffStreamSameThread(hipStreamCaptureModeGlobal, true, true, hipStreamCaptureStatusInvalidated);
}

HIP_TEST_CASE(Unit_hipStreamSync_DiffStream_SameThread_ThreadLocalCapture_NonBlocking_SyncNonBlocking) {
  testSyncDiffStreamSameThread(hipStreamCaptureModeThreadLocal, true, true, hipStreamCaptureStatusInvalidated);
}

HIP_TEST_CASE(Unit_hipStreamSync_DiffStream_SameThread_RelaxedCapture_NonBlocking_SyncNonBlocking) {
  testSyncDiffStreamSameThread(hipStreamCaptureModeRelaxed, true, true, hipStreamCaptureStatusActive);
}

// Blocking capture stream, Blocking sync stream
HIP_TEST_CASE(Unit_hipStreamSync_DiffStream_SameThread_GlobalCapture_Blocking_SyncBlocking) {
  testSyncDiffStreamSameThread(hipStreamCaptureModeGlobal, false, false, hipStreamCaptureStatusInvalidated);
}

HIP_TEST_CASE(Unit_hipStreamSync_DiffStream_SameThread_ThreadLocalCapture_Blocking_SyncBlocking) {
  testSyncDiffStreamSameThread(hipStreamCaptureModeThreadLocal, false, false, hipStreamCaptureStatusInvalidated);
}

HIP_TEST_CASE(Unit_hipStreamSync_DiffStream_SameThread_RelaxedCapture_Blocking_SyncBlocking) {
  testSyncDiffStreamSameThread(hipStreamCaptureModeRelaxed, false, false, hipStreamCaptureStatusActive);
}

// NonBlocking capture stream, Blocking sync stream
HIP_TEST_CASE(Unit_hipStreamSync_DiffStream_SameThread_GlobalCapture_NonBlocking_SyncBlocking) {
  testSyncDiffStreamSameThread(hipStreamCaptureModeGlobal, true, false, hipStreamCaptureStatusInvalidated);
}

HIP_TEST_CASE(Unit_hipStreamSync_DiffStream_SameThread_ThreadLocalCapture_NonBlocking_SyncBlocking) {
  testSyncDiffStreamSameThread(hipStreamCaptureModeThreadLocal, true, false, hipStreamCaptureStatusInvalidated);
}

HIP_TEST_CASE(Unit_hipStreamSync_DiffStream_SameThread_RelaxedCapture_NonBlocking_SyncBlocking) {
  testSyncDiffStreamSameThread(hipStreamCaptureModeRelaxed, true, false, hipStreamCaptureStatusActive);
}

// ============================================================================
// E: hipStreamSynchronize on DIFFERENT stream, DIFFERENT thread
// ============================================================================

static void testSyncDiffStreamCrossThread(hipStreamCaptureMode mode, bool cap_nonblocking,
                                          bool sync_nonblocking,
                                          hipStreamCaptureStatus expectedStatus) {
  CaptureState state;
  std::thread t(captureThread, mode, cap_nonblocking, std::ref(state));

  while (!state.capture_active) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  hipStream_t sync_stream;
  if (sync_nonblocking) {
    HIP_CHECK(hipStreamCreateWithFlags(&sync_stream, hipStreamNonBlocking));
  } else {
    HIP_CHECK(hipStreamCreate(&sync_stream));
  }

  (void)hipStreamSynchronize(sync_stream);
  (void)hipGetLastError();

  state.api_done = true;
  t.join();

  REQUIRE(state.capture_status == expectedStatus);
  HIP_CHECK(hipStreamDestroy(sync_stream));
}

// CUDA semantics for cross-thread sync on a different stream:
//   Global capture → Invalidated (regardless of stream flags)
//   ThreadLocal/Relaxed capture → Active

HIP_TEST_CASE(Unit_hipStreamSync_DiffStream_CrossThread_GlobalCapture_Blocking_Invalidated) {
  testSyncDiffStreamCrossThread(hipStreamCaptureModeGlobal, false, true,
                                hipStreamCaptureStatusInvalidated);
}

HIP_TEST_CASE(Unit_hipStreamSync_DiffStream_CrossThread_GlobalCapture_NonBlocking_Invalidated) {
  testSyncDiffStreamCrossThread(hipStreamCaptureModeGlobal, true, true,
                                hipStreamCaptureStatusInvalidated);
}

HIP_TEST_CASE(Unit_hipStreamSync_DiffStream_CrossThread_ThreadLocalCapture_Blocking_Active) {
  testSyncDiffStreamCrossThread(hipStreamCaptureModeThreadLocal, false, true,
                                hipStreamCaptureStatusActive);
}

HIP_TEST_CASE(Unit_hipStreamSync_DiffStream_CrossThread_ThreadLocalCapture_NonBlocking_Active) {
  testSyncDiffStreamCrossThread(hipStreamCaptureModeThreadLocal, true, true,
                                hipStreamCaptureStatusActive);
}

HIP_TEST_CASE(Unit_hipStreamSync_DiffStream_CrossThread_RelaxedCapture_Blocking_Active) {
  testSyncDiffStreamCrossThread(hipStreamCaptureModeRelaxed, false, false,
                                hipStreamCaptureStatusActive);
}

HIP_TEST_CASE(Unit_hipStreamSync_DiffStream_CrossThread_RelaxedCapture_NonBlocking_Active) {
  testSyncDiffStreamCrossThread(hipStreamCaptureModeRelaxed, true, false,
                                hipStreamCaptureStatusActive);
}

/** @} */
