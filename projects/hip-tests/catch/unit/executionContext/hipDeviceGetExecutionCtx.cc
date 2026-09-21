/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipDeviceGetExecutionCtx hipDeviceGetExecutionCtx
 * @{
 * @ingroup ExecutionContextTest
 * `hipDeviceGetExecutionCtx`, `hipExecutionCtxGetDevice`, and
 * `hipExecutionCtxGetId` APIs
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include "hip_executionctx_common.hh"

#include <cstdlib>
#include <vector>

/**
 * Test Description
 * ------------------------
 *  - Retrieves the primary execution context for device 0 via
 *    hipDeviceGetExecutionCtx and verifies the returned context is non-null.
 */
HIP_TEST_CASE(Unit_hipDeviceGetExecutionCtx_Sanity) {
  HIP_CHECK(hipSetDevice(0));

  hipExecutionCtx_t ctx = nullptr;
  HIP_CHECK(hipDeviceGetExecutionCtx(&ctx, 0));
  REQUIRE(ctx != nullptr);
}

/**
 * Test Description
 * ------------------------
 *  - Calls hipDeviceGetExecutionCtx twice for the same device and verifies
 *    that the same primary context is returned both times.
 */
HIP_TEST_CASE(Unit_hipDeviceGetExecutionCtx_Consistent) {
  HIP_CHECK(hipSetDevice(0));

  hipExecutionCtx_t ctx1 = nullptr;
  hipExecutionCtx_t ctx2 = nullptr;
  HIP_CHECK(hipDeviceGetExecutionCtx(&ctx1, 0));
  HIP_CHECK(hipDeviceGetExecutionCtx(&ctx2, 0));
  REQUIRE(ctx1 == ctx2);
}

/**
 * Test Description
 * ------------------------
 *  - Retrieves the primary execution context for each available device and
 *    verifies each is non-null and that contexts for different devices differ.
 */
HIP_TEST_CASE(Unit_hipDeviceGetExecutionCtx_MultiDevice) {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));

  std::vector<hipExecutionCtx_t> contexts(deviceCount, nullptr);
  for (int dev = 0; dev < deviceCount; dev++) {
    HIP_CHECK(hipDeviceGetExecutionCtx(&contexts[dev], dev));
    REQUIRE(contexts[dev] != nullptr);
  }

  for (int i = 0; i < deviceCount; i++) {
    for (int j = i + 1; j < deviceCount; j++) {
      REQUIRE(contexts[i] != contexts[j]);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Retrieves the primary execution context for each device, then calls
 *    hipExecutionCtxGetDevice to verify the returned device ID matches.
 */
HIP_TEST_CASE(Unit_hipDeviceGetExecutionCtx_GetDevice_RoundTrip) {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));

  for (int dev = 0; dev < deviceCount; dev++) {
    hipExecutionCtx_t ctx = nullptr;
    HIP_CHECK(hipDeviceGetExecutionCtx(&ctx, dev));
    REQUIRE(ctx != nullptr);

    int returnedDev = -1;
    HIP_CHECK(hipExecutionCtxGetDevice(&returnedDev, ctx));
    REQUIRE(returnedDev == dev);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Retrieves the primary execution context and calls hipExecutionCtxGetId
 *    to verify a non-zero context ID is returned.  Also verifies that different
 *    devices produce different context IDs.
 */
HIP_TEST_CASE(Unit_hipDeviceGetExecutionCtx_GetId) {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));

  std::vector<unsigned long long> ids(deviceCount, 0);
  for (int dev = 0; dev < deviceCount; dev++) {
    hipExecutionCtx_t ctx = nullptr;
    HIP_CHECK(hipDeviceGetExecutionCtx(&ctx, dev));

    HIP_CHECK(hipExecutionCtxGetId(ctx, &ids[dev]));
    REQUIRE(ids[dev] != 0);
  }

  for (int i = 0; i < deviceCount; i++) {
    for (int j = i + 1; j < deviceCount; j++) {
      REQUIRE(ids[i] != ids[j]);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates error codes from hipDeviceGetExecutionCtx, hipExecutionCtxGetDevice,
 *    and hipExecutionCtxGetId for invalid parameters.
 */
HIP_TEST_CASE(Unit_hipDeviceGetExecutionCtx_Negative) {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));

  // NULL ctx pointer
  REQUIRE(hipDeviceGetExecutionCtx(nullptr, 0) == hipErrorInvalidValue);

  // Invalid device (negative)
  hipExecutionCtx_t ctx = nullptr;
  REQUIRE(hipDeviceGetExecutionCtx(&ctx, -1) == hipErrorInvalidDevice);

  // Invalid device (out of range)
  REQUIRE(hipDeviceGetExecutionCtx(&ctx, deviceCount) == hipErrorInvalidDevice);

  // hipExecutionCtxGetDevice: NULL device pointer
  HIP_CHECK(hipDeviceGetExecutionCtx(&ctx, 0));
  REQUIRE(hipExecutionCtxGetDevice(nullptr, ctx) == hipErrorInvalidValue);

  // hipExecutionCtxGetDevice: NULL ctx
  int dev = -1;
  REQUIRE(hipExecutionCtxGetDevice(&dev, nullptr) == hipErrorInvalidValue);

  // hipExecutionCtxGetId: NULL ctx
  unsigned long long id = 0;
  REQUIRE(hipExecutionCtxGetId(nullptr, &id) == hipErrorInvalidValue);

  // hipExecutionCtxGetId: NULL id pointer
  REQUIRE(hipExecutionCtxGetId(ctx, nullptr) == hipErrorInvalidValue);
}

/**
 * Test Description
 * ------------------------
 *  - Retrieves the primary execution context and creates a green context from
 *    a split SM resource.  Creates a stream from each context, launches a
 *    vectorADD kernel on both streams concurrently, synchronizes, and verifies
 *    that both produce identical correct results.  The primary context must NOT
 *    be destroyed; only the green context is cleaned up.
 */
HIP_TEST_CASE(Unit_hipDeviceGetExecutionCtx_KernelLaunch_Functional) {
  HIP_CHECK(hipSetDevice(0));

  // Get the primary execution context
  hipExecutionCtx_t primaryCtx = nullptr;
  HIP_CHECK(hipDeviceGetExecutionCtx(&primaryCtx, 0));
  REQUIRE(primaryCtx != nullptr);

  // Create a stream from the primary context
  hipStream_t primaryStream = nullptr;
  HIP_CHECK(hipExecutionCtxStreamCreate(&primaryStream, primaryCtx, hipStreamNonBlocking, 0));
  REQUIRE(primaryStream != nullptr);

  // Create a green context from a split SM resource
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));

  hipDevResource input{};
  HIP_CHECK(hipDeviceGetDevResource(device, &input, hipDevResourceTypeSm));

  unsigned int alignment = input.sm.smCoscheduledAlignment;
  unsigned int groupSize = (input.sm.smCount / 2 / alignment) * alignment;
  REQUIRE(groupSize >= alignment);

  hipDevSmResourceGroupParams params[1] = {};
  params[0].smCount = groupSize;

  hipDevResource splitResult[1] = {};
  hipDevResource remainder{};
  HIP_CHECK(hipDevSmResourceSplit(splitResult, 1, &input, &remainder, 0, params));

  hipDevResourceDesc_t desc{};
  HIP_CHECK(hipDevResourceGenerateDesc(&desc, splitResult, 1));

  hipExecutionCtx_t greenCtx = nullptr;
  HIP_CHECK(hipGreenCtxCreate(&greenCtx, desc, 0, 0));
  REQUIRE(greenCtx != nullptr);

  // Create a stream from the green context
  hipStream_t greenStream = nullptr;
  HIP_CHECK(hipExecutionCtxStreamCreate(&greenStream, greenCtx, hipStreamNonBlocking, 0));
  REQUIRE(greenStream != nullptr);

  // Verify both contexts return valid device IDs
  int primaryDev = -1, greenDev = -1;
  HIP_CHECK(hipExecutionCtxGetDevice(&primaryDev, primaryCtx));
  HIP_CHECK(hipExecutionCtxGetDevice(&greenDev, greenCtx));
  REQUIRE(primaryDev == 0);
  REQUIRE(greenDev == 0);

  // Verify contexts have different IDs
  unsigned long long primaryId = 0, greenId = 0;
  HIP_CHECK(hipExecutionCtxGetId(primaryCtx, &primaryId));
  HIP_CHECK(hipExecutionCtxGetId(greenCtx, &greenId));
  REQUIRE(primaryId != greenId);

  // Prepare data for vectorADD on both streams
  constexpr size_t kN = 1024;
  const size_t kBytes = kN * sizeof(int);

  int* h_a = reinterpret_cast<int*>(malloc(kBytes));
  int* h_b = reinterpret_cast<int*>(malloc(kBytes));
  int* h_c_primary = reinterpret_cast<int*>(malloc(kBytes));
  int* h_c_green = reinterpret_cast<int*>(malloc(kBytes));
  REQUIRE(h_a != nullptr);
  REQUIRE(h_b != nullptr);
  REQUIRE(h_c_primary != nullptr);
  REQUIRE(h_c_green != nullptr);

  for (size_t i = 0; i < kN; i++) {
    h_a[i] = static_cast<int>(i);
    h_b[i] = static_cast<int>(i * 2);
  }

  // Allocate device memory for both streams (separate output buffers)
  int *d_a = nullptr, *d_b = nullptr;
  int *d_c_primary = nullptr, *d_c_green = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kBytes));
  HIP_CHECK(hipMalloc(&d_b, kBytes));
  HIP_CHECK(hipMalloc(&d_c_primary, kBytes));
  HIP_CHECK(hipMalloc(&d_c_green, kBytes));

  // Launch on primary stream
  HIP_CHECK(hipMemcpyAsync(d_a, h_a, kBytes, hipMemcpyHostToDevice, primaryStream));
  HIP_CHECK(hipMemcpyAsync(d_b, h_b, kBytes, hipMemcpyHostToDevice, primaryStream));

  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((kN + kThreads - 1) / kThreads);
  HipTest::vectorADD<<<blocks, kThreads, 0, primaryStream>>>(d_a, d_b, d_c_primary, kN);
  HIP_CHECK(hipGetLastError());

  // Launch on green stream (reuse same input buffers after primary stream sync)
  HIP_CHECK(hipStreamSynchronize(primaryStream));
  HIP_CHECK(hipMemcpyAsync(d_a, h_a, kBytes, hipMemcpyHostToDevice, greenStream));
  HIP_CHECK(hipMemcpyAsync(d_b, h_b, kBytes, hipMemcpyHostToDevice, greenStream));
  HipTest::vectorADD<<<blocks, kThreads, 0, greenStream>>>(d_a, d_b, d_c_green, kN);
  HIP_CHECK(hipGetLastError());

  // Copy results back
  HIP_CHECK(hipMemcpyAsync(h_c_primary, d_c_primary, kBytes, hipMemcpyDeviceToHost, primaryStream));
  HIP_CHECK(hipMemcpyAsync(h_c_green, d_c_green, kBytes, hipMemcpyDeviceToHost, greenStream));
  HIP_CHECK(hipStreamSynchronize(primaryStream));
  HIP_CHECK(hipStreamSynchronize(greenStream));

  // Verify both produce correct and identical results
  for (size_t i = 0; i < kN; i++) {
    REQUIRE(h_c_primary[i] == h_a[i] + h_b[i]);
    REQUIRE(h_c_green[i] == h_a[i] + h_b[i]);
    REQUIRE(h_c_primary[i] == h_c_green[i]);
  }

  // Cleanup
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c_primary));
  HIP_CHECK(hipFree(d_c_green));
  free(h_a);
  free(h_b);
  free(h_c_primary);
  free(h_c_green);

  HIP_CHECK(hipStreamDestroy(greenStream));
  HIP_CHECK(hipStreamDestroy(primaryStream));
  HIP_CHECK(hipExecutionCtxDestroy(greenCtx));
  // Primary context must NOT be destroyed
}

/**
 * Test Description
 * ------------------------
 *  - Simulates two independent application modules that each obtain the primary
 *    execution context, create their own stream, and launch work.  Module A
 *    produces partial results, Module B (getting the same context later) launches
 *    dependent work using event synchronization.  Verifies that:
 *    1. Both modules get the same primary context handle.
 *    2. Streams from separate acquisitions of the primary context work correctly.
 *    3. Event-based synchronization across streams from the same context works.
 *    4. The primary context remains valid throughout, demonstrating its
 *       application-lifetime persistence.
 *    5. Attempting to destroy the primary context returns hipErrorInvalidValue.
 */
HIP_TEST_CASE(Unit_hipDeviceGetExecutionCtx_PrimaryCtx_Persistence) {
  HIP_CHECK(hipSetDevice(0));

  // --- Module A: get primary context, create stream, launch work ---
  hipExecutionCtx_t ctxA = nullptr;
  HIP_CHECK(hipDeviceGetExecutionCtx(&ctxA, 0));
  REQUIRE(ctxA != nullptr);

  hipStream_t streamA = nullptr;
  HIP_CHECK(hipExecutionCtxStreamCreate(&streamA, ctxA, hipStreamNonBlocking, 0));
  REQUIRE(streamA != nullptr);

  constexpr size_t kN = 1024;
  const size_t kBytes = kN * sizeof(int);

  int* h_a = reinterpret_cast<int*>(malloc(kBytes));
  int* h_b = reinterpret_cast<int*>(malloc(kBytes));
  int* h_c = reinterpret_cast<int*>(malloc(kBytes));
  REQUIRE(h_a != nullptr);
  REQUIRE(h_b != nullptr);
  REQUIRE(h_c != nullptr);

  for (size_t i = 0; i < kN; i++) {
    h_a[i] = static_cast<int>(i);
    h_b[i] = static_cast<int>(i * 2);
  }

  int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kBytes));
  HIP_CHECK(hipMalloc(&d_b, kBytes));
  HIP_CHECK(hipMalloc(&d_c, kBytes));

  HIP_CHECK(hipMemcpyAsync(d_a, h_a, kBytes, hipMemcpyHostToDevice, streamA));
  HIP_CHECK(hipMemcpyAsync(d_b, h_b, kBytes, hipMemcpyHostToDevice, streamA));

  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((kN + kThreads - 1) / kThreads);
  HipTest::vectorADD<<<blocks, kThreads, 0, streamA>>>(d_a, d_b, d_c, kN);
  HIP_CHECK(hipGetLastError());

  hipEvent_t event = nullptr;
  HIP_CHECK(hipEventCreate(&event));
  HIP_CHECK(hipEventRecord(event, streamA));

  // --- Module B: independently get the same primary context ---
  hipExecutionCtx_t ctxB = nullptr;
  HIP_CHECK(hipDeviceGetExecutionCtx(&ctxB, 0));
  REQUIRE(ctxB == ctxA);

  hipStream_t streamB = nullptr;
  HIP_CHECK(hipExecutionCtxStreamCreate(&streamB, ctxB, hipStreamNonBlocking, 0));
  REQUIRE(streamB != nullptr);

  // Wait for Module A's work to complete before reading its output
  HIP_CHECK(hipStreamWaitEvent(streamB, event, 0));

  // Module B reads the result produced by Module A
  HIP_CHECK(hipMemcpyAsync(h_c, d_c, kBytes, hipMemcpyDeviceToHost, streamB));
  HIP_CHECK(hipStreamSynchronize(streamB));

  for (size_t i = 0; i < kN; i++) {
    REQUIRE(h_c[i] == h_a[i] + h_b[i]);
  }

  // Verify primary context cannot be destroyed
  REQUIRE(hipExecutionCtxDestroy(ctxA) == hipErrorInvalidValue);

  // Cleanup
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));
  free(h_a);
  free(h_b);
  free(h_c);
  HIP_CHECK(hipStreamDestroy(streamB));
  HIP_CHECK(hipStreamDestroy(streamA));
}

/**
 * End doxygen group hipDeviceGetExecutionCtx.
 * @}
 */
