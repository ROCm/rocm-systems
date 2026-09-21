/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

static bool streamWaitValueSupported() {
  int device_num = 0;
  HIP_CHECK(hipGetDeviceCount(&device_num));
  for (int device_id = 0; device_id < device_num; ++device_id) {
    HIP_CHECK(hipSetDevice(device_id));
    int waitValueSupport = 0;
    auto getAttributeError = hipDeviceGetAttribute(
        &waitValueSupport, hipDeviceAttributeCanUseStreamWaitValue, device_id);
    if (getAttributeError != hipSuccess) {
      return false;
    }
    if (waitValueSupport == 1) return true;
  }
  return false;
}

/**
 * @addtogroup hipStreamWaitValue32 hipStreamWaitValue32
 * @{
 * @ingroup StreamTest
 * `hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, uint32_t value,
 *                                  unsigned int flags, uint32_t mask);`
 * - Enqueues a wait on a memory location in the stream.
 */

/**
 * Test Description
 * ------------------------
 * - Verify that hipStreamWriteValue32 followed by hipStreamWaitValue32 are captured
 *   as BatchMemOp graph nodes during stream capture and produce the correct result
 *   after hipGraphLaunch.
 * Test source
 * ------------------------
 *    - unit/stream/hipStreamValueCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
HIP_TEST_CASE(Unit_hipStreamWaitWriteValue32_Capture) {
  if (!streamWaitValueSupported()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kStreamWaitValueUnsupported);
  }

  hipDeviceptr_t devPtr = 0;
  HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&devPtr), sizeof(uint64_t),
                                  hipMallocSignalMemory));
  *reinterpret_cast<uint64_t*>(devPtr) = 0;
  HIP_CHECK(hipDeviceSynchronize());

  hipStream_t captureStream;
  HIP_CHECK(hipStreamCreate(&captureStream));

  // Capture: write 0x42, then wait for 0x42
  HIP_CHECK(hipStreamBeginCapture(captureStream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipStreamWriteValue32(captureStream, reinterpret_cast<void*>(devPtr), 0x42,
                                  hipStreamWriteValueDefault));
  HIP_CHECK(hipStreamWaitValue32(captureStream, reinterpret_cast<void*>(devPtr), 0x42,
                                 hipStreamWaitValueEq, 0xFFFFFFFF));
  hipGraph_t graph;
  HIP_CHECK(hipStreamEndCapture(captureStream, &graph));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  hipStream_t launchStream;
  HIP_CHECK(hipStreamCreate(&launchStream));
  HIP_CHECK(hipGraphLaunch(graphExec, launchStream));
  HIP_CHECK(hipStreamSynchronize(launchStream));

  uint32_t result = 0;
  HIP_CHECK(hipMemcpy(&result, reinterpret_cast<void*>(devPtr), sizeof(uint32_t),
                      hipMemcpyDeviceToHost));
  REQUIRE(result == 0x42);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(launchStream));
  HIP_CHECK(hipStreamDestroy(captureStream));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(devPtr)));
}

/**
 * Test Description
 * ------------------------
 * - Verify that hipStreamWriteValue64 followed by hipStreamWaitValue64 are captured
 *   as BatchMemOp graph nodes during stream capture and produce the correct result
 *   after hipGraphLaunch.
 * Test source
 * ------------------------
 *    - unit/stream/hipStreamValueCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
HIP_TEST_CASE(Unit_hipStreamWaitWriteValue64_Capture) {
  if (!streamWaitValueSupported()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kStreamWaitValueUnsupported);
  }

  hipDeviceptr_t devPtr = 0;
  HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&devPtr), sizeof(uint64_t),
                                  hipMallocSignalMemory));
  *reinterpret_cast<uint64_t*>(devPtr) = 0;
  HIP_CHECK(hipDeviceSynchronize());

  hipStream_t captureStream;
  HIP_CHECK(hipStreamCreate(&captureStream));

  HIP_CHECK(hipStreamBeginCapture(captureStream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipStreamWriteValue64(captureStream, reinterpret_cast<void*>(devPtr),
                                  0xCAFEBABECAFEBABEULL, hipStreamWriteValueDefault));
  HIP_CHECK(hipStreamWaitValue64(captureStream, reinterpret_cast<void*>(devPtr),
                                 0xCAFEBABECAFEBABEULL, hipStreamWaitValueEq,
                                 0xFFFFFFFFFFFFFFFFULL));
  hipGraph_t graph;
  HIP_CHECK(hipStreamEndCapture(captureStream, &graph));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  hipStream_t launchStream;
  HIP_CHECK(hipStreamCreate(&launchStream));
  HIP_CHECK(hipGraphLaunch(graphExec, launchStream));
  HIP_CHECK(hipStreamSynchronize(launchStream));

  uint64_t result = 0;
  HIP_CHECK(hipMemcpy(&result, reinterpret_cast<void*>(devPtr), sizeof(uint64_t),
                      hipMemcpyDeviceToHost));
  REQUIRE(result == 0xCAFEBABECAFEBABEULL);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(launchStream));
  HIP_CHECK(hipStreamDestroy(captureStream));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(devPtr)));
}

/**
 * Test Description
 * ------------------------
 * - Verify that hipStreamBatchMemOp (WriteValue32 + WaitValue32) is captured
 *   as a BatchMemOpNode during stream capture and produces the correct result
 *   after hipGraphLaunch, with ops executing sequentially in array order.
 * Test source
 * ------------------------
 *    - unit/stream/hipStreamValueCapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
HIP_TEST_CASE(Unit_hipStreamBatchMemOp_Capture) {
  if (!streamWaitValueSupported()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kStreamWaitValueUnsupported);
  }

  hipCtx_t ctx;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&ctx, 0, device));

  hipDeviceptr_t devPtr = 0;
  HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&devPtr), sizeof(uint64_t),
                                  hipMallocSignalMemory));
  *reinterpret_cast<uint64_t*>(devPtr) = 0;
  HIP_CHECK(hipDeviceSynchronize());

  // Batch: WriteValue32(0x1234) then WaitValue32(== 0x1234)
  hipStreamBatchMemOpParams params[2] = {};
  params[0].operation          = hipStreamMemOpWriteValue32;
  params[0].writeValue.address = devPtr;
  params[0].writeValue.value   = 0x1234;
  params[0].writeValue.flags   = hipStreamWriteValueDefault;

  params[1].operation         = hipStreamMemOpWaitValue32;
  params[1].waitValue.address = devPtr;
  params[1].waitValue.value   = 0x1234;
  params[1].waitValue.flags   = hipStreamWaitValueEq;

  hipStream_t captureStream;
  HIP_CHECK(hipStreamCreate(&captureStream));

  HIP_CHECK(hipStreamBeginCapture(captureStream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipStreamBatchMemOp(captureStream, 2, params, 0));
  hipGraph_t graph;
  HIP_CHECK(hipStreamEndCapture(captureStream, &graph));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  hipStream_t launchStream;
  HIP_CHECK(hipStreamCreate(&launchStream));
  HIP_CHECK(hipGraphLaunch(graphExec, launchStream));
  HIP_CHECK(hipStreamSynchronize(launchStream));

  uint32_t result = 0;
  HIP_CHECK(hipMemcpy(&result, reinterpret_cast<void*>(devPtr), sizeof(uint32_t),
                      hipMemcpyDeviceToHost));
  REQUIRE(result == 0x1234);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(launchStream));
  HIP_CHECK(hipStreamDestroy(captureStream));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(devPtr)));
  HIP_CHECK(hipCtxPopCurrent(&ctx));
  HIP_CHECK(hipCtxDestroy(ctx));
}

/**
 * End doxygen group StreamTest.
 * @}
 */
