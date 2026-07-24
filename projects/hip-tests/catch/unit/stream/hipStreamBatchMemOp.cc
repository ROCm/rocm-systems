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
    if (getAttributeError != hipSuccess) return false;
    if (waitValueSupport == 1) return true;
  }
  return false;
}
/**
 * @addtogroup hipStreamBatchMemOp hipStreamBatchMemOp
 * @{
 * @ingroup StreamTest
 * `hipError_t hipStreamBatchMemOp(hipStream_t stream, unsigned int count,
                               hipStreamBatchMemOpParams* paramArray, unsigned
 int flags);` -
 * Enqueues an array of stream memory operations in the stream.
 */
/**
 * Test Description
 * ------------------------
 * - Verify the Negative cases of the hipStreamBatchMemOp API.
 * Test source
 * ------------------------
 *    - unit/stream/hipStreamBatchMemOp.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
HIP_TEST_CASE(Unit_hipStreamBatchMemOp_Negative_Tests) {
  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));
  REQUIRE(stream != nullptr);
  int totalOps = 2;
  static hipStreamBatchMemOpParams paramArray[2], invalidParamArray[2];
  std::vector<hipDeviceptr_t> opsArray(1);
  HIP_CHECK(hipMalloc((void**)&opsArray[0], sizeof(uint32_t)));

  paramArray[0].operation = hipStreamMemOpWriteValue32;
  paramArray[0].writeValue.address = opsArray[0];
  paramArray[0].writeValue.value = 1000;
  paramArray[0].writeValue.flags = 0x0;
  paramArray[0].writeValue.alias = 0;

  paramArray[1].operation = hipStreamMemOpWaitValue32;
  paramArray[1].waitValue.address = opsArray[0];
  paramArray[1].waitValue.value = 1000;
  paramArray[1].waitValue.flags = hipStreamWaitValueEq;

  invalidParamArray[0].operation = hipStreamMemOpBarrier;
  invalidParamArray[0].writeValue.address = opsArray[0];
  invalidParamArray[0].writeValue.value = 1000;
  invalidParamArray[0].writeValue.flags = 32;
  invalidParamArray[0].writeValue.alias = 0;

  invalidParamArray[1].operation = hipStreamMemOpBarrier;
  invalidParamArray[1].waitValue.address = opsArray[0];
  invalidParamArray[1].waitValue.value = 1000;
  invalidParamArray[1].waitValue.flags = hipStreamWaitValueEq;

  SECTION("Invalid Stream") {
    HIP_CHECK_ERROR(hipStreamBatchMemOp(reinterpret_cast<hipStream_t>(-1), totalOps, paramArray, 0),
                    hipErrorContextIsDestroyed);
  }

  SECTION("Parameter Array as a nullptr") {
    HIP_CHECK_ERROR(hipStreamBatchMemOp(stream, totalOps, nullptr, 0), hipErrorInvalidValue);
  }

  SECTION("More than 256 Total Operations") {
    HIP_CHECK_ERROR(hipStreamBatchMemOp(stream, 1000, paramArray, 0), hipErrorInvalidValue);
  }

  SECTION("Total Operations less than 0") {
    HIP_CHECK_ERROR(hipStreamBatchMemOp(stream, -4, paramArray, 0), hipErrorInvalidValue);
  }
  SECTION("Total Operations Zero") {
    HIP_CHECK_ERROR(hipStreamBatchMemOp(stream, 0, paramArray, 0), hipErrorInvalidValue);
  }

  SECTION("Flag value not Zero") {
    HIP_CHECK_ERROR(hipStreamBatchMemOp(stream, totalOps, paramArray, -6), hipErrorInvalidValue);
  }

// Disabled due to defect SWDEV-502219
#if 0
  SECTION("InValid Parameter Array") {
    HIP_CHECK_ERROR(hipStreamBatchMemOp(stream, totalOps, invalidParamArray, 0),
                    hipErrorInvalidValue);
  }
#endif
  HIP_CHECK(hipFree((void*)opsArray[0]));
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * Test Description
 * ------------------------
 * - Verify that hipStreamBatchMemOp executes ops sequentially in array order.
 *   A write-after-wait dependency in the same batch (Write flag=1, Wait flag==1,
 *   Write flag=0) must not race: the wait must observe the write before the reset.
 *   This validates the globalWorkSize=1 fix in batchMemOps() that prevents
 *   parallel work-items from racing writes against waits in the same batch.
 * Test source
 * ------------------------
 *    - unit/stream/hipStreamBatchMemOp.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
HIP_TEST_CASE(Unit_hipStreamBatchMemOp_SequentialOrdering) {
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

  // Batch: Write(1), Wait(==1), Write(0)
  // If ops run in parallel, Write(0) can race ahead of Wait(==1) causing a deadlock.
  // Sequential execution guarantees Write(1) completes before Wait(==1) is evaluated,
  // and Wait(==1) completes before Write(0) resets the flag.
  // Run many iterations to stress the race — reproduces reliably on both SGPU and MGPU
  // without the globalWorkSize=1 fix.
  hipStreamBatchMemOpParams params[3] = {};
  params[0].operation          = hipStreamMemOpWriteValue32;
  params[0].writeValue.address = devPtr;
  params[0].writeValue.value   = 1;
  params[0].writeValue.flags   = hipStreamWriteValueDefault;

  params[1].operation         = hipStreamMemOpWaitValue32;
  params[1].waitValue.address = devPtr;
  params[1].waitValue.value   = 1;
  params[1].waitValue.flags   = hipStreamWaitValueEq;

  params[2].operation          = hipStreamMemOpWriteValue32;
  params[2].writeValue.address = devPtr;
  params[2].writeValue.value   = 0;
  params[2].writeValue.flags   = hipStreamWriteValueDefault;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  constexpr int kIterations = 100;
  for (int i = 0; i < kIterations; i++) {
    *reinterpret_cast<uint32_t*>(devPtr) = 0;
    HIP_CHECK(hipStreamBatchMemOp(stream, 3, params, 0));
    HIP_CHECK(hipStreamSynchronize(stream));

    uint32_t result = 0;
    HIP_CHECK(hipMemcpy(&result, reinterpret_cast<void*>(devPtr), sizeof(uint32_t),
                        hipMemcpyDeviceToHost));
    REQUIRE(result == 0);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(devPtr)));
  HIP_CHECK(hipCtxPopCurrent(&ctx));
  HIP_CHECK(hipCtxDestroy(ctx));
}

/**
 * Test Description
 * ------------------------
 * - Verify that hipStreamBatchMemOp executes ops sequentially across multiple GPUs.
 *   Each GPU runs the same Write(1)/Wait(==1)/Write(0) batch on its own signal memory.
 *   This mirrors the original RCCL multi-GPU reproducer where the race caused hangs.
 * Test source
 * ------------------------
 *    - unit/stream/hipStreamBatchMemOp.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 *    - At least 2 GPUs
 */
HIP_TEST_CASE(Unit_hipStreamBatchMemOp_SequentialOrdering_MultiGPU) {
  if (!streamWaitValueSupported()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kStreamWaitValueUnsupported);
  }

  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    HIP_SKIP_TEST("Test requires at least 2 GPUs");
  }

  // Fewer iterations than SGPU test — each iteration runs on all GPUs simultaneously
  // so the race is triggered more frequently per iteration.
  constexpr int kIterations = 100;

  std::vector<hipDeviceptr_t> devPtrs(deviceCount, 0);
  std::vector<hipStream_t> streams(deviceCount, nullptr);
  std::vector<hipCtx_t> ctxs(deviceCount, nullptr);

  for (int d = 0; d < deviceCount; d++) {
    HIP_CHECK(hipSetDevice(d));
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, d));
    HIP_CHECK(hipCtxCreate(&ctxs[d], 0, device));
    HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&devPtrs[d]), sizeof(uint64_t),
                                    hipMallocSignalMemory));
    *reinterpret_cast<uint64_t*>(devPtrs[d]) = 0;
    HIP_CHECK(hipStreamCreate(&streams[d]));
  }
  HIP_CHECK(hipDeviceSynchronize());

  for (int i = 0; i < kIterations; i++) {
    for (int d = 0; d < deviceCount; d++) {
      HIP_CHECK(hipSetDevice(d));
      *reinterpret_cast<uint32_t*>(devPtrs[d]) = 0;

      hipStreamBatchMemOpParams params[3] = {};
      params[0].operation          = hipStreamMemOpWriteValue32;
      params[0].writeValue.address = devPtrs[d];
      params[0].writeValue.value   = 1;
      params[0].writeValue.flags   = hipStreamWriteValueDefault;

      params[1].operation         = hipStreamMemOpWaitValue32;
      params[1].waitValue.address = devPtrs[d];
      params[1].waitValue.value   = 1;
      params[1].waitValue.flags   = hipStreamWaitValueEq;

      params[2].operation          = hipStreamMemOpWriteValue32;
      params[2].writeValue.address = devPtrs[d];
      params[2].writeValue.value   = 0;
      params[2].writeValue.flags   = hipStreamWriteValueDefault;

      HIP_CHECK(hipStreamBatchMemOp(streams[d], 3, params, 0));
    }

    for (int d = 0; d < deviceCount; d++) {
      HIP_CHECK(hipSetDevice(d));
      HIP_CHECK(hipStreamSynchronize(streams[d]));
      uint32_t result = 0;
      HIP_CHECK(hipMemcpy(&result, reinterpret_cast<void*>(devPtrs[d]), sizeof(uint32_t),
                          hipMemcpyDeviceToHost));
      REQUIRE(result == 0);
    }
  }

for (int d = deviceCount - 1; d >= 0; d--) {
    HIP_CHECK(hipSetDevice(d));
    HIP_CHECK(hipStreamDestroy(streams[d]));
    HIP_CHECK(hipFree(reinterpret_cast<void*>(devPtrs[d])));
    hipCtx_t popped = nullptr;
    HIP_CHECK(hipCtxPopCurrent(&popped));
    HIP_CHECK(hipCtxDestroy(popped));
  }
}

/**
 * End doxygen group StreamTest.
 * @}
 */
