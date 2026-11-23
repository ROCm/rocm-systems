/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <utils.hh>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

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
 * Test Description
 * ------------------------
 * - Verify that hipStreamBatchMemOp writes 64-bit and 32-bit values and waits for them, and that
 *   the runtime doesn't read the parameter array after the call returns. The stream first waits
 *   for an event of another stream whose kernel spins until the host releases it, and the host
 *   clears the array before the release. A runtime that submits from a worker thread (Windows)
 *   then can't see the array before it is cleared.
 * Test source
 * ------------------------
 *    - unit/stream/hipStreamBatchMemOp.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipStreamBatchMemOp_WriteAndWaitValues) {
  if (!streamWaitValueSupported()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kStreamWaitValueUnsupported);
  }
  HIP_CHECK(hipSetDevice(0));

  void* sig64 = nullptr;
  void* sig32 = nullptr;
  HIP_CHECK(hipExtMallocWithFlags(&sig64, sizeof(uint64_t), hipMallocSignalMemory));
  HIP_CHECK(hipExtMallocWithFlags(&sig32, sizeof(uint64_t), hipMallocSignalMemory));
  *reinterpret_cast<uint64_t*>(sig64) = 0;
  *reinterpret_cast<uint64_t*>(sig32) = 0;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  constexpr uint64_t kValue64 = 0x123456789ABCDEF0ULL;
  constexpr uint32_t kValue32 = 0xCAFE;
  hipStreamBatchMemOpParams params[4] = {};
  params[0].operation = hipStreamMemOpWriteValue64;
  params[0].writeValue.address = sig64;
  params[0].writeValue.value64 = kValue64;
  params[1].operation = hipStreamMemOpWaitValue64;
  params[1].waitValue.address = sig64;
  params[1].waitValue.value64 = kValue64;
  params[1].waitValue.flags = hipStreamWaitValueEq;
  params[2].operation = hipStreamMemOpWriteValue32;
  params[2].writeValue.address = sig32;
  params[2].writeValue.value = kValue32;
  params[3].operation = hipStreamMemOpWaitValue32;
  params[3].waitValue.address = sig32;
  params[3].waitValue.value = kValue32;
  params[3].waitValue.flags = hipStreamWaitValueEq;

  // Hold the stream until the host releases it: it waits for an event recorded behind a kernel
  // that spins on a host flag (with a 1 s timeout, below the Windows GPU timeout, so a failure
  // doesn't hang the GPU)
  hipStream_t blocker;
  HIP_CHECK(hipStreamCreateWithFlags(&blocker, hipStreamNonBlocking));
  hipEvent_t gate;
  HIP_CHECK(hipEventCreateWithFlags(&gate, hipEventDisableTiming));
  int* release = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&release), sizeof(int), hipHostMallocDefault));
  *release = 0;
  WaitForHostReleaseWithTimeout<<<1, 1, 0, blocker>>>(release, SpinTicksPerUs() * 1000000LL);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipEventRecord(gate, blocker));
  HIP_CHECK(hipStreamWaitEvent(stream, gate, 0));

  const hipError_t batchError = hipStreamBatchMemOp(stream, 4, params, 0);
  // Operation 0 is not a valid operation, so a batch that read the array now would do nothing
  std::memset(params, 0, sizeof(params));
  *reinterpret_cast<volatile int*>(release) = 1;
  HIP_CHECK(batchError);
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamSynchronize(blocker));

  uint64_t result64 = 0;
  uint32_t result32 = 0;
  HIP_CHECK(hipMemcpy(&result64, sig64, sizeof(result64), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&result32, sig32, sizeof(result32), hipMemcpyDeviceToHost));
  REQUIRE(result64 == kValue64);
  REQUIRE(result32 == kValue32);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipStreamDestroy(blocker));
  HIP_CHECK(hipEventDestroy(gate));
  HIP_CHECK(hipHostFree(release));
  HIP_CHECK(hipFree(sig64));
  HIP_CHECK(hipFree(sig32));
}

/**
 * Test Description
 * ------------------------
 * - Verify that a wait operation in hipStreamBatchMemOp holds back the operations after it in the
 *   batch and the work queued after the batch until the host stores the awaited value.
 * Test source
 * ------------------------
 *    - unit/stream/hipStreamBatchMemOp.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipStreamBatchMemOp_WaitHoldsBackLaterWork) {
  if (!streamWaitValueSupported()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kStreamWaitValueUnsupported);
  }
  HIP_CHECK(hipSetDevice(0));

  void* flag = nullptr;
  void* data = nullptr;
  HIP_CHECK(hipExtMallocWithFlags(&flag, sizeof(uint64_t), hipMallocSignalMemory));
  HIP_CHECK(hipExtMallocWithFlags(&data, sizeof(uint64_t), hipMallocSignalMemory));
  volatile uint64_t* hostFlag = reinterpret_cast<volatile uint64_t*>(flag);
  volatile uint64_t* hostData = reinterpret_cast<volatile uint64_t*>(data);
  *hostFlag = 0;
  *hostData = 0;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Batch: Wait(flag == 1), Write(data = 2). Then a scalar Write(data = 3) after the batch.
  hipStreamBatchMemOpParams params[2] = {};
  params[0].operation = hipStreamMemOpWaitValue64;
  params[0].waitValue.address = flag;
  params[0].waitValue.value64 = 1;
  params[0].waitValue.flags = hipStreamWaitValueEq;
  params[1].operation = hipStreamMemOpWriteValue64;
  params[1].writeValue.address = data;
  params[1].writeValue.value64 = 2;
  HIP_CHECK(hipStreamBatchMemOp(stream, 2, params, 0));
  HIP_CHECK(hipStreamWriteValue64(stream, data, 3, 0));

  // Give the stream about 100 ms to run whatever is not held back. Bound it by time, not by a
  // count of sleeps: a 1 ms sleep can last a 15.6 ms timer tick on Windows, and the wait kernel
  // spins on the GPU for as long as the host holds it.
  const auto holdStart = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - holdStart < std::chrono::milliseconds(100) &&
         hipStreamQuery(stream) == hipErrorNotReady) {
    std::this_thread::yield();
  }
  const hipError_t queryBeforeRelease = hipStreamQuery(stream);
  const uint64_t dataBeforeRelease = *hostData;
  // Release the wait before any check, so a failed check doesn't leave the stream blocked
  *hostFlag = 1;
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(queryBeforeRelease == hipErrorNotReady);
  REQUIRE(dataBeforeRelease == 0);
  REQUIRE(*hostData == 3);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(flag));
  HIP_CHECK(hipFree(data));
}

/**
 * Test Description
 * ------------------------
 * - Verify that a value written by hipStreamBatchMemOp or hipStreamWriteValue32 is ordered after
 *   an earlier copy on the same stream that touches pinned host memory (such copies may run on a
 *   DMA engine). Signal: a D2H copy into pinned memory, then the write of a flag in other host
 *   memory; once the host sees the flag, the copied data must be there. Overwrite: an H2D copy,
 *   then a batch write into the last word of the copy's destination; the write must win.
 *   Before the copy, a kernel that spins until the host releases it holds the stream, and the
 *   host queries an event recorded behind that kernel after queuing the copy. On Windows (PAL)
 *   the runtime then flushed the copy and forgot its SDMA event, which let the write overtake it
 *   (rocm-systems#12952 keeps the pending event).
 * Test source
 * ------------------------
 *    - unit/stream/hipStreamBatchMemOp.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipStreamBatchMemOp_WriteOrderedAfterCopy) {
  if (!streamWaitValueSupported()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kStreamWaitValueUnsupported);
  }
  HIP_CHECK(hipSetDevice(0));

  enum class Mode { kBatchSignal, kScalarSignal, kBatchOverwrite };
  Mode mode = Mode::kBatchSignal;
  const char* name = "";
  SECTION("batch write signals a D2H copy") {
    mode = Mode::kBatchSignal;
    name = "batch write signals a D2H copy";
  }
  SECTION("hipStreamWriteValue32 signals a D2H copy") {
    mode = Mode::kScalarSignal;
    name = "hipStreamWriteValue32 signals a D2H copy";
  }
  SECTION("batch write into the destination of an H2D copy") {
    mode = Mode::kBatchOverwrite;
    name = "batch write into the destination of an H2D copy";
  }

  constexpr size_t kSize = 64 << 20;
  constexpr size_t kCount = kSize / sizeof(uint32_t);
  uint32_t* dev = nullptr;
  uint32_t* host = nullptr;
  uint32_t* flag = nullptr;
  int* release = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dev), kSize));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&host), kSize, hipHostMallocDefault));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&flag), sizeof(uint32_t), hipHostMallocDefault));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&release), sizeof(int), hipHostMallocDefault));
  volatile uint32_t* hostFlag = flag;
  volatile uint32_t* hostData = host;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  hipEvent_t held, done;
  HIP_CHECK(hipEventCreateWithFlags(&held, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&done, hipEventDisableTiming));

  constexpr int kIterations = 10;
  int staleSignals = 0;
  int lostWrites = 0;
  int flagTimeouts = 0;
  for (int i = 1; i <= kIterations; ++i) {
    const uint32_t copied = 0xC0DE0000u + i;
    const uint32_t old = 0x01D00000u + i;
    const uint32_t mark = 0x5A5A0000u + i;
    const bool overwrite = mode == Mode::kBatchOverwrite;
    if (overwrite) {
      std::fill(host, host + kCount, copied);
      HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(dev), old, kCount));
    } else {
      HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(dev), copied, kCount));
      std::fill(host, host + kCount, old);
    }
    HIP_CHECK(hipDeviceSynchronize());
    *hostFlag = 0;
    *release = 0;

    // 1 s fallback, below the Windows GPU timeout; the host normally releases it within ms
    WaitForHostReleaseWithTimeout<<<1, 1, 0, stream>>>(release, SpinTicksPerUs() * 1000000LL);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(held, stream));
    if (overwrite) {
      HIP_CHECK(hipMemcpyAsync(dev, host, kSize, hipMemcpyHostToDevice, stream));
    } else {
      HIP_CHECK(hipMemcpyAsync(host, dev, kSize, hipMemcpyDeviceToHost, stream));
    }
    HIP_CHECK_ERROR(hipEventQuery(held), hipErrorNotReady);
    hipStreamBatchMemOpParams op = {};
    op.operation = hipStreamMemOpWriteValue32;
    op.writeValue.address = overwrite ? static_cast<void*>(dev + kCount - 1) : flag;
    op.writeValue.value = overwrite ? mark : i;
    if (mode == Mode::kScalarSignal) {
      HIP_CHECK(hipStreamWriteValue32(stream, flag, i, 0));
    } else {
      HIP_CHECK(hipStreamBatchMemOp(stream, 1, &op, 0));
    }
    // The event makes a runtime that batches commands submit the write
    HIP_CHECK(hipEventRecord(done, stream));
    *reinterpret_cast<volatile int*>(release) = 1;

    if (overwrite) {
      HIP_CHECK(hipStreamSynchronize(stream));
      uint32_t last = 0;
      HIP_CHECK(hipMemcpy(&last, dev + kCount - 1, sizeof(last), hipMemcpyDeviceToHost));
      lostWrites += last != mark;
    } else {
      // Read the copy's first and last word as soon as the flag is set
      const auto start = std::chrono::steady_clock::now();
      while (*hostFlag != static_cast<uint32_t>(i) &&
             std::chrono::steady_clock::now() - start < std::chrono::seconds(2)) {
      }
      std::atomic_thread_fence(std::memory_order_acquire);
      const bool flagSet = *hostFlag == static_cast<uint32_t>(i);
      const uint32_t first = hostData[0];
      const uint32_t last = hostData[kCount - 1];
      HIP_CHECK(hipStreamSynchronize(stream));
      flagTimeouts += !flagSet;
      staleSignals += flagSet && (first != copied || last != copied);
      if (!flagSet) break;  // the write doesn't happen at all; don't wait for it again
    }
  }

  HIP_CHECK(hipEventDestroy(held));
  HIP_CHECK(hipEventDestroy(done));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipHostFree(release));
  HIP_CHECK(hipHostFree(flag));
  HIP_CHECK(hipHostFree(host));
  HIP_CHECK(hipFree(dev));
  INFO(name);
  REQUIRE(flagTimeouts == 0);
  REQUIRE(staleSignals == 0);
  REQUIRE(lostWrites == 0);
}

/**
 * End doxygen group StreamTest.
 * @}
 */
