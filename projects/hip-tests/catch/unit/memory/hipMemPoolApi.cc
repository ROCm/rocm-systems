/*
   Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

/* Test Case Description:
   1) This testcase verifies the  basic scenario - supported on
     all devices
*/
#include "mempool_common.hh"
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <thread>
#include <chrono>

static hipMemPoolProps kPoolProps;
void initMemPoolProps() {
  kPoolProps.allocType = hipMemAllocationTypePinned;
  kPoolProps.handleTypes = hipMemHandleTypeNone;
  kPoolProps.location.type = hipMemLocationTypeDevice;
  kPoolProps.location.id = 0;
  kPoolProps.win32SecurityAttributes = nullptr;
};
/*
   This testcase verifies HIP Mem Pool API basic scenario - supported on all devices
 */

TEST_CASE("Unit_hipMemPoolApi_Basic") {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  int numElements = 64 * 1024 * 1024;
  float *A = nullptr, *B = nullptr;

  hipMemPool_t mem_pool = nullptr;
  int device = 0;
  HIP_CHECK(hipDeviceGetDefaultMemPool(&mem_pool, device));
  HIP_CHECK(hipDeviceSetMemPool(device, mem_pool));
  HIP_CHECK(hipDeviceGetMemPool(&mem_pool, device));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&A), numElements * sizeof(float), stream));
  INFO("hipMallocAsync result: " << A);

  HIP_CHECK(hipFreeAsync(A, stream));
  // Reset the default memory pool usage for the following tests
  hipMemPoolAttr attr = hipMemPoolAttrUsedMemHigh;
  std::uint64_t value64 = 0;
  HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &value64));

  size_t min_bytes_to_hold = 1024 * 1024;
  HIP_CHECK(hipMemPoolTrimTo(mem_pool, min_bytes_to_hold));

  attr = hipMemPoolReuseFollowEventDependencies;
  int value = 0;

  HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &value));

  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value));

  hipMemAccessDesc desc_list = {{hipMemLocationTypeDevice, 0}, hipMemAccessFlagsProtReadWrite};
  int count = 1;
  HIP_CHECK(hipMemPoolSetAccess(mem_pool, &desc_list, count));

  hipMemAccessFlags flags = hipMemAccessFlagsProtNone;
  hipMemLocation location = {hipMemLocationTypeDevice, 0};
  HIP_CHECK(hipMemPoolGetAccess(&flags, mem_pool, &location));
  initMemPoolProps();
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolProps));
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&B), numElements * sizeof(float),
                                   mem_pool, stream));
  HIP_CHECK(hipMemPoolDestroy(mem_pool));

  HIP_CHECK(hipStreamDestroy(stream));
}

TEST_CASE("Unit_hipMemPoolApi_BasicAlloc") {
  int mem_pool_support = 0;
  HIP_CHECK(hipSetDevice(0));

  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }
  unsigned int* notified = nullptr;
  HIP_CHECK(hipHostMalloc(&notified, sizeof(unsigned int)));
  *notified = 0;
  initMemPoolProps();
  hipMemPool_t mem_pool;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolProps));

  float *B, *C;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  size_t numElements = 8 * 1024 * 1024;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&B), numElements * sizeof(float),
                                   mem_pool, stream));

  numElements = 1024;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&C), numElements * sizeof(float),
                                   mem_pool, stream));

  int blocks = 1024;
  hipMemPoolAttr attr;
  notifiedKernel<<<32, blocks, 0, stream>>>(B, notified);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(B), stream));

  attr = hipMemPoolAttrReservedMemCurrent;
  std::uint64_t res_before_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &res_before_sync));

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  *notified = 1;  // Notify kernel loop to exit

  HIP_CHECK(hipStreamSynchronize(stream));

  std::uint64_t res_after_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &res_after_sync));
  // Sync must releaae memory to OS
  REQUIRE(res_after_sync <= res_before_sync);

  int value = 0;
  attr = hipMemPoolReuseFollowEventDependencies;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value));
  // Default enabled
  REQUIRE(1 == value);

  attr = hipMemPoolReuseAllowOpportunistic;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value));
  // Default enabled
  REQUIRE(1 == value);

  attr = hipMemPoolReuseAllowInternalDependencies;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value));
  // Default enabled
  REQUIRE(1 == value);

  attr = hipMemPoolAttrReleaseThreshold;
  std::uint64_t value64 = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Default is 0
  REQUIRE(0 == value64);

  attr = hipMemPoolAttrReservedMemHigh;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Must be bigger than current
  REQUIRE(value64 >= res_after_sync);

  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the current usage query works - just small buffer left
  REQUIRE(sizeof(float) * 1024 == value64);

  attr = hipMemPoolAttrUsedMemHigh;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the high watermark usage works - the both buffers must be reported
  REQUIRE(sizeof(float) * (8 * 1024 * 1024 + 1024) == value64);

  HIP_CHECK(hipMemPoolDestroy(mem_pool));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(C), stream));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipHostFree(notified));
}

TEST_CASE("Unit_hipMemPoolApi_BasicTrim") {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }
  unsigned int* notified = nullptr;
  HIP_CHECK(hipHostMalloc(&notified, sizeof(unsigned int)));
  *notified = 0;
  initMemPoolProps();
  hipMemPool_t mem_pool;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolProps));

  float *B, *C;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  size_t numElements = 8 * 1024 * 1024;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&B), numElements * sizeof(float),
                                   mem_pool, stream));

  numElements = 1024;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&C), numElements * sizeof(float),
                                   mem_pool, stream));

  int blocks = 2;
  notifiedKernel<<<32, blocks, 0, stream>>>(B, notified);

  hipMemPoolAttr attr;
  attr = hipMemPoolAttrReleaseThreshold;
  // The pool must hold 128MB
  std::uint64_t threshold = 128 * 1024 * 1024;
  HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &threshold));

  // Not a real free, since kernel isn't done
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(B), stream));

  // Get reserved memory before trim
  attr = hipMemPoolAttrReservedMemCurrent;
  std::uint64_t res_before_trim = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &res_before_trim));

  size_t min_bytes_to_hold = sizeof(float) * 1024;
  HIP_CHECK(hipMemPoolTrimTo(mem_pool, min_bytes_to_hold));

  std::uint64_t res_after_trim = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &res_after_trim));
  // Trim must be a nop because execution isn't done
  REQUIRE(res_before_trim == res_after_trim);

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  *notified = 1;  // Notify kernel loop to exit
  HIP_CHECK(hipStreamSynchronize(stream));

  std::uint64_t res_after_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &res_after_sync));
  // Since hipMemPoolAttrReleaseThreshold is 128 MB sync does nothing to the freed memory
  REQUIRE(res_after_trim == res_after_sync);

  HIP_CHECK(hipMemPoolTrimTo(mem_pool, min_bytes_to_hold));

  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &res_after_trim));
  // Validate memory after real trim. The pool must hold less memory than before
  REQUIRE(res_after_trim < res_after_sync);

  attr = hipMemPoolAttrReleaseThreshold;
  std::uint64_t value64 = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the threshold query works
  REQUIRE(threshold == value64);

  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the current usage query works - just small buffer left
  REQUIRE(sizeof(float) * 1024 == value64);

  attr = hipMemPoolAttrUsedMemHigh;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the high watermark usage works - the both buffers must be reported
  REQUIRE(sizeof(float) * (8 * 1024 * 1024 + 1024) == value64);

  HIP_CHECK(hipMemPoolDestroy(mem_pool));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(C), stream));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipHostFree(notified));
}

TEST_CASE("Unit_hipMemPoolApi_BasicReuse") {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }
  unsigned int* notified = nullptr;
  HIP_CHECK(hipHostMalloc(&notified, sizeof(unsigned int)));
  *notified = 0;
  initMemPoolProps();
  hipMemPool_t mem_pool;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolProps));

  float *A, *B, *C;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  size_t numElements = 8 * 1024 * 1024;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A), numElements * sizeof(float),
                                   mem_pool, stream));

  numElements = 1024;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&C), numElements * sizeof(float),
                                   mem_pool, stream));

  int blocks = 2;
  notifiedKernel<<<32, blocks, 0, stream>>>(A, notified);

  hipMemPoolAttr attr;
  // Not a real free, since kernel isn't done
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A), stream));

  numElements = 8 * 1024 * 1024;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&B), numElements * sizeof(float),
                                   mem_pool, stream));
  // Runtime must reuse the pointer
  REQUIRE(A == B);

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  *notified = 1;  // Notify kernel loop to exit
  // Make a sync before the second kernel launch to make sure memory B isn't gone
  HIP_CHECK(hipStreamSynchronize(stream));

  // Second kernel launch with new memory
  *notified = 0;
  notifiedKernel<<<32, blocks, 0, stream>>>(B, notified);

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  *notified = 1;  // Notify kernel loop to exit
  HIP_CHECK(hipStreamSynchronize(stream));

  attr = hipMemPoolAttrUsedMemCurrent;
  std::uint64_t value64 = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the current usage reports the both buffers
  REQUIRE(sizeof(float) * (8 * 1024 * 1024 + 1024) == value64);

  attr = hipMemPoolAttrUsedMemHigh;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the high watermark usage works - the both buffers must be reported
  REQUIRE(sizeof(float) * (8 * 1024 * 1024 + 1024) == value64);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(B), stream));
  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the current usage reports just one buffer, because the above free doesn't hold memory
  REQUIRE(sizeof(float) * 1024 == value64);

  HIP_CHECK(hipMemPoolDestroy(mem_pool));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(C), stream));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipHostFree(notified));
}

TEST_CASE("Unit_hipMemPoolApi_Opportunistic") {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }
  unsigned int *notified1 = nullptr, *notified2 = nullptr;
  HIP_CHECK(hipHostMalloc(&notified1, sizeof(unsigned int)));
  HIP_CHECK(hipHostMalloc(&notified2, sizeof(unsigned int)));
  *notified1 = 0;
  *notified2 = 0;
  initMemPoolProps();
  hipMemPool_t mem_pool;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolProps));

  hipMemPoolAttr attr;
  int blocks = 2;

  float *A, *B, *C;
  hipStream_t stream1, stream2;

  // Create 2 async non-blocking streams
  HIP_CHECK(hipStreamCreateWithFlags(&stream1, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&stream2, hipStreamNonBlocking));

  size_t numElements = 1024;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&C), numElements * sizeof(float),
                                   mem_pool, stream1));
  int value = 0;

  SECTION("Disallow Opportunistic - No Reuse") {
    numElements = 8 * 1024 * 1024;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A), numElements * sizeof(float),
                                     mem_pool, stream1));

    // Disable all default pool states
    attr = hipMemPoolReuseFollowEventDependencies;
    HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &value));
    attr = hipMemPoolReuseAllowOpportunistic;
    HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &value));
    attr = hipMemPoolReuseAllowInternalDependencies;
    HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &value));

    // Run kernel in the first stream
    notifiedKernel<<<32, blocks, 0, stream1>>>(A, notified1);

    // Not a real free, since kernel isn't done
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A), stream1));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    *notified1 = 1;  // Notify kernel loop to exit

    // Sleep for 1 second GPU should be idle by now
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    numElements = 8 * 1024 * 1024;
    // Allocate memory for the second stream
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&B), numElements * sizeof(float),
                                     mem_pool, stream2));
    // Without Opportunistic state runtime must allocate another buffer
    REQUIRE(A != B);

    // Run kernel with the new memory in the second streamn
    notifiedKernel<<<32, blocks, 0, stream2>>>(B, notified2);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    *notified2 = 1;  // Notify kernel loop to exit
    HIP_CHECK(hipStreamSynchronize(stream1));
    HIP_CHECK(hipStreamSynchronize(stream2));

    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(B), stream2));
  }

  SECTION("Allow Opportunistic - Reuse") {
    numElements = 8 * 1024 * 1024;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A), numElements * sizeof(float),
                                     mem_pool, stream1));

    value = 1;
    attr = hipMemPoolReuseAllowOpportunistic;
    // Enable Opportunistic
    HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &value));

    // Run kernel in the first stream
    notifiedKernel<<<32, blocks, 0, stream1>>>(A, notified1);

    // Not a real free, since kernel isn't done
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A), stream1));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    *notified1 = 1;  // Notify kernel loop to exit

    // Sleep for 1 second GPU should be idle by now
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    numElements = 8 * 1024 * 1024;
    // Allocate memory for the second stream
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&B), numElements * sizeof(float),
                                     mem_pool, stream2));
    // With Opportunistic state runtime will reuse freed buffer A
    REQUIRE(A == B);

    // Run kernel with the new memory in the second stream
    notifiedKernel<<<32, blocks, 0, stream2>>>(B, notified2);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    *notified2 = 1;  // Notify kernel loop to exit

    HIP_CHECK(hipStreamSynchronize(stream1));
    HIP_CHECK(hipStreamSynchronize(stream2));

    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(B), stream2));
  }

  SECTION("Allow Opportunistic - No Reuse") {
    numElements = 8 * 1024 * 1024;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A), numElements * sizeof(float),
                                     mem_pool, stream1));

    value = 1;
    attr = hipMemPoolReuseAllowOpportunistic;
    // Enable Opportunistic
    HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attr, &value));

    // Run kernel in the first stream
    notifiedKernel<<<32, blocks, 0, stream1>>>(A, notified1);

    // Not a real free, since kernel isn't done
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A), stream1));

    numElements = 8 * 1024 * 1024;
    // Allocate memory for the second stream
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&B), numElements * sizeof(float),
                                     mem_pool, stream2));
    // With Opportunistic state runtime can't reuse freed buffer A, because it's still busy with the
    // kernel
    REQUIRE(A != B);

    // Run kernel with the new memory in the second stream
    notifiedKernel<<<32, blocks, 0, stream2>>>(B, notified2);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    *notified1 = 1;  // Notify kernel loop to exit
    *notified2 = 1;  // Notify kernel loop to exit
    HIP_CHECK(hipStreamSynchronize(stream1));
    HIP_CHECK(hipStreamSynchronize(stream2));

    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(B), stream2));
  }

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(C), stream1));
  HIP_CHECK(hipMemPoolDestroy(mem_pool));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipHostFree(notified1));
  HIP_CHECK(hipHostFree(notified2));
}

TEST_CASE("Unit_hipMemPoolApi_Default") {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }
  unsigned int* notified = nullptr;
  HIP_CHECK(hipHostMalloc(&notified, sizeof(unsigned int)));
  *notified = 0;
  hipMemPool_t mem_pool;
  HIP_CHECK(hipDeviceGetDefaultMemPool(&mem_pool, 0));

  float *A, *B, *C;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  size_t numElements = 8 * 1024 * 1024;
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&A), numElements * sizeof(float), stream));

  numElements = 1024;
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&C), numElements * sizeof(float), stream));

  int blocks = 2;
  notifiedKernel<<<32, blocks, 0, stream>>>(A, notified);

  hipMemPoolAttr attr;
  // Not a real free, since kernel isn't done
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A), stream));

  numElements = 8 * 1024 * 1024;
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&B), numElements * sizeof(float), stream));
  // Runtime must reuse the pointer
  REQUIRE(A == B);

  // Make a sync before the second kernel launch to make sure memory B isn't gone
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  *notified = 1;  // Notify kernel loop to exit
  HIP_CHECK(hipStreamSynchronize(stream));

  // Second kernel launch with new memory
  *notified = 0;
  notifiedKernel<<<32, blocks, 0, stream>>>(B, notified);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(B), stream));

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  *notified = 1;  // Notify kernel loop to exit
  HIP_CHECK(hipStreamSynchronize(stream));

  std::uint64_t value64 = 0;
  attr = hipMemPoolAttrReservedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the current reserved is at least allocation size of buffer C (4KB)
  REQUIRE(sizeof(float) * 1024 <= value64);

  attr = hipMemPoolAttrUsedMemHigh;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the high watermark usage works - the both buffers must be reported
  REQUIRE(sizeof(float) * (8 * 1024 * 1024 + 1024) == value64);

  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &value64));
  // Make sure the current usage reports just one buffer, because the above free doesn't hold memory
  REQUIRE(sizeof(float) * 1024 == value64);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(C), stream));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipHostFree(notified));
}
