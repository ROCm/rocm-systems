/*
   Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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


#include "mempool_common.hh"

#include <resource_guards.hh>
#include <utils.hh>
/**
 * @addtogroup hipMemPoolTrimTo hipMemPoolTrimTo
 * @{
 * @ingroup StreamOTest
 * `hipMemPoolTrimTo(hipMemPool_t mem_pool, size_t min_bytes_to_hold)` -
 * Releases freed memory back to the OS
 */


/**
 * Test Description
 * ------------------------
 *  - Test to verify hipMemPoolTrimTo behavior with invalid arguments:
 *    -# Nullptr mem_pool
 *
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolTrimTo.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolTrimTo_Negative_Parameter") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));
  checkMempoolSupported(device_id) size_t trim_size = 1024;

  SECTION("Passing nullptr to mem_pool") {
    HIP_CHECK_ERROR(hipMemPoolTrimTo(nullptr, trim_size), hipErrorInvalidValue);
  }
}


/**
 * Test Description
 * ------------------------
 *  - Basic test to verify hipMemPoolTrimTo releases memory correctly to the OS.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolTrimTo.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolTrimTo_Positive_Basic") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));
  checkMempoolSupported(device_id) unsigned int* notified = nullptr;
  HIP_CHECK(hipHostMalloc(&notified, sizeof(unsigned int)));
  *notified = 0;

  const size_t allocation_size1 = kPageSize * kPageSize * 2;
  const size_t allocation_size2 = kPageSize / 2;
  MemPoolGuard mempool(MemPools::created, device_id);

  int* alloc_mem1;
  int* alloc_mem2;
  StreamGuard stream(Streams::created);

  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem1), allocation_size1,
                                   mempool.mempool(), stream.stream()));
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem2), allocation_size2,
                                   mempool.mempool(), stream.stream()));

  int blocks = 2;
  notifiedKernel<<<32, blocks, 0, stream.stream()>>>(alloc_mem1, notified);

  hipMemPoolAttr attr;
  attr = hipMemPoolAttrReleaseThreshold;
  // The pool must hold 128MB
  std::uint64_t threshold = 128 * 1024 * 1024;
  HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &threshold));

  // Not a real free, since kernel isn't done
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream.stream()));

  // Get reserved memory before trim
  attr = hipMemPoolAttrReservedMemCurrent;
  std::uint64_t res_before_trim = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &res_before_trim));

  size_t min_bytes_to_hold = allocation_size2;
  HIP_CHECK(hipMemPoolTrimTo(mempool.mempool(), min_bytes_to_hold));

  std::uint64_t res_after_trim = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &res_after_trim));
  // Trim must be a nop because execution isn't done
  REQUIRE(res_before_trim == res_after_trim);

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  *notified = 1;
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::uint64_t res_after_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &res_after_sync));
  // Since hipMemPoolAttrReleaseThreshold is 128 MB sync does nothing to the freed memory
  REQUIRE(res_after_trim == res_after_sync);

  HIP_CHECK(hipMemPoolTrimTo(mempool.mempool(), min_bytes_to_hold));

  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &res_after_trim));
  // Validate memory after real trim. The pool must hold less memory than before
  REQUIRE(res_after_trim < res_after_sync);

  attr = hipMemPoolAttrReleaseThreshold;
  std::uint64_t value64 = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &value64));
  // Make sure the threshold query works
  REQUIRE(threshold == value64);

  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &value64));
  // Make sure the current usage query works - just small buffer left
  REQUIRE(allocation_size2 == value64);

  attr = hipMemPoolAttrUsedMemHigh;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &value64));
  // Make sure the high watermark usage works - the both buffers must be reported
  REQUIRE((allocation_size1 + allocation_size2) == value64);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream.stream()));
  HIP_CHECK(hipHostFree(notified));
}

static bool thread_results[NUMBER_OF_THREADS];

/**
 * Local function to test hipMemPoolAttrReleaseThreshold.
 */
static bool checkhipMemPoolTrimTo(hipStream_t stream, int N, int dev = 0) {
  streamMemAllocTest testObj(N);
  size_t byte_size = N * sizeof(int);
  // assign memory to host pointers
  testObj.createHostBufferWithData();
  // Create mempool in current device
  hipMemPool_t mem_pool;
  hipMemPoolProps pool_props{};
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = dev;
  pool_props.location.type = hipMemLocationTypeDevice;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));
  uint64_t setThreshold = UINT64_MAX;
  HIP_CHECK(hipMemPoolSetAttribute(mem_pool, hipMemPoolAttrReleaseThreshold, &setThreshold));
  testObj.useCommonMempool(mem_pool);
  for (int iter = 1; iter <= LAUNCH_ITERATIONS; iter++) {
    // Set different min_bytes_to_hold for each iteration
    size_t min_bytes_to_hold = (byte_size * 3 * (LAUNCH_ITERATIONS - iter)) / LAUNCH_ITERATIONS;
    HIP_CHECK(hipMemPoolTrimTo(mem_pool, min_bytes_to_hold));
    // assign memory to device pointers
    testObj.allocFromMempool(stream);
    testObj.transferToMempool(stream);
    testObj.runKernel(stream);
    testObj.transferFromMempool(stream);
    testObj.freeDevBuf(stream);
    // verify and validate
    REQUIRE(true == testObj.validateResult());
    HIP_CHECK(hipStreamSynchronize(stream));
  }
  HIP_CHECK(hipMemPoolDestroy(mem_pool));
  return true;
}

/**
 * Test Description
 * ------------------------
 *    - Create explicit mempool1 on default GPU and set attribute
 * hipMemPoolAttrReleaseThreshold to UINT64_MAX.
 * LOOP for 10 times: {Trim the memory pool in each iteration, then
 * Allocate A_d1, B_d1, C_d1 from pool1, memcpy data to (A_d1, B_d1).
 * Launch kernel to perform C_d1(x)=A_d1(x)+B_d1(x), verify
 * result and free the memory.} After loop free the pool.
 * ------------------------
 *    - catch\unit\memory\hipMemPoolTrimTo.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolTrimTo_VaryingMinBytesToHold") {
  checkMempoolSupported(0)
      // create a stream
      hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  constexpr int N = 1 << 20;
  REQUIRE(true == checkhipMemPoolTrimTo(stream, N));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *    - MultiGPU scenario: Execute the above scenario in each device.
 * ------------------------
 *    - catch\unit\memory\hipMemPoolTrimTo.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolTrimTo_MGpuVaryingMinBytesToHold") {
  constexpr int N = 1 << 20;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices < 2) {
    WARN("Number of GPUs insufficient for test");
  } else {
    for (int dev = 0; dev < numDevices; dev++) {
      checkMempoolSupported(dev) HIP_CHECK(hipSetDevice(dev));
      // create a stream
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));
      REQUIRE(true == checkhipMemPoolTrimTo(stream, N, dev));
      HIP_CHECK(hipStreamDestroy(stream));
    }
  }
}

/**
 * Local Thread Functions
 */
static void thread_Test(hipStream_t stream, int N, int threadNum) {
  thread_results[threadNum] = checkhipMemPoolTrimTo(stream, N, false);
}

/**
 * Test Description
 * ------------------------
 *    - Multithread scenario: Execute the above scenario in each thread.
 * ------------------------
 *    - catch\unit\memory\hipMemPoolTrimTo.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemPoolTrimTo_Multithreaded") {
  checkMempoolSupported(0)
      // create a stream
      constexpr int N = 1 << 20;
  std::vector<std::thread> tests;
  hipStream_t stream[NUMBER_OF_THREADS];
  // Initialize and create streams
  for (int idx = 0; idx < NUMBER_OF_THREADS; idx++) {
    thread_results[idx] = false;
    HIP_CHECK(hipStreamCreate(&stream[idx]));
  }
  // Spawn the test threads
  for (int idx = 0; idx < NUMBER_OF_THREADS; idx++) {
    tests.push_back(std::thread(thread_Test, stream[idx], N, idx));
  }
  // Wait for all threads to complete
  for (std::thread& t : tests) {
    t.join();
  }
  // Wait for thread and destroy stream
  bool status = true;
  for (int idx = 0; idx < NUMBER_OF_THREADS; idx++) {
    status = status & thread_results[idx];
    HIP_CHECK(hipStreamDestroy(stream[idx]));
  }
}

/**
 * End doxygen group StreamOTest.
 * @}
 */
