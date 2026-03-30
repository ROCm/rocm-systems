/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "mempool_common.hh"

#include <limits>

static bool thread_results[NUMBER_OF_THREADS];
static constexpr int streamPerAsic = 2;
static hipMemPool_t mem_pool_common;

/**
 * @addtogroup hipMallocFromPoolAsync hipMallocFromPoolAsync
 * @{
 * @ingroup StreamOTest
 * `hipMallocFromPoolAsync(void** dev_ptr, size_t size, hipMemPool_t mem_pool, hipStream_t stream)`
 * - Allocates memory from a specified pool with stream ordered semantics
 */


/**
 * Test Description
 * ------------------------
 *  - Basic test to verify proper allocation and stream ordering of hipMallocFromPoolAsync when one
 * memory allocation is performed.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMallocFromPoolAsync_Basic_OneAlloc) {
  MallocMemPoolAsync_OneAlloc(
      [](void** dev_ptr, size_t size, hipMemPool_t mem_pool, hipStream_t stream) {
        return hipMallocFromPoolAsync(dev_ptr, size, mem_pool, stream);
      },
      MemPools::created);
}


/**
 * Test Description
 * ------------------------
 *  - Basic test to verify proper allocation and stream ordering of hipMallocFromPoolAsync when two
 * memory allocations are performed.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMallocFromPoolAsync_Basic_TwoAllocs) {
  MallocMemPoolAsync_TwoAllocs(
      [](void** dev_ptr, size_t size, hipMemPool_t mem_pool, hipStream_t stream) {
        return hipMallocFromPoolAsync(dev_ptr, size, mem_pool, stream);
      },
      MemPools::created);
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify that memory allocated with hipMallocFromPoolAsync can be properly reused.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMallocFromPoolAsync_Basic_Reuse) {
  MallocMemPoolAsync_Reuse(
      [](void** dev_ptr, size_t size, hipMemPool_t mem_pool, hipStream_t stream) {
        return hipMallocFromPoolAsync(dev_ptr, size, mem_pool, stream);
      },
      MemPools::created);
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipMallocFromPoolAsync behavior with invalid arguments:
 *    -# Nullptr dev_ptr
 *    -# Nullptr mem_pool
 *    -# Invalid stream handle
 *    -# Size is max size_t
 *
 * Test source
 * ------------------------
 *  - /unit/memory/hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMallocFromPoolAsync_Negative_Parameters) {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  checkMempoolSupported(0);

  void* p = nullptr;
  size_t max_size = std::numeric_limits<size_t>::max();
  size_t alloc_size = 1024;
  MemPoolGuard mempool(MemPools::created, device_id);
  StreamGuard stream(Streams::created);

  SECTION("dev_ptr is nullptr") {
    HIP_CHECK_ERROR(hipMallocFromPoolAsync(nullptr, alloc_size, mempool.mempool(), stream.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("Mempool not created") {
    hipMemPool_t dummy_mem_pool = nullptr;
    HIP_CHECK_ERROR(hipMallocFromPoolAsync(static_cast<void**>(&p), alloc_size, dummy_mem_pool,
                                           stream.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("Size is max size_t") {
    HIP_CHECK_ERROR(hipMallocFromPoolAsync(static_cast<void**>(&p), max_size, mempool.mempool(),
                                           stream.stream()),
                    hipErrorOutOfMemory);
  }
}

/**
 * Local function to test mempool allocation, usage and freeing on
 * multiple user created Streams with inter Stream synchonization.
 */
static bool checkMempoolMultStreamSync(int N) {
  streamMemAllocTest testObj(N);
  // create multiple streams
  hipStream_t streamMemCreate, streamMemAccess, streamMemDestroy;
  HIP_CHECK(hipStreamCreate(&streamMemCreate));
  HIP_CHECK(hipStreamCreate(&streamMemAccess));
  HIP_CHECK(hipStreamCreate(&streamMemDestroy));
  // Create host buffer with test data
  testObj.createHostBufferWithData();
  // Create mempool in current device = 0
  testObj.createMempool(hipMemPoolAttrReleaseThreshold, testdefault, 0);
  hipEvent_t Event1, Event2;
  HIP_CHECK(hipEventCreate(&Event1));
  HIP_CHECK(hipEventCreate(&Event2));
  // Allocate memory and initialize it on streamMemCreate
  testObj.allocFromMempool(streamMemCreate);
  testObj.transferToMempool(streamMemCreate);
  HIP_CHECK(hipEventRecord(Event1, streamMemCreate));
  // Launch Kernel on streamMemAccess
  HIP_CHECK(hipStreamWaitEvent(streamMemAccess, Event1, 0));
  testObj.runKernel(streamMemAccess);
  testObj.transferFromMempool(streamMemAccess);
  HIP_CHECK(hipEventRecord(Event2, streamMemAccess));
  // Launch Kernel on streamMemAccess
  HIP_CHECK(hipStreamWaitEvent(streamMemDestroy, Event2, 0));
  testObj.freeDevBuf(streamMemDestroy);
  HIP_CHECK(hipStreamSynchronize(streamMemDestroy));
  // Validate test result and clean all host buffers and mempool
  bool results = false;
  results = testObj.validateResult();
  testObj.freeMempool();
  testObj.freeHostBuf();
  HIP_CHECK(hipEventDestroy(Event2));
  HIP_CHECK(hipEventDestroy(Event1));
  HIP_CHECK(hipStreamDestroy(streamMemDestroy));
  HIP_CHECK(hipStreamDestroy(streamMemAccess));
  HIP_CHECK(hipStreamDestroy(streamMemCreate));
  return results;
}

/**
 * Local function to test mempool functionality on a user created
 * stream, null stream and hipStreamPerThread concurrently. Wait
 * for all the streams to complete and validate result.
 */
static bool checkMempoolMultStreamConcurrentExec(int N, bool useDefStrm = true) {
  streamMemAllocTest testObj[3] = {streamMemAllocTest(N), streamMemAllocTest(N),
                                   streamMemAllocTest(N)};
  // create multiple streams
  hipStream_t testStreams[3];
  HIP_CHECK(hipStreamCreate(&testStreams[0]));
  if (useDefStrm) {
    testStreams[1] = 0;  // null stream
    testStreams[2] = hipStreamPerThread;
  } else {
    HIP_CHECK(hipStreamCreate(&testStreams[1]));
    HIP_CHECK(hipStreamCreate(&testStreams[2]));
  }
  // Create common mempool
  hipMemPoolProps pool_props{};
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = 0;
  pool_props.location.type = hipMemLocationTypeDevice;
  HIP_CHECK(hipMemPoolCreate(&mem_pool_common, &pool_props));
  bool results = true;
  for (int idx = 0; idx < 3; idx++) {
    // Create mempool in current device = 0
    testObj[idx].useCommonMempool(mem_pool_common);
    // Create host buffer with test data
    testObj[idx].createHostBufferWithData();
    // Allocate memory and initialize it on testStreams[idx]
    testObj[idx].allocFromMempool(testStreams[idx]);
    testObj[idx].transferToMempool(testStreams[idx]);
    // Launch Kernel on testStreams[idx]
    testObj[idx].runKernel(testStreams[idx]);
    testObj[idx].transferFromMempool(testStreams[idx]);
    testObj[idx].freeDevBuf(testStreams[idx]);
  }
  for (int idx = 0; idx < 3; idx++) {
    HIP_CHECK(hipStreamSynchronize(testStreams[idx]));
    // Validate test result and clean all host buffers and mempool
    results &= testObj[idx].validateResult();
    testObj[idx].freeHostBuf();
  }
  HIP_CHECK(hipStreamDestroy(testStreams[0]));
  if (!useDefStrm) {
    HIP_CHECK(hipStreamDestroy(testStreams[1]));
    HIP_CHECK(hipStreamDestroy(testStreams[2]));
  }
  // Destroy common mempool
  HIP_CHECK(hipMemPoolDestroy(mem_pool_common));
  return results;
}

/**
 * Local function to test hipMemPoolAttrReleaseThreshold.
 */
static bool checkMaximumAndDefaultThreshold(hipStream_t stream, int N, enum eTestValue testtype,
                                            int dev = 0) {
  streamMemAllocTest testObj(N);
  // Create host buffer with test data
  testObj.createHostBufferWithData();
  // Create mempool in current device = dev
  testObj.createMempool(hipMemPoolAttrReleaseThreshold, testtype, dev);
  bool results = true;
  for (int iter = 0; iter < LAUNCH_ITERATIONS; iter++) {
    // Allocate memory and initialize it on stream
    testObj.allocFromMempool(stream);
    testObj.transferToMempool(stream);
    testObj.runKernel(stream);
    testObj.transferFromMempool(stream);
    // validate
    testObj.freeDevBuf(stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    results = testObj.validateResult();
    if (!results) {
      break;
    }
  }
  testObj.freeMempool();
  testObj.freeHostBuf();
  return results;
}


/**
 * Local function to test hipMemPoolAttrReleaseThreshold with multiple threads.
 * One thread per device; uses HIP_CHECK_THREAD and REQUIRE_THREAD.
 */
static void checkMaximumAndDefaultThresholdMultiThreaded(hipStream_t stream, int N,
                                                         enum eTestValue testtype, int dev) {
  HIP_CHECK_THREAD(hipSetDevice(dev));

  size_t byte_size = N * sizeof(int);
  int *A_h, *B_h, *C_h;
  int *A_d, *B_d, *C_d;

  A_h = reinterpret_cast<int*>(malloc(byte_size));
  REQUIRE_THREAD(A_h != nullptr);
  B_h = reinterpret_cast<int*>(malloc(byte_size));
  REQUIRE_THREAD(B_h != nullptr);
  C_h = reinterpret_cast<int*>(malloc(byte_size));
  REQUIRE_THREAD(C_h != nullptr);
  for (int i = 0; i < N; i++) {
    A_h[i] = 2 * i + 1;
    B_h[i] = 2 * i;
    C_h[i] = 0;
  }

  hipMemPool_t mem_pool;
  hipMemPoolProps pool_props{};
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.location.id = dev;
  pool_props.location.type = hipMemLocationTypeDevice;
  HIP_CHECK_THREAD(hipMemPoolCreate(&mem_pool, &pool_props));
  uint64_t setThreshold = 0;
  if (testtype == testMaximum) {
    setThreshold = UINT64_MAX;
  }
  HIP_CHECK_THREAD(hipMemPoolSetAttribute(mem_pool, hipMemPoolAttrReleaseThreshold, &setThreshold));

  for (int iter = 0; iter < LAUNCH_ITERATIONS; iter++) {
    HIP_CHECK_THREAD(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d),
                     byte_size, mem_pool, stream));
    HIP_CHECK_THREAD(hipMallocFromPoolAsync(reinterpret_cast<void**>(&B_d),
                     byte_size, mem_pool, stream));
    HIP_CHECK_THREAD(hipMallocFromPoolAsync(reinterpret_cast<void**>(&C_d),
                     byte_size, mem_pool, stream));

    HIP_CHECK_THREAD(hipMemcpyAsync(A_d, A_h, byte_size, hipMemcpyHostToDevice, stream));
    HIP_CHECK_THREAD(hipMemcpyAsync(B_d, B_h, byte_size, hipMemcpyHostToDevice, stream));

    int blocks = (N % THREADS_PER_BLOCK == 0) ? (N / THREADS_PER_BLOCK)
                                              : ((N / THREADS_PER_BLOCK) + 1);
    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(THREADS_PER_BLOCK), 0, stream,
                       static_cast<const int*>(A_d), static_cast<const int*>(B_d), C_d, N);
    HIP_CHECK_THREAD(hipGetLastError());

    HIP_CHECK_THREAD(hipMemcpyAsync(C_h, C_d, byte_size, hipMemcpyDeviceToHost, stream));

    HIP_CHECK_THREAD(hipFreeAsync(reinterpret_cast<void*>(A_d), stream));
    HIP_CHECK_THREAD(hipFreeAsync(reinterpret_cast<void*>(B_d), stream));
    HIP_CHECK_THREAD(hipFreeAsync(reinterpret_cast<void*>(C_d), stream));

    HIP_CHECK_THREAD(hipStreamSynchronize(stream));

    for (int i = 0; i < N; i++) {
      auto res = A_h[i] + B_h[i];
      REQUIRE_THREAD(res == C_h[i]);
    }
  }

  HIP_CHECK_THREAD(hipMemPoolDestroy(mem_pool));
  free(A_h);
  free(B_h);
  free(C_h);
}


/**
 * Test Description
 * ------------------------
 *    -  Create explicit mempool1 on default GPU and set attribute
 * hipMemPoolAttrReleaseThreshold to UINT64_MAX. Create another explicit
 * mempool2 on default GPU with default attribute.
 * LOOP for 10 times: {Allocate A_d1, B_d1, C_d1 from pool1, memcpy data to
 * (A_d1, B_d1). Launch kernel to perform C_d1(x)=A_d1(x)+B_d1(x), verify
 * result and free the memory.} After loop free the pool.
 * LOOP for 10 times: {Allocate A_d2, B_d2, C_d2 from pool2, memcpy data to
 * (A_d2, B_d2). Launch kernel to perform C_d2(x)=A_d2(x)+B_d2(x), verify
 * result and free the memory.} After loop free the pool.
 * ------------------------
 *    - catch\unit\memory\hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMallocFromPoolAsync_ReleaseThreshold) {
  checkMempoolSupported(0)
      // create a stream
      hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  constexpr int N = 1 << 20;
  REQUIRE(true == checkMaximumAndDefaultThreshold(stream, N, testdefault));
  REQUIRE(true == checkMaximumAndDefaultThreshold(stream, N, testMaximum));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *    - Validate hipMallocFromPoolAsync functionality on null stream.
 * ------------------------
 *    - catch\unit\memory\hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMallocFromPoolAsync_NullStream) {
  checkMempoolSupported(0) constexpr int N = 1 << 20;
  REQUIRE(true == checkMaximumAndDefaultThreshold(0, N, testdefault));
  REQUIRE(true == checkMaximumAndDefaultThreshold(0, N, testMaximum));
}

/**
 * Test Description
 * ------------------------
 *    - Validate hipMallocFromPoolAsync functionality on hipStreamPerThread.
 * ------------------------
 *    - catch\unit\memory\hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMallocFromPoolAsync_hipStreamPerThread) {
  checkMempoolSupported(0) constexpr int N = 1 << 20;
  REQUIRE(true == checkMaximumAndDefaultThreshold(hipStreamPerThread, N, testdefault));
  REQUIRE(true == checkMaximumAndDefaultThreshold(hipStreamPerThread, N, testMaximum));
}

/**
 * Test Description
 * ------------------------
 *    - Check Release Threshold for multiple device.
 * ------------------------
 *    - catch\unit\memory\hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMallocFromPoolAsync_ReleaseThreshold_Mgpu) {
  constexpr int N = 1 << 20;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  for (int dev = 0; dev < numDevices; dev++) {
    checkMempoolSupported(dev) HIP_CHECK(hipSetDevice(dev));
    // create a stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    REQUIRE(true == checkMaximumAndDefaultThreshold(stream, N, testdefault, dev));
    REQUIRE(true == checkMaximumAndDefaultThreshold(stream, N, testMaximum, dev));
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

/**
 * Local Thread Functions
 */
static void threadQAsyncCommands(streamMemAllocTest* testObj, hipStream_t strm, int idx) {
  HIP_CHECK(hipSetDevice(idx));
  // Create host buffer with test data.
  testObj->createHostBufferWithData();
  // Allocate device memory and transfer data to it asyncronously on stream.
  testObj->allocFromMempool(strm);
  testObj->transferToMempool(strm);
  // Execute kernel and transfer result back to host asynchronously on stream.
  testObj->runKernel(strm);
  testObj->transferFromMempool(strm);
  // Free Buffer Asynchronously on stream.
  testObj->freeDevBuf(strm);
}

static void thread_Test1(hipStream_t stream, int N, enum eTestValue testtype, int threadNum) {
  thread_results[threadNum] = checkMaximumAndDefaultThreshold(stream, N, testtype, 0);
}

static bool test_hipMallocFromPoolAsync_MThread(enum eTestValue testtype) {
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
    tests.push_back(std::thread(thread_Test1, stream[idx], N, testtype, idx));
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
  return status;
}

static void thread_Test2(hipMemPool_t mempool, hipStream_t stream, int N, int threadNum) {
  streamMemAllocTest testObj(N);
  // Create host buffer with test data
  testObj.createHostBufferWithData();
  // Use the common mempool
  testObj.useCommonMempool(mempool);
  bool results = true;
  for (int iter = 0; iter < LAUNCH_ITERATIONS; iter++) {
    // Allocate memory and initialize it on stream
    testObj.allocFromMempool(stream);
    testObj.transferToMempool(stream);
    testObj.runKernel(stream);
    testObj.transferFromMempool(stream);
    testObj.freeDevBuf(stream);
    // verify and validate
    HIP_CHECK(hipStreamSynchronize(stream));
    results = testObj.validateResult();
    if (!results) {
      break;
    }
  }
  testObj.freeHostBuf();
  thread_results[threadNum] = results;
}

static bool test_hipMallocFromPoolAsync_MThread_CommonMpool(enum eTestValue testtype,
                                                            bool bUseDefault = false) {
  // create a stream
  constexpr int N = 1 << 20;
  std::vector<std::thread> tests;
  hipStream_t stream[NUMBER_OF_THREADS];
  // Create common mempool
  if (bUseDefault) {
    HIP_CHECK(hipDeviceGetDefaultMemPool(&mem_pool_common, 0));
  } else {
    hipMemPoolProps pool_props{};
    pool_props.allocType = hipMemAllocationTypePinned;
    pool_props.location.id = 0;
    pool_props.location.type = hipMemLocationTypeDevice;
    HIP_CHECK(hipMemPoolCreate(&mem_pool_common, &pool_props));
  }
  if (testtype == testMaximum) {
    uint64_t setThreshold = UINT64_MAX;
    HIP_CHECK(
        hipMemPoolSetAttribute(mem_pool_common, hipMemPoolAttrReleaseThreshold, &setThreshold));
  }
  // Initialize and create streams
  for (int idx = 0; idx < NUMBER_OF_THREADS; idx++) {
    thread_results[idx] = false;
    HIP_CHECK(hipStreamCreate(&stream[idx]));
  }
  // Spawn the test threads
  for (int idx = 0; idx < NUMBER_OF_THREADS; idx++) {
    tests.push_back(std::thread(thread_Test2, mem_pool_common, stream[idx], N, idx));
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
  // Destroy common mempool
  if (!bUseDefault) {
    HIP_CHECK(hipMemPoolDestroy(mem_pool_common));
  }
  return status;
}

/**
 * Local function to test hipMemPoolReuseFollowEventDependencies.
 */
static bool checkReuseFollowEventDepFlag(int N, enum eTestValue testtype) {
  streamMemAllocTest testObj(N);
  // Create host buffer with test data
  testObj.createHostBufferWithData();
  // Create mempool in current device = 0
  testObj.createMempool(hipMemPoolReuseFollowEventDependencies, testtype, 0);
  hipStream_t testStream1, testStream2;
  HIP_CHECK(hipStreamCreate(&testStream1));
  HIP_CHECK(hipStreamCreate(&testStream2));
  bool results = true;
  for (int iter = 0; iter < LAUNCH_ITERATIONS; iter++) {
    hipEvent_t Event1;
    HIP_CHECK(hipEventCreate(&Event1));
    // Allocate memory and initialize it on testStream1
    testObj.allocFromMempool(testStream1);
    testObj.transferToMempool(testStream1);
    testObj.runKernel(testStream1);
    testObj.transferFromMempool(testStream1);
    testObj.freeDevBuf(testStream1);
    HIP_CHECK(hipEventRecord(Event1, testStream1));
    HIP_CHECK(hipStreamWaitEvent(testStream2, Event1, 0));
    // Allocate memory and initialize it on testStream2
    testObj.allocFromMempool(testStream2);
    testObj.transferToMempool(testStream2);
    testObj.runKernel(testStream2);
    testObj.transferFromMempool(testStream2);
    testObj.freeDevBuf(testStream2);
    // validate
    HIP_CHECK(hipStreamSynchronize(testStream2));
    HIP_CHECK(hipEventDestroy(Event1));
    results = testObj.validateResult();
    if (!results) {
      break;
    }
  }
  testObj.freeMempool();
  testObj.freeHostBuf();
  HIP_CHECK(hipStreamDestroy(testStream2));
  HIP_CHECK(hipStreamDestroy(testStream1));
  return results;
}

/**
 * Local function to test hipMemPoolReuseAllowOpportunistic and
 * hipMemPoolReuseAllowInternalDependencies.
 */
static bool checkReuseAllowOtherFlags(int N, hipMemPoolAttr attr, enum eTestValue testtype) {
  streamMemAllocTest testObj(N);
  // Create host buffer with test data
  testObj.createHostBufferWithData();
  // Create mempool in current device = 0
  testObj.createMempool(attr, testtype, 0);
  hipStream_t testStream1, testStream2;
  HIP_CHECK(hipStreamCreate(&testStream1));
  HIP_CHECK(hipStreamCreate(&testStream2));
  bool results = true;
  for (int iter = 0; iter < LAUNCH_ITERATIONS; iter++) {
    // Allocate memory and initialize it on testStream1
    testObj.allocFromMempool(testStream1);
    testObj.transferToMempool(testStream1);
    testObj.runKernel(testStream1);
    testObj.transferFromMempool(testStream1);
    testObj.freeDevBuf(testStream1);
    // Allocate memory and initialize it on testStream2
    testObj.allocFromMempool(testStream2);
    testObj.transferToMempool(testStream2);
    testObj.runKernel(testStream2);
    testObj.transferFromMempool(testStream2);
    testObj.freeDevBuf(testStream2);
    // validate
    HIP_CHECK(hipStreamSynchronize(testStream2));
    results = testObj.validateResult();
    if (!results) {
      break;
    }
  }
  testObj.freeMempool();
  testObj.freeHostBuf();
  HIP_CHECK(hipStreamDestroy(testStream2));
  HIP_CHECK(hipStreamDestroy(testStream1));
  return results;
}

/**
 * Test Description
 * ------------------------
 *    - Queue the following commands hipMallocFromPoolAsync, transfer data to it
 * asynchrously, launch Kernel, transfer results back to host asynchronously and
 * free buffer async in streams across all GPUs. The execution in of the queued
 * commands must happen concurrently.
 * ------------------------
 *    - catch\unit\memory\hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
#if HT_AMD
HIP_TEST_CASE(Unit_hipMallocFromPoolAsync_Multidevice_Concurrent) {
  auto testType = GENERATE(testdefault, testMaximum);
  constexpr int N = 1 << 20;
  int num_devices;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  checkIfMultiDev(num_devices) hipStream_t* stream_buf = new hipStream_t[num_devices];
  std::vector<streamMemAllocTest*> tesObjBuf;
  // Allocate resources in each device
  for (int idx = 0; idx < num_devices; idx++) {
    checkMempoolSupported(idx) HIP_CHECK(hipSetDevice(idx));
    HIP_CHECK(hipStreamCreate(&stream_buf[idx]));
    streamMemAllocTest* testObj = new streamMemAllocTest(N);
    testObj->createMempool(hipMemPoolAttrReleaseThreshold, testType, idx);
    tesObjBuf.push_back(testObj);
  }
  // Queue commands in each device
  for (int idx = 0; idx < num_devices; idx++) {
    HIP_CHECK(hipSetDevice(idx));
    std::thread test(threadQAsyncCommands, tesObjBuf[idx], stream_buf[idx], idx);
    test.join();
  }
  // Wait for the streams
  for (int idx = 0; idx < num_devices; idx++) {
    HIP_CHECK(hipSetDevice(idx));
    HIP_CHECK(hipStreamSynchronize(stream_buf[idx]));
    // verify and validate
    REQUIRE(true == tesObjBuf[idx]->validateResult());
  }
  // Deallocate resources in each device
  for (int idx = 0; idx < num_devices; idx++) {
    HIP_CHECK(hipSetDevice(idx));
    // Destroy resources
    tesObjBuf[idx]->freeMempool();
    tesObjBuf[idx]->freeHostBuf();
    HIP_CHECK(hipStreamDestroy(stream_buf[idx]));
    delete tesObjBuf[idx];
  }
  delete[] stream_buf;
}

/**
 * Test Description
 * ------------------------
 *    - Queue the following commands hipMallocFromPoolAsync, transfer data to it
 * asynchrously, launch Kernel, transfer results back to host asynchronously and
 * free buffer async in streams across all GPUs using multiple streams per GPU.
 * ------------------------
 *    - catch\unit\memory\hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMallocFromPoolAsync_Multidevice_MultiStream) {
  int num_devices;
  auto testType = GENERATE(testdefault, testMaximum);
  constexpr int N = 1 << 20;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  checkIfMultiDev(num_devices)
      // 2 stream per ASIC
      hipStream_t* stream_buf = new hipStream_t[streamPerAsic * num_devices];
  std::vector<streamMemAllocTest*> tesObjBuf;
  // Allocate resources in each device
  for (int idx = 0; idx < num_devices; idx++) {
    checkMempoolSupported(idx) HIP_CHECK(hipSetDevice(idx));
    HIP_CHECK(hipStreamCreate(&stream_buf[streamPerAsic * idx]));
    HIP_CHECK(hipStreamCreate(&stream_buf[streamPerAsic * idx + 1]));
    streamMemAllocTest* testObj1 = new streamMemAllocTest(N);
    testObj1->createMempool(hipMemPoolAttrReleaseThreshold, testType, idx);
    tesObjBuf.push_back(testObj1);
    streamMemAllocTest* testObj2 = new streamMemAllocTest(N);
    testObj2->createMempool(hipMemPoolAttrReleaseThreshold, testType, idx);
    tesObjBuf.push_back(testObj2);
  }
  // Queue commands in each device
  for (int idx = 0; idx < num_devices; idx++) {
    HIP_CHECK(hipSetDevice(idx));
    std::thread test1(threadQAsyncCommands, tesObjBuf[streamPerAsic * idx],
                      stream_buf[streamPerAsic * idx], idx);
    std::thread test2(threadQAsyncCommands, tesObjBuf[streamPerAsic * idx + 1],
                      stream_buf[streamPerAsic * idx + 1], idx);
    test1.join();
    test2.join();
  }
  // Wait for the streams
  for (int idx = 0; idx < num_devices; idx++) {
    HIP_CHECK(hipSetDevice(idx));
    HIP_CHECK(hipStreamSynchronize(stream_buf[streamPerAsic * idx]));
    HIP_CHECK(hipStreamSynchronize(stream_buf[streamPerAsic * idx + 1]));
    // verify and validate
    REQUIRE(true == tesObjBuf[streamPerAsic * idx]->validateResult());
    REQUIRE(true == tesObjBuf[streamPerAsic * idx + 1]->validateResult());
  }
  // Deallocate resources in each device
  for (int idx = 0; idx < num_devices; idx++) {
    HIP_CHECK(hipSetDevice(idx));
    // Destroy resources
    tesObjBuf[streamPerAsic * idx]->freeMempool();
    tesObjBuf[streamPerAsic * idx]->freeHostBuf();
    tesObjBuf[streamPerAsic * idx + 1]->freeMempool();
    tesObjBuf[streamPerAsic * idx + 1]->freeHostBuf();
    HIP_CHECK(hipStreamDestroy(stream_buf[streamPerAsic * idx]));
    HIP_CHECK(hipStreamDestroy(stream_buf[streamPerAsic * idx + 1]));
    delete tesObjBuf[streamPerAsic * idx];
    delete tesObjBuf[streamPerAsic * idx + 1];
  }
  delete[] stream_buf;
}
#endif
/**
 * Common multithreaded test for release threshold (default or max).
 * Runs one thread per device that supports memory pools.
 */
static void runMThreadReleaseThresholdTest(enum eTestValue testtype) {
  constexpr int N = 1 << 10;

  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));

  // Determine which devices support memory pools
  std::vector<int> supported_devices;
  supported_devices.reserve(num_devices);
  for (int idx = 0; idx < num_devices; ++idx) {
    int mempool_supported = 0;
    HIP_CHECK(hipDeviceGetAttribute(&mempool_supported,
                                    hipDeviceAttributeMemoryPoolsSupported, idx));
    if (mempool_supported) {
      supported_devices.push_back(idx);
    }
  }
  // If no devices support memory pools, defer to common helper logic
  if (supported_devices.empty()) {
    checkMempoolSupported(0);
    return;
  }

  std::vector<std::thread> tests;
  std::vector<hipStream_t> streams(supported_devices.size());

  // Initialize and create streams only on supported devices
  for (std::size_t i = 0; i < supported_devices.size(); ++i) {
    int device = supported_devices[i];
    HIP_CHECK(hipSetDevice(device));
    HIP_CHECK(hipStreamCreate(&streams[i]));
  }

  // Spawn the test threads for supported devices
  for (std::size_t i = 0; i < supported_devices.size(); ++i) {
    int device = supported_devices[i];
    tests.push_back(
        std::thread(checkMaximumAndDefaultThresholdMultiThreaded, streams[i], N, testtype, device));
  }

  // Wait for all threads to complete
  for (std::thread& t : tests) {
    t.join();
  }

  // Destroy streams on supported devices
  for (std::size_t i = 0; i < supported_devices.size(); ++i) {
    int device = supported_devices[i];
    HIP_CHECK(hipSetDevice(device));
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }

  HIP_CHECK_THREAD_FINALIZE();
}

/**
 * Test Description
 * ------------------------
 *    - Validate memory pool creation, allocation of memory from the
 * memory pool and usage in multithreaded environment.
 * ------------------------
 *    - catch\unit\memory\hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE(Unit_hipMallocFromPoolAsync_MThread_DefaultThresh) {
  runMThreadReleaseThresholdTest(testdefault);
}

TEST_CASE(Unit_hipMallocFromPoolAsync_MThread_MaxThresh) {
  runMThreadReleaseThresholdTest(testMaximum);
}

/**
 * Test Description
 * ------------------------
 *    - Validate memory pool creation in main thread and its usage -
 * device memory allocation, data transfer to and from device and
 * kernel launch from multiple threads.
 * ------------------------
 *    - catch\unit\memory\hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMallocFromPoolAsync_MThread_CommonMpool_DefaultMempool) {
  checkMempoolSupported(0)
      REQUIRE(true == test_hipMallocFromPoolAsync_MThread_CommonMpool(testdefault, true));
}

HIP_TEST_CASE(Unit_hipMallocFromPoolAsync_MThread_CommonMpool_MaxThresh) {
  checkMempoolSupported(0)
      REQUIRE(true == test_hipMallocFromPoolAsync_MThread_CommonMpool(testMaximum, false));
}

/**
 * Test Description
 * ------------------------
 *    - Multiple stream scenario: Create explicit memory pool. Create 3 streams.
 * Allocate device memory and initialize on 1st stream, Invoke kernel to
 * perform operation on 2nd stream and Free the device memory on 3rd stream.
 * Synchronize between stream1, stream2 and stream3 using events.
 * ------------------------
 *    - catch\unit\memory\hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMallocFromPoolAsync_MultStream_Sync) {
  checkMempoolSupported(0) constexpr int N = 1 << 20;
  REQUIRE(true == checkMempoolMultStreamSync(N));
}

/**
 * Test Description
 * ------------------------
 *    - Multiple stream concurrent execution scenario: Create common memory pool.
 * Execute mempool functionality on a user created stream, null stream and
 * hipStreamPerThread concurrently. Wait for all the streams to complete and
 * validate result.
 * ------------------------
 *    - catch\unit\memory\hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMallocFromPoolAsync_MultStream_DefaultStreams) {
  checkMempoolSupported(0) constexpr int N = 1 << 20;
  REQUIRE(true == checkMempoolMultStreamConcurrentExec(N, true));
}

/**
 * Test Description
 * ------------------------
 *    - Multiple stream concurrent execution scenario: Create common memory pool.
 * Execute mempool functionality on multiple user created streams concurrently.
 * Wait for all the streams to complete and validate result.
 * ------------------------
 *    - catch\unit\memory\hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMallocFromPoolAsync_MultStream_UserStreams) {
  checkMempoolSupported(0) constexpr int N = 1 << 20;
  REQUIRE(true == checkMempoolMultStreamConcurrentExec(N, false));
}

/**
 * Test Description
 * ------------------------
 *    - Test to validate mempool functionality when enabling and disabling
 * hipMemPoolReuseFollowEventDependencies attribute.
 * ------------------------
 *    - catch\unit\memory\hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMallocFromPoolAsync_ReuseFollowEventDependencies) {
  checkMempoolSupported(0) constexpr int N = 1 << 20;
  REQUIRE(true == checkReuseFollowEventDepFlag(N, testDisabled));
  REQUIRE(true == checkReuseFollowEventDepFlag(N, testEnabled));
}

/**
 * Test Description
 * ------------------------
 *    - Test to validate mempool functionality when enabling and disabling
 * hipMemPoolReuseAllowOpportunistic attribute.
 * ------------------------
 *    - catch\unit\memory\hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMallocFromPoolAsync_ReuseAllowOpportunistic) {
  checkMempoolSupported(0) constexpr int N = 1 << 20;
  REQUIRE(true == checkReuseAllowOtherFlags(N, hipMemPoolReuseAllowOpportunistic, testDisabled));
  REQUIRE(true == checkReuseAllowOtherFlags(N, hipMemPoolReuseAllowOpportunistic, testEnabled));
}

/**
 * Test Description
 * ------------------------
 *    - Test to validate mempool functionality when enabling and disabling
 * hipMemPoolReuseAllowInternalDependencies attribute.
 * ------------------------
 *    - catch\unit\memory\hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMallocFromPoolAsync_ReuseAllowInternalDependencies) {
  checkMempoolSupported(0) constexpr int N = 1 << 20;
  REQUIRE(true ==
          checkReuseAllowOtherFlags(N, hipMemPoolReuseAllowInternalDependencies, testDisabled));
  REQUIRE(true ==
          checkReuseAllowOtherFlags(N, hipMemPoolReuseAllowInternalDependencies, testEnabled));
}

/**
 * End doxygen group StreamOTest.
 * @}
 */

