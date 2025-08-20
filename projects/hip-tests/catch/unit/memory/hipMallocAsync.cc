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

#include <limits>

#pragma clang diagnostic ignored "-Wunused-parameter"

static bool thread_results[NUMBER_OF_THREADS];
static constexpr auto NUM_ELM{1024 * 1024};
static constexpr int streamPerAsic = 2;

/**
 * @addtogroup hipMallocAsync hipMallocAsync
 * @{
 * @ingroup StreamOTest
 * `hipMallocAsync(void** dev_ptr, size_t size, hipStream_t stream)`
 * - Allocates memory with stream ordered semantics
 */

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify proper allocation and stream ordering of hipMallocAsync when one
 * memory allocation is performed.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMallocAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMallocAsync_Basic_OneAlloc") {
  MallocMemPoolAsync_OneAlloc(
      [](void** dev_ptr, size_t size, hipMemPool_t mem_pool, hipStream_t stream) {
        return hipMallocAsync(dev_ptr, size, stream);
      },
      MemPools::dev_default);
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify proper allocation and stream ordering of hipMallocAsync when two
 * memory allocations are performed.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMallocAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMallocAsync_Basic_TwoAllocs") {
  MallocMemPoolAsync_TwoAllocs(
      [](void** dev_ptr, size_t size, hipMemPool_t mem_pool, hipStream_t stream) {
        return hipMallocAsync(dev_ptr, size, stream);
      },
      MemPools::dev_default);
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify that memory allocated with hipMallocAsync can be properly reused.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMallocAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMallocAsync_Basic_Reuse") {
  MallocMemPoolAsync_Reuse([](void** dev_ptr, size_t size, hipMemPool_t mem_pool,
                              hipStream_t stream) { return hipMallocAsync(dev_ptr, size, stream); },
                           MemPools::dev_default);
}


/**
 * Test Description
 * ------------------------
 *  - Test to verify hipMallocAsync behavior with invalid arguments:
 *    -# Nullptr dev_ptr
 *    -# Invalid stream handle
 *    -# Size is max size_t
 *
 * Test source
 * ------------------------
 *  - /unit/memory/hipMallocAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMallocAsync_Negative_Parameters") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));
  checkMempoolSupported(0)

      int* p = nullptr;
  size_t max_size = std::numeric_limits<size_t>::max();
  size_t alloc_size = 1024;
  MemPoolGuard mempool(MemPools::dev_default, device_id);
  StreamGuard stream(Streams::created);

  SECTION("dev_ptr is nullptr") {
    HIP_CHECK_ERROR(hipMallocAsync(nullptr, alloc_size, stream.stream()), hipErrorInvalidValue);
  }

  SECTION("Size is max size_t") {
    HIP_CHECK_ERROR(hipMallocAsync(reinterpret_cast<void**>(&p), max_size, stream.stream()),
                    hipErrorOutOfMemory);
  }
}

/**
 * Common function to allocate memory using hipMallocAsync API through a stream,
 * launch kernel and perform vectorADD and validate results. Free memory using
 * hipFreeAsync.
 */
static bool checkMallocAsync(hipStream_t stream) {
  streamMemAllocTest testObj(NUM_ELM);
  // Create host buffer with test data.
  testObj.createHostBufferWithData();
  // Allocate device memory.
  testObj.allocFromDefMempool(stream);
  // Transfer data to it asyncronously on stream.
  testObj.transferToMempool(stream);
  // Execute kernel and transfer result back to host asynchronously on stream.
  testObj.runKernel(stream);
  testObj.transferFromMempool(stream);
  // Free Buffer Asynchronously on stream.
  testObj.freeDevBuf(stream);
  HIP_CHECK(hipStreamSynchronize(stream));
  // verify and validate
  REQUIRE(true == testObj.validateResult());
  // Destroy resources
  testObj.freeHostBuf();
  return true;
}
/**
 * Test Description
 * ------------------------
 *    - Test case to perform basic scenario.
 * ------------------------
 *    - catch\unit\memory\hipMallocAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMallocAsync_basic") {
  checkMempoolSupported(0)
      // create a stream
      hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  REQUIRE(true == checkMallocAsync(stream));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *    - Test case to perform multi stream, allocate memory using
 * hipMallocAsync API for a stream1 and stream2, launch kernel and
 * perform vectorADD, synchronize stream1 and stream2 and validate
 * results. Free memory using hipFreeAsync.
 * ------------------------
 *    - catch\unit\memory\hipMallocAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMallocAsync_Multistream_Concurrent") {
  checkMempoolSupported(0) streamMemAllocTest testObj1(NUM_ELM), testObj2(NUM_ELM);
  // create multiple streams
  hipStream_t stream1, stream2;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  // Create host buffer with test data.
  testObj1.createHostBufferWithData();
  testObj2.createHostBufferWithData();
  // Allocate device memory and transfer data to it asyncronously on streams.
  testObj1.allocFromDefMempool(stream1);
  testObj2.allocFromDefMempool(stream2);
  testObj1.transferToMempool(stream1);
  testObj2.transferToMempool(stream2);
  // Execute kernel and transfer result back to host asynchronously on streams.
  testObj1.runKernel(stream1);
  testObj2.runKernel(stream2);
  testObj1.transferFromMempool(stream1);
  testObj2.transferFromMempool(stream2);
  // Free Buffer Asynchronously on streams.
  testObj1.freeDevBuf(stream1);
  testObj2.freeDevBuf(stream2);
  // synchronize both stream1 and stream2
  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipStreamSynchronize(stream2));
  // verify and validate
  REQUIRE(true == testObj1.validateResult());
  REQUIRE(true == testObj2.validateResult());
  // Destroy resources
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  testObj1.freeHostBuf();
  testObj2.freeHostBuf();
}

/**
 * Test Description
 * ------------------------
 *    - Allocate memory using hipMallocAsync API through stream1 and record event1,
 * allocate event1 to stream2 and put stream2 to wait, launch kernel through
 * stream2 and perform vectorADD and validate results. Free memory using hipFreeAsync.
 * ------------------------
 *    - catch\unit\memory\hipMallocAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMallocAsync_StreamEvent_CrissCross") {
  checkMempoolSupported(0) streamMemAllocTest testObj1(NUM_ELM), testObj2(NUM_ELM);
  // create two streams.
  hipStream_t stream1, stream2;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  // create an event
  hipEvent_t event1 = nullptr, event2 = nullptr;
  HIP_CHECK(hipEventCreate(&event1));
  HIP_CHECK(hipEventCreate(&event2));
  // Create host buffer with test data.
  testObj1.createHostBufferWithData();
  testObj2.createHostBufferWithData();
  // Allocate device memory and transfer data to it asyncronously on streams.
  testObj1.allocFromDefMempool(stream1);
  testObj2.allocFromDefMempool(stream2);
  testObj1.transferToMempool(stream1);
  testObj2.transferToMempool(stream2);
  // create event record
  HIP_CHECK(hipEventRecord(event1, stream1));
  HIP_CHECK(hipEventRecord(event2, stream2));
  HIP_CHECK(hipStreamWaitEvent(stream2, event1, 0));
  HIP_CHECK(hipStreamWaitEvent(stream1, event2, 0));
  // Execute kernel and transfer result back to host asynchronously on streams.
  testObj1.runKernel(stream2);
  testObj2.runKernel(stream1);
  testObj1.transferFromMempool(stream2);
  testObj2.transferFromMempool(stream1);
  // Free Buffer Asynchronously on streams.
  testObj1.freeDevBuf(stream2);
  testObj2.freeDevBuf(stream1);
  // Wait for stream2.
  HIP_CHECK(hipStreamSynchronize(stream2));
  HIP_CHECK(hipStreamSynchronize(stream1));
  // verify and validate
  REQUIRE(true == testObj1.validateResult());
  REQUIRE(true == testObj2.validateResult());
  // Destroy resources
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipEventDestroy(event2));
  testObj1.freeHostBuf();
  testObj2.freeHostBuf();
}

/**
 * Test Description
 * ------------------------
 *    - Test case to perform multi device scenario, get number of devices available
 * and call checkMallocAsync function for each device available.
 * ------------------------
 *    - catch\unit\memory\hipMallocAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMallocAsync_Multidevice") {
  int num_devices;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  for (int i = 0; i < num_devices; i++) {
    checkMempoolSupported(i) HIP_CHECK(hipSetDevice(i));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    REQUIRE(true == checkMallocAsync(stream));
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

/**
 * Test Description
 * ------------------------
 *    - Queue the following commands hipMallocAsync, transfer data
 * to it asynchrously, launch Kernel, transfer results back to host
 * asynchronously and free buffer async in streams across all GPUs.
 * The execution in of the queued commands must happen concurrently.
 * ------------------------
 *    - catch\unit\memory\hipMallocAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
#if HT_AMD
static void threadQAsyncCommands(streamMemAllocTest* testObj, hipStream_t strm, int idx) {
  HIP_CHECK(hipSetDevice(idx));
  // Create host buffer with test data.
  testObj->createHostBufferWithData();
  // Allocate device memory and transfer data to it asyncronously on stream.
  testObj->allocFromDefMempool(strm);
  testObj->transferToMempool(strm);
  // Execute kernel and transfer result back to host asynchronously on stream.
  testObj->runKernel(strm);
  testObj->transferFromMempool(strm);
  // Free Buffer Asynchronously on stream.
  testObj->freeDevBuf(strm);
}

TEST_CASE("Unit_hipMallocAsync_Multidevice_Concurrent") {
  int num_devices;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  checkIfMultiDev(num_devices) hipStream_t* stream_buf = new hipStream_t[num_devices];
  std::vector<streamMemAllocTest*> tesObjBuf;
  // Allocate resources in each device
  for (int idx = 0; idx < num_devices; idx++) {
    checkMempoolSupported(idx) HIP_CHECK(hipSetDevice(idx));
    HIP_CHECK(hipStreamCreate(&stream_buf[idx]));
    streamMemAllocTest* testObj = new streamMemAllocTest(NUM_ELM);
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
    tesObjBuf[idx]->freeHostBuf();
    HIP_CHECK(hipStreamDestroy(stream_buf[idx]));
    delete tesObjBuf[idx];
  }
  delete[] stream_buf;
}

/**
 * Test Description
 * ------------------------
 *    - Queue the following commands hipMallocAsync, transfer data
 * to it asynchrously, launch Kernel, transfer results back to host
 * asynchronously and free buffer async in streams across all GPUs
 * using multiple streams per GPU.
 * ------------------------
 *    - catch\unit\memory\hipMallocAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMallocAsync_Multidevice_MultiStream") {
  int num_devices;
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
    streamMemAllocTest* testObj1 = new streamMemAllocTest(NUM_ELM);
    tesObjBuf.push_back(testObj1);
    streamMemAllocTest* testObj2 = new streamMemAllocTest(NUM_ELM);
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
    tesObjBuf[streamPerAsic * idx]->freeHostBuf();
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
 * Test Description
 * ------------------------
 *    - Assign device memory using hipMalloc, launch kernel and perform
 * vector square and validate. Free memory using hipFreeAsync API.
 * ------------------------
 *    - catch\unit\memory\hipMallocAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMallocAsync_ByUsinghipMalloc") {
  checkMempoolSupported(0) size_t byte_size = NUM_ELM * sizeof(float);
  // create a stream
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  float *A_h, *C_h;
  float *A_d, *C_d;
  // assign memory to host pointers
  A_h = reinterpret_cast<float*>(malloc(byte_size));
  C_h = reinterpret_cast<float*>(malloc(byte_size));
  // set data to host
  for (int i = 0; i < NUM_ELM; i++) {
    A_h[i] = 7.0f;
    C_h[i] = 0;
  }
  // assign memory to device pointers
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_d), byte_size));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&C_d), byte_size));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, byte_size, hipMemcpyHostToDevice, stream));
  hipLaunchKernelGGL(HipTest::vector_square, dim3(NUM_ELM / THREADS_PER_BLOCK),
                     dim3(THREADS_PER_BLOCK), 0, stream, static_cast<const float*>(A_d), C_d,
                     NUM_ELM);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, byte_size, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(A_d), stream));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(C_d), stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  // verify and validate
  for (int i = 0; i < NUM_ELM; i++) {
    REQUIRE(C_h[i] == (A_h[i] * A_h[i]));
  }
  HIP_CHECK(hipStreamDestroy(stream));
  free(A_h);
  free(C_h);
}

/**
 * Test Description
 * ------------------------
 *    - Assign device memory using hipMallocAsync, launch kernel and perform
 * vector square and validate. Free memory using hipFree API.
 * ------------------------
 *    - catch\unit\memory\hipMallocAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMallocAsync_ByUsinghipFree") {
  size_t byte_size = NUM_ELM * sizeof(float);
  checkMempoolSupported(0)
      // create a stream
      hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  float *A_h, *C_h;
  float *A_d, *C_d;
  // assign memory to host pointers
  A_h = reinterpret_cast<float*>(malloc(byte_size));
  C_h = reinterpret_cast<float*>(malloc(byte_size));
  // set data to host
  for (int i = 0; i < NUM_ELM; i++) {
    A_h[i] = 5.0f;
    C_h[i] = 0;
  }
  // assign memory to device pointers
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&A_d), byte_size, stream));
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&C_d), byte_size, stream));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, byte_size, hipMemcpyHostToDevice, stream));
  hipLaunchKernelGGL(HipTest::vector_square, dim3(NUM_ELM / THREADS_PER_BLOCK),
                     dim3(THREADS_PER_BLOCK), 0, stream, static_cast<const float*>(A_d), C_d,
                     NUM_ELM);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, byte_size, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(A_d)));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(C_d)));
  // verify and validate
  for (int i = 0; i < NUM_ELM; i++) {
    REQUIRE(C_h[i] == (A_h[i] * A_h[i]));
  }
  HIP_CHECK(hipStreamDestroy(stream));
  free(A_h);
  free(C_h);
}

/**
 * Test Description
 * ------------------------
 *    - Test case to check hipMallocAsync allocation and usage in multiple
 * threads. Each thread will use a local stream.
 * ------------------------
 *    - catch\unit\memory\hipMallocAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
static void threadTestLocalStream(int threadNum) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  thread_results[threadNum] = checkMallocAsync(stream);
  HIP_CHECK(hipStreamDestroy(stream));
}

static bool testhipMallocAsyncMThreadLocalStrm() {
  std::vector<std::thread> tests;
  // Spawn the test threads
  for (int idx = 0; idx < NUMBER_OF_THREADS; idx++) {
    thread_results[idx] = false;
    tests.push_back(std::thread(threadTestLocalStream, idx));
  }
  // Wait for all threads to complete
  for (std::thread& t : tests) {
    t.join();
  }
  // Wait for thread
  bool status = true;
  for (int idx = 0; idx < NUMBER_OF_THREADS; idx++) {
    status = status & thread_results[idx];
  }
  return status;
}

TEST_CASE("Unit_hipMallocAsync_MThread_ThreadLocalStream") {
  checkMempoolSupported(0) REQUIRE(true == testhipMallocAsyncMThreadLocalStrm());
}

/**
 * Test Description
 * ------------------------
 *    - Test case to check hipMallocAsync allocation and usage in multiple
 * threads. Threads will use a common shared stream.
 * ------------------------
 *    - catch\unit\memory\hipMallocAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
static void threadTestCommonStream(int threadNum, hipStream_t stream) {
  thread_results[threadNum] = checkMallocAsync(stream);
}

static bool testhipMallocAsyncMThreadLocalStrm(hipStream_t stream) {
  std::vector<std::thread> tests;
  // Spawn the test threads
  for (int idx = 0; idx < NUMBER_OF_THREADS; idx++) {
    thread_results[idx] = false;
    tests.push_back(std::thread(threadTestCommonStream, idx, stream));
  }
  // Wait for all threads to complete
  for (std::thread& t : tests) {
    t.join();
  }
  // Wait for thread
  bool status = true;
  for (int idx = 0; idx < NUMBER_OF_THREADS; idx++) {
    status = status & thread_results[idx];
  }
  return status;
}

TEST_CASE("Unit_hipMallocAsync_MThread_ThreadSharedStream") {
  checkMempoolSupported(0) hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  REQUIRE(true == testhipMallocAsyncMThreadLocalStrm(stream));
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * Test Description
 * ------------------------
 *    - Test case to check MallocAsync functionality on user created stream,
 * null stream and hipstreamperthread concurrently. launch kernel and wait
 * for all streams to complete and validate results.
 * ------------------------
 *    - catch\unit\memory\hipMallocAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMallocAsync_DefaultStreams_Concurrent") {
  checkMempoolSupported(0) streamMemAllocTest testObj[3] = {
      streamMemAllocTest(NUM_ELM), streamMemAllocTest(NUM_ELM), streamMemAllocTest(NUM_ELM)};
  // create multiple streams
  hipStream_t stream[3];
  HIP_CHECK(hipStreamCreate(&stream[0]));
  stream[1] = 0;  // Null stream
  stream[2] = hipStreamPerThread;
  // Queue operations on the 3 streams
  for (int idx = 0; idx < 3; idx++) {
    // Create host buffer with test data.
    testObj[idx].createHostBufferWithData();
    // Allocate device memory and transfer data to it asyncronously on stream.
    testObj[idx].allocFromDefMempool(stream[idx]);
    testObj[idx].transferToMempool(stream[idx]);
    // Execute kernel and transfer result back to host asynchronously on stream.
    testObj[idx].runKernel(stream[idx]);
    testObj[idx].transferFromMempool(stream[idx]);
    // Free Buffer Asynchronously on stream.
    testObj[idx].freeDevBuf(stream[idx]);
  }
  // Wait for the 3 streams
  for (int idx = 0; idx < 3; idx++) {
    HIP_CHECK(hipStreamSynchronize(stream[idx]));
    // verify and validate
    REQUIRE(true == testObj[idx].validateResult());
    // Destroy resources
    testObj[idx].freeHostBuf();
  }
  HIP_CHECK(hipStreamDestroy(stream[0]));
}

/**
 * End doxygen group StreamOTest.
 * @}
 */
