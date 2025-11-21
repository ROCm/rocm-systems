/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*
Testcase Scenarios :
1) Negative tests for hipStreamGetId.
2) Basic Positive scenario for hipStreamGetId
*/

#include <hip_test_common.hh>

#if __linux__
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

std::vector<unsigned long long> streamIds;
std::mutex mutex;

/**
 * Helper function to check all the StreamId's in given list unique or not
 */
bool hasUniqueStreamIds(const std::vector<unsigned long long>& streamIds) {
  std::unordered_set<unsigned long long> seenElements;
  for (unsigned long long element : streamIds) {
    if (seenElements.count(element)) {
      return false;
    }
    seenElements.insert(element);
  }
  return true;
}

/**
 *  @brief Pass uninitialized stream and id as nullptr to check if the API behaves as expected.
 */
TEST_CASE("Unit_hipStreamGetId_Negative") {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  SECTION("Null Pointer") {
    HIP_CHECK_ERROR(hipStreamGetId(stream, nullptr), hipErrorInvalidValue);
  }
}

/**
 *  @brief Pass null stream, legacy stream and streamperthread, check the API behaves as expected.
 *  Also, check the stream id generated is not same for any two streams.
 */
TEST_CASE("Unit_hipStreamGetId_Basic") {
  hipStream_t stream1, stream2;
  unsigned long long id1, id2, id3, id4, id5;
  SECTION("Unique Stream Id") {
    HIP_CHECK(hipStreamCreate(&stream1));
    HIP_CHECK(hipStreamCreate(&stream2));
    HIP_CHECK(hipStreamGetId(stream1, &id1));
    HIP_CHECK(hipStreamGetId(stream2, &id2));
    HIP_CHECK(hipStreamDestroy(stream1));
    HIP_CHECK(hipStreamDestroy(stream2));
    REQUIRE(id1 != id2);
  }
  SECTION("Null and legacy stream") {
    HIP_CHECK(hipStreamGetId(nullptr, &id3));
    HIP_CHECK(hipStreamGetId(hipStreamLegacy, &id4));
    REQUIRE(id3 == id4);
  }
  SECTION("StreamPerThread") {
    HIP_CHECK(hipStreamGetId(hipStreamPerThread, &id5));
    REQUIRE(id5);
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks the behavior of hipStreamGetId
 *  - with hipStreamCreateWithFlags and hipStreamCreateWithPriority API's.
 * Test source
 * ------------------------
 *  - unit/device/hipStreamGetId.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipStreamGetId_WithDifferentStreamCreateAPIs") {
  hipStream_t stream_1 = nullptr, stream_2 = nullptr;
  unsigned long long streamId_1 = 0, streamId_2 = 0;

  SECTION("With hipStreamCreateWithFlags") {
    HIP_CHECK(hipStreamCreateWithFlags(&stream_1, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&stream_2, hipStreamNonBlocking));

    HIP_CHECK(hipStreamGetId(stream_1, &streamId_1));
    HIP_CHECK(hipStreamGetId(stream_2, &streamId_2));

    REQUIRE(streamId_1 != streamId_2);
  }

  SECTION("With hipStreamCreateWithPriority") {
    int priority_low{};
    int priority_high{};
    HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));

    int priority = priority_high;
    HIP_CHECK(hipStreamCreateWithPriority(&stream_1, hipStreamDefault, priority));
    HIP_CHECK(hipStreamCreateWithPriority(&stream_2, hipStreamDefault, priority));

    HIP_CHECK(hipStreamGetId(stream_1, &streamId_1));
    HIP_CHECK(hipStreamGetId(stream_2, &streamId_2));

    REQUIRE(streamId_1 != streamId_2);
  }
  HIP_CHECK(hipStreamDestroy(stream_1));
  HIP_CHECK(hipStreamDestroy(stream_2));
}

/**
 * Helper function to get hipStreamGetId.
 * This function used in multi threaded scenario.
 */
void launchFunction() {
  hipStream_t stream;
  unsigned long long streamId;
  HIP_CHECK_THREAD(hipStreamCreate(&stream));

  HIP_CHECK_THREAD(hipStreamGetId(stream, &streamId));

  mutex.lock();
  streamIds.push_back(streamId);
  mutex.unlock();

  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks the behavior of hipStreamGetId
 *  - in multiple threads.
 * Test source
 * ------------------------
 *  - unit/device/hipStreamGetId.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipStreamGetId_MultipleThreads") {
  const unsigned int threadsSupported = std::thread::hardware_concurrency();
  INFO("Number of threads supported : " << threadsSupported);

  std::vector<std::thread> threads;
  streamIds.clear();
  for (int t = 0; t < threadsSupported; t++) {
    threads.push_back(std::thread(launchFunction));
  }

  for (int t = 0; (t < threadsSupported) && (t < threads.size()); t++) {
    threads[t].join();
  }

  HIP_CHECK_THREAD_FINALIZE();

  REQUIRE(hasUniqueStreamIds(streamIds) == true);
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks the behavior of hipStreamGetId
 *  - in multiple devices.
 * Test source
 * ------------------------
 *  - unit/device/hipStreamGetId.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipStreamGetId_MultiDevice") {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 2");
    return;
  }

  std::vector<unsigned long long> streamIds;

  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    HIP_CHECK(hipSetDevice(deviceId));

    hipStream_t stream;
    unsigned long long streamId;
    HIP_CHECK(hipStreamCreate(&stream));

    HIP_CHECK(hipStreamGetId(stream, &streamId));
    streamIds.push_back(streamId);

    HIP_CHECK(hipStreamDestroy(stream));
  }

  REQUIRE(hasUniqueStreamIds(streamIds) == true);
}

#if __linux__
/**
 * Test Description
 * ------------------------
 *  - This test case checks the behavior of hipStreamGetId
 *  - in multi process (In child and parent process)
 * Test source
 * ------------------------
 *  - unit/device/hipStreamGetId.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipStreamGetId_MultiProcess") {
  auto pid = fork();
  REQUIRE(pid >= 0);

  if (pid != 0) {  // Parent process
    hipStream_t stream_1 = nullptr, stream_2 = nullptr;
    unsigned long long streamId_1 = 0, streamId_2 = 0;

    HIP_CHECK(hipStreamCreate(&stream_1));
    HIP_CHECK(hipStreamCreate(&stream_2));

    HIP_CHECK(hipStreamGetId(stream_1, &streamId_1));
    HIP_CHECK(hipStreamGetId(stream_2, &streamId_2));

    REQUIRE(streamId_1 != streamId_2);
    HIP_CHECK(hipStreamDestroy(stream_1));
    HIP_CHECK(hipStreamDestroy(stream_2));

    int status;
    REQUIRE(wait(&status) >= 0);
  } else {  // Child process
    hipStream_t stream_1 = nullptr, stream_2 = nullptr;
    unsigned long long streamId_1 = 0, streamId_2 = 0;

    HIP_CHECK(hipStreamCreate(&stream_1));
    HIP_CHECK(hipStreamCreate(&stream_2));

    HIP_CHECK(hipStreamGetId(stream_1, &streamId_1));
    HIP_CHECK(hipStreamGetId(stream_2, &streamId_2));

    REQUIRE(streamId_1 != streamId_2);
    HIP_CHECK(hipStreamDestroy(stream_1));
    HIP_CHECK(hipStreamDestroy(stream_2));

    exit(0);
  }
}

#endif
