/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include <utils.hh>

static constexpr unsigned int N = 2 * 1024 * 1024;
static constexpr size_t NBYTES = N * sizeof(int);

/**
 * Local Function to fill the array with given value
 */
static void fillArr(int* arr, unsigned int size, int value) {
  for (int i = 0; i < size; i++) {
    arr[i] = value;
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test, tests the following scenario with per thread compiler option:-
 *  - Do Memory copy Asynchronously H2D in Legacy stream
 *  - and do Memory copy Asynchronously D2H in Legacy stream.
 *  - The task 2 should wait till the execution of task 1.
 *  - Even it is compiled with stream per thread, since one stream for both
 *  - these should be in sync.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamLegacy_compiler_options.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE(Unit_hipStreamLegacy_WithSptCompilerOption) {
  int* hostArrSrc = new int[N];
  REQUIRE(hostArrSrc != nullptr);
  fillArr(hostArrSrc, N, 1);

  int* devArr = nullptr;
  HIP_CHECK(hipMalloc(&devArr, NBYTES));
  REQUIRE(devArr != nullptr);

  int* hostArrDst = new int[N];
  REQUIRE(hostArrDst != nullptr);
  fillArr(hostArrDst, N, 3);

  HIP_CHECK(hipMemcpyAsync(devArr, hostArrSrc, NBYTES, hipMemcpyHostToDevice, hipStreamLegacy));
  HIP_CHECK(hipMemcpyAsync(hostArrDst, devArr, NBYTES, hipMemcpyDeviceToHost, hipStreamLegacy));
  HIP_CHECK(hipStreamSynchronize(hipStreamLegacy));

  for (int i = 0; i < N; i++) {
    INFO("At index : " << i << " Got value : " << hostArrDst[i] << " Expected value : 1 \n");
    REQUIRE(hostArrDst[i] == 1);
  }

  delete[] hostArrSrc;
  delete[] hostArrDst;
  HIP_CHECK(hipFree(devArr));
}

/*
 * Local helper function to copy data from host to device
 */
static void copyHostToDevice(int* hostArr, int* devArr) {
  HIP_CHECK(hipMemcpyAsync(devArr, hostArr, NBYTES, hipMemcpyHostToDevice, hipStreamLegacy));
}

/*
 * Local helper function to copy data from device to host
 */
static void copyDeviceToHost(int* devArr, int* hostArr) {
  HIP_CHECK(hipMemcpyAsync(hostArr, devArr, NBYTES, hipMemcpyDeviceToHost, hipStreamLegacy));
}

/**
 * Test Description
 * ------------------------
 *  - This test, tests the following scenario with per thread compiler option:-
 *  - Launch two threads,
 *  -  In thread  1  : H -> D Copy
 *  -  In thread  2  : D -> H Copy
 *  - These two threads should run successfully and update the data.
 *  - Note : Joined first thread before launching second thread to avoid,
 *  - the scenario of second thread can be launched before first
 *  - thread while scheduling.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamLegacy_compiler_options.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE(Unit_hipStreamLegacy_TwoThreadsDiffOperationWithSptCompOption) {
  const unsigned int threadsSupported = std::thread::hardware_concurrency();

  if (threadsSupported < 2) {
    HipTest::HIP_SKIP_TEST(
        "Skipping due to machine does't "
        "support two concurrent threads");
    return;
  }

  int* hostArrSrc = new int[N];
  REQUIRE(hostArrSrc != nullptr);
  fillArr(hostArrSrc, N, 50);

  int* devArr = nullptr;
  HIP_CHECK(hipMalloc(&devArr, NBYTES));
  REQUIRE(devArr != nullptr);

  int* hostArrDst = new int[N];
  REQUIRE(hostArrDst != nullptr);
  fillArr(hostArrDst, N, 52);

  std::thread H2D_Thread(copyHostToDevice, hostArrSrc, devArr);
  H2D_Thread.join();
  std::thread D2H_Thread(copyDeviceToHost, devArr, hostArrDst);
  D2H_Thread.join();
  HIP_CHECK(hipStreamSynchronize(hipStreamLegacy));

  for (int i = 0; i < N; i++) {
    INFO("At index : " << i << " Got value : " << hostArrDst[i] << " Expected value : 50 \n");
    REQUIRE(hostArrDst[i] == 50);
  }

  delete[] hostArrSrc;
  delete[] hostArrDst;
  HIP_CHECK(hipFree(devArr));
}
