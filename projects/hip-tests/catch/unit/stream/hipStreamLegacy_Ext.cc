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
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include <utils.hh>
#include <hip_test_process.hh>

static constexpr int N = 2 * 1024 * 1024;
static constexpr size_t NBYTES = N * sizeof(int);

/**
 * Local Function to fill the array with given value
 */
static void fillHostArray(int* arr, int size, int value) {
  for (int i = 0; i < size; i++) {
    arr[i] = value;
  }
}

/**
 * Kernel to fill the array with given value
 */
static __global__ void fillArray(int* arr, int size, int value) {
  for (int i = 0; i < size; i++) {
    arr[i] = value;
  }
}

/**
 * Local Function to fill the device array with given value
 */
static void fillDeviceArray(int* arr, int size, int value) {
  fillArray<<<1, 1>>>(arr, size, value);
}

/**
 * In addOneKernel function, all elements of the array a increased by 1
 */
static __global__ void addOneKernel(int* a, int size) {
  int offset = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = offset; i < size; i += stride) {
    a[i] += 1;
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test case tests the following scenario:-
 *  - Do Memory copy Asynchronously H2D in user defined blocking stream,
 *  - do Memory copy Asynchronously D2H in Legacy stream.
 *  - The task 2 which is in legacy stream should wait till the
 *  - task 1 completes its execution which is in userdefined stream.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamLegacy_Ext.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipStreamLegacy_WithBlockingStream") {
  int* hostArrSrc = new int[N];
  REQUIRE(hostArrSrc != nullptr);
  fillHostArray(hostArrSrc, N, 1);

  int* devArr = nullptr;
  HIP_CHECK(hipMalloc(&devArr, NBYTES));
  REQUIRE(devArr != nullptr);
  fillDeviceArray(devArr, N, 2);

  int* hostArrDst = new int[N];
  REQUIRE(hostArrDst != nullptr);
  fillHostArray(hostArrDst, N, 3);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamDefault));

  HIP_CHECK(hipMemcpyAsync(devArr, hostArrSrc, NBYTES, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(hostArrDst, devArr, NBYTES, hipMemcpyDeviceToHost, hipStreamLegacy));
  HIP_CHECK(hipStreamSynchronize(hipStreamLegacy));

  for (int i = 0; i < N; i++) {
    INFO("At index : " << i << " Got value : " << hostArrDst[i] << " Expected value : 1 \n");
    REQUIRE(hostArrDst[i] == 1);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  delete[] hostArrSrc;
  delete[] hostArrDst;
  HIP_CHECK(hipFree(devArr));
}

/**
 * Local Function to perform the below operations:-
 * Do Memory copy Asynchronously H2D in Legacy stream
 * and then do Memory copy Asynchronously D2H in Legacy stream.
 * Task 2 should wait till the execution of task 1.
 */
void launchFunction(hipStream_t stream) {
  int* hostArrSrc = new int[N];
  REQUIRE_THREAD(hostArrSrc != nullptr);
  fillHostArray(hostArrSrc, N, 5);

  int* devArr = nullptr;
  HIP_CHECK_THREAD(hipMalloc(&devArr, NBYTES));
  REQUIRE_THREAD(devArr != nullptr);
  fillDeviceArray(devArr, N, 6);

  int* hostArrDst = new int[N];
  REQUIRE_THREAD(hostArrDst != nullptr);
  fillHostArray(hostArrDst, N, 7);

  HIP_CHECK_THREAD(hipMemcpyAsync(devArr, hostArrSrc, NBYTES, hipMemcpyHostToDevice, stream));
  HIP_CHECK_THREAD(hipMemcpyAsync(hostArrDst, devArr, NBYTES, hipMemcpyDeviceToHost, stream));
  HIP_CHECK_THREAD(hipStreamSynchronize(hipStreamLegacy));

  for (int i = 0; i < N; i++) {
    if (hostArrDst[i] != 5) {
      std::cout << "At index : " << i << " Got value : " << hostArrDst[i]
                << " Expected value : 5 \n"
                << std::endl;
      REQUIRE_THREAD(false);
    }
  }

  delete[] hostArrSrc;
  delete[] hostArrDst;
  HIP_CHECK_THREAD(hipFree(devArr));
}

/**
 * Test Description
 * ------------------------
 *  - This test case tests hipStreamLegacy in multi threaded scenario:-
 *  - All threads should launch successfully and run independently
 *  - and uses the same legacy stream.
 * ------------------------
 *  - unit/stream/hipStreamLegacy_Ext.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipStreamLegacy_MultipleThreads") {
  const unsigned int threadsSupported = std::thread::hardware_concurrency();
  const int numberOfThreads = (threadsSupported >= 10) ? 10 : threadsSupported;

  std::vector<std::thread> threads;
  for (int t = 0; t < numberOfThreads; t++) {
    threads.push_back(std::thread(launchFunction, hipStreamLegacy));
  }

  for (int t = 0; (t < numberOfThreads) && (t < threads.size()); t++) {
    threads[t].join();
  }
  HIP_CHECK_THREAD_FINALIZE();
}

/**
 * Test Description
 * ------------------------
 *  - Pass the hipStreamLegacy to hipStreamBeginCapture() api
 *  - and the api should return hipErrorStreamCaptureUnsupported.
 * ------------------------
 *  - unit/stream/hipStreamLegacy_Ext.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipStreamLegacy_NegetiveCase") {
  hipStream_t stream = hipStreamLegacy;
  REQUIRE(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal) ==
          hipErrorStreamCaptureUnsupported);
}

/**
 * Test Description
 * ------------------------
 *  - This test case tests the following scenario:-
 *  - Do Memory copy Asynchronously H2D in user defined Non-blocking stream
 *  - and do Memory copy Asynchronously D2H in Legacy stream.
 *  - The task 2 which is in legacy stream should not wait till the
 *  - task 1 completes its execution, two streams should run concurrently.
 *  - And the host thread should wait for the task 2 which is in legacy stream.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamLegacy_Ext.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipStreamLegacy_WithNonBlockingStream") {
  int* hostArrSrc = new int[N];
  REQUIRE(hostArrSrc != nullptr);
  fillHostArray(hostArrSrc, N, 10);

  int* devArr = nullptr;
  HIP_CHECK(hipMalloc(&devArr, NBYTES));
  REQUIRE(devArr != nullptr);
  fillDeviceArray(devArr, N, 11);

  int* hostArrDst = new int[N];
  REQUIRE(hostArrDst != nullptr);
  fillHostArray(hostArrDst, N, 12);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  HIP_CHECK(hipMemcpyAsync(devArr, hostArrSrc, NBYTES, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(hostArrDst, devArr, NBYTES, hipMemcpyDeviceToHost, hipStreamLegacy));
  HIP_CHECK(hipStreamSynchronize(hipStreamLegacy));

  for (int i = 0; i < N; i++) {
    INFO("At index : " << i << " Got value : " << hostArrDst[i] << " Expected value : 10 or 11 \n");
    REQUIRE(((hostArrDst[i] == 10) || (hostArrDst[i] == 11)));
  }

  HIP_CHECK(hipStreamDestroy(stream));
  delete[] hostArrSrc;
  delete[] hostArrDst;
  HIP_CHECK(hipFree(devArr));
}

/**
 * Test Description
 * ------------------------
 *  - This test case tests the following scenario:-
 *  - Do Memory copy Asynchronously H2D using hipStreamPerThread
 *  - and do Memory copy Asynchronously D2H using hipStreamLegacy.
 *  - The task 2 which is in legacy stream should wait till the
 *  - task 1 completes its execution which is in hipStreamPerThread.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamLegacy_Ext.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipStreamLegacy_WithStreamPerThread") {
  int* hostArrSrc = new int[N];
  REQUIRE(hostArrSrc != nullptr);
  fillHostArray(hostArrSrc, N, 15);

  int* devArr = nullptr;
  HIP_CHECK(hipMalloc(&devArr, NBYTES));
  REQUIRE(devArr != nullptr);
  fillDeviceArray(devArr, N, 16);

  int* hostArrDst = new int[N];
  REQUIRE(hostArrDst != nullptr);

  HIP_CHECK(hipMemcpyAsync(devArr, hostArrSrc, NBYTES, hipMemcpyHostToDevice, hipStreamPerThread));
  HIP_CHECK(hipMemcpyAsync(hostArrDst, devArr, NBYTES, hipMemcpyDeviceToHost, hipStreamLegacy));
  HIP_CHECK(hipStreamSynchronize(hipStreamLegacy));

  for (int i = 0; i < N; i++) {
    INFO("At index : " << i << " Got value : " << hostArrDst[i] << " Expected value : 15 \n");
    REQUIRE(hostArrDst[i] == 15);
  }

  delete[] hostArrSrc;
  delete[] hostArrDst;
  HIP_CHECK(hipFree(devArr));
}

/**
 * Test Description
 * ------------------------
 *  - This test case tests the following scenario with all available devices :-
 *  - Do Memory copy Asynchronously H2D in Legacy stream
 *  - and do Memory copy Asynchronously D2H in Legacy stream.
 *  - The task 2 which is in legacy stream should wait till the
 *  - task 1 completes its execution, in all the devices.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamLegacy_Ext.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipStreamLegacy_MultiDevice") {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 2");
    return;
  }

  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    HIP_CHECK(hipSetDevice(deviceId));

    int* hostArrSrc = new int[N];
    REQUIRE(hostArrSrc != nullptr);
    fillHostArray(hostArrSrc, N, 20);

    int* devArr = nullptr;
    HIP_CHECK(hipMalloc(&devArr, NBYTES));
    REQUIRE(devArr != nullptr);
    fillDeviceArray(devArr, N, 21);

    int* hostArrDst = new int[N];
    REQUIRE(hostArrDst != nullptr);
    fillHostArray(hostArrDst, N, 22);

    HIP_CHECK(hipMemcpyAsync(devArr, hostArrSrc, NBYTES, hipMemcpyHostToDevice, hipStreamLegacy));
    HIP_CHECK(hipMemcpyAsync(hostArrDst, devArr, NBYTES, hipMemcpyDeviceToHost, hipStreamLegacy));
    HIP_CHECK(hipStreamSynchronize(hipStreamLegacy));

    for (int i = 0; i < N; i++) {
      INFO("At index : " << i << " Got value : " << hostArrDst[i]
                         << " Expected value : 20 "
                            " For deviceId : "
                         << deviceId << "\n");
      REQUIRE(hostArrDst[i] == 20);
    }

    delete[] hostArrSrc;
    delete[] hostArrDst;
    HIP_CHECK(hipFree(devArr));
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test case testing the hipStreamLegacy with hipMemcpyAsync
 *  - in all the ways like H2H, H2D, D2D, D2H, also with the hipMemcpyDefault.
 *  - All the operations should success.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamLegacy_Ext.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipStreamLegacy_H2H_H2D_D2D_D2H_Default") {
  int* hostArr1 = new int[N];
  REQUIRE(hostArr1 != nullptr);
  fillHostArray(hostArr1, N, 30);

  int* hostArr2 = new int[N];
  REQUIRE(hostArr2 != nullptr);
  fillHostArray(hostArr2, N, 31);

  int* devArr1 = nullptr;
  HIP_CHECK(hipMalloc(&devArr1, NBYTES));
  REQUIRE(devArr1 != nullptr);
  fillDeviceArray(devArr1, N, 32);

  int* devArr2 = nullptr;
  HIP_CHECK(hipMalloc(&devArr2, NBYTES));
  REQUIRE(devArr2 != nullptr);
  fillDeviceArray(devArr2, N, 33);

  int* hostArr3 = new int[N];
  REQUIRE(hostArr3 != nullptr);
  fillHostArray(hostArr3, N, 34);

  int* hostArr4 = new int[N];
  REQUIRE(hostArr4 != nullptr);
  fillHostArray(hostArr4, N, 35);

  HIP_CHECK(hipMemcpyAsync(hostArr2, hostArr1, NBYTES, hipMemcpyHostToHost, hipStreamLegacy));
  HIP_CHECK(hipMemcpyAsync(devArr1, hostArr2, NBYTES, hipMemcpyHostToDevice, hipStreamLegacy));
  HIP_CHECK(hipMemcpyAsync(devArr2, devArr1, NBYTES, hipMemcpyDeviceToDevice, hipStreamLegacy));
  HIP_CHECK(hipMemcpyAsync(hostArr3, devArr2, NBYTES, hipMemcpyDeviceToHost, hipStreamLegacy));
  HIP_CHECK(hipMemcpyAsync(hostArr4, hostArr3, NBYTES, hipMemcpyDefault, hipStreamLegacy));
  HIP_CHECK(hipStreamSynchronize(hipStreamLegacy));

  for (int i = 0; i < N; i++) {
    INFO("At index : " << i << " Got value : " << hostArr4[i] << " Expected value : 30 \n");
    REQUIRE(hostArr4[i] == 30);
  }

  delete[] hostArr1;
  delete[] hostArr2;
  HIP_CHECK(hipFree(devArr1));
  HIP_CHECK(hipFree(devArr2));
  delete[] hostArr3;
  delete[] hostArr4;
}

/**
 * Test Description
 * ------------------------
 *  - This test case tests the following scenario with two devices:-
 *  -   h1Dev0  ->  d1Dev0  : memcpy from host to device 0
 *  -   d1Dev0  ->  d1Dev1  : memcpy from device 0 to device 1
 *  -   d1Dev1  ->  d2Dev1  : memcpy from device to device with in device 1
 *  -   d2Dev1  ->  d2Dev0  : memcpy from device 1 to device 0
 *  -   d2Dev0  ->  h2Dev0  : memcpy from device 0 to host
 *  - The opeations in device 0 and device 1 should run concurrently.
 *  - Multiple opeations in multiple devices should run without
 *  - any conflicts.
 *  - And the final host array should have value other than 45,
 *  - since host thread waits.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamLegacy_Ext.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipStreamLegacy_MultiDeviceMultiOperation") {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 2");
    return;
  }

  int currentDevice = 0;
  int peerDevice = 1;

  // Set arrays in device 0
  HIP_CHECK(hipSetDevice(currentDevice));

  int* h1Dev0 = new int[N];
  REQUIRE(h1Dev0 != nullptr);
  fillHostArray(h1Dev0, N, 40);

  int* d1Dev0 = nullptr;
  HIP_CHECK(hipMalloc(&d1Dev0, NBYTES));
  REQUIRE(d1Dev0 != nullptr);
  fillDeviceArray(d1Dev0, N, 41);

  // Set arrays in device 1
  HIP_CHECK(hipSetDevice(peerDevice));

  int* d1Dev1 = nullptr;
  HIP_CHECK(hipMalloc(&d1Dev1, NBYTES));
  REQUIRE(d1Dev1 != nullptr);
  fillDeviceArray(d1Dev1, N, 42);

  int* d2Dev1 = nullptr;
  HIP_CHECK(hipMalloc(&d2Dev1, NBYTES));
  REQUIRE(d2Dev1 != nullptr);
  fillDeviceArray(d2Dev1, N, 43);

  // Set destination arrays in device 0
  HIP_CHECK(hipSetDevice(currentDevice));

  int* d2Dev0 = nullptr;
  HIP_CHECK(hipMalloc(&d2Dev0, NBYTES));
  REQUIRE(d2Dev0 != nullptr);
  fillDeviceArray(d2Dev0, N, 44);

  int* h2Dev0 = new int[N];
  REQUIRE(h2Dev0 != nullptr);
  fillHostArray(h2Dev0, N, 45);

  // Do operations in current device
  HIP_CHECK(hipSetDevice(currentDevice));
  HIP_CHECK(hipMemcpyAsync(d1Dev0, h1Dev0, NBYTES, hipMemcpyHostToHost, hipStreamLegacy));

  // Copy from current device to peer device
  HIP_CHECK(hipMemcpyPeerAsync(d1Dev1, peerDevice,     // des
                               d1Dev0, currentDevice,  // src
                               NBYTES, hipStreamLegacy));

  // Do operations in peer device
  HIP_CHECK(hipSetDevice(peerDevice));
  HIP_CHECK(hipMemcpyAsync(d2Dev1, d1Dev1, NBYTES, hipMemcpyDeviceToDevice, hipStreamLegacy));

  // Copy from peer device to current device
  HIP_CHECK(hipMemcpyPeerAsync(d2Dev0, currentDevice,  // des
                               d2Dev1, peerDevice,     // src
                               NBYTES, hipStreamLegacy));

  // Finally copy daat to hostArr4
  HIP_CHECK(hipSetDevice(currentDevice));
  HIP_CHECK(hipMemcpyAsync(h2Dev0, d2Dev0, NBYTES, hipMemcpyDeviceToHost, hipStreamLegacy));
  HIP_CHECK(hipStreamSynchronize(hipStreamLegacy));

  for (int i = 0; i < N; i++) {
    INFO("At index : " << i << " Got value : " << h2Dev0[i]
                       << " Expected value : 40/41/42/43/44 \n");
    REQUIRE(h2Dev0[i] != 45);
  }

  HIP_CHECK(hipSetDevice(currentDevice));
  delete[] h1Dev0;
  delete[] h2Dev0;
  HIP_CHECK(hipFree(d1Dev0));
  HIP_CHECK(hipFree(d2Dev0));

  HIP_CHECK(hipSetDevice(peerDevice));
  HIP_CHECK(hipFree(d1Dev1));
  HIP_CHECK(hipFree(d2Dev1));

  HIP_CHECK(hipSetDevice(currentDevice));
}

/*
 * Local helper function to copy data from host to device
 */
static void copyFromHostToDevice(int* hostArr, int* devArr) {
  HIP_CHECK(hipMemcpyAsync(devArr, hostArr, NBYTES, hipMemcpyHostToDevice, hipStreamLegacy));
}

/*
 * Local helper function to copy data from device to host
 */
static void copyFromDeviceToHost(int* devArr, int* hostArr) {
  HIP_CHECK(hipMemcpyAsync(hostArr, devArr, NBYTES, hipMemcpyDeviceToHost, hipStreamLegacy));
}

/**
 * Test Description
 * ------------------------
 *  - This test case tests the following scenario:-
 *  - Launch two threads,
 *  -  In thread  1  : H -> D Copy
 *  -  In thread  2  : D -> H Copy
 *  - These two thredas should run sequentially as they are using one stream.
 *  - Note : Joined first thread before launching second thread just to avoid,
 *  - the scenario of second thread can be launched before first
 *  - thread while scheduling.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamLegacy_Ext.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipStreamLegacy_TwoThreadsEachOneDiffOperation") {
  const unsigned int threadsSupported = std::thread::hardware_concurrency();

  if (threadsSupported < 2) {
    HipTest::HIP_SKIP_TEST(
        "Skipping due to machine does't "
        "support two concurrent threads");
    return;
  }

  int* hostArrSrc = new int[N];
  REQUIRE(hostArrSrc != nullptr);
  fillHostArray(hostArrSrc, N, 50);

  int* devArr = nullptr;
  HIP_CHECK(hipMalloc(&devArr, NBYTES));
  REQUIRE(devArr != nullptr);
  fillDeviceArray(devArr, N, 51);

  int* hostArrDst = new int[N];
  REQUIRE(hostArrDst != nullptr);
  fillHostArray(hostArrDst, N, 52);

  std::thread H2D_Thread(copyFromHostToDevice, hostArrSrc, devArr);
  H2D_Thread.join();

  std::thread D2H_Thread(copyFromDeviceToHost, devArr, hostArrDst);
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

/**
 * Test Description
 * ------------------------
 *  - This test case tests the following scenario with two devices:-
 *  -   devArrDev0  ->  devArrDev1  : memcpy from device 0 to device 1
 *  -   devArrDev1  ->  hostArrDst  : memcpy from device 1 to host
 *  - The opeations in device 0 and device 1 should run concurrently.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamLegacy_Ext.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipStreamLegacy_TwoDevicesEachOneDiffOperation") {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 2");
    return;
  }

  // Set arrays in device 0
  HIP_CHECK(hipSetDevice(0));

  int* devArrDev0 = nullptr;
  HIP_CHECK(hipMalloc(&devArrDev0, NBYTES));
  REQUIRE(devArrDev0 != nullptr);
  fillDeviceArray(devArrDev0, N, 500);

  // Set arrays in device 1
  HIP_CHECK(hipSetDevice(1));

  int* devArrDev1 = nullptr;
  HIP_CHECK(hipMalloc(&devArrDev1, NBYTES));
  REQUIRE(devArrDev1 != nullptr);
  fillDeviceArray(devArrDev1, N, 501);

  int* hostArrDst = new int[N];
  REQUIRE(hostArrDst != nullptr);
  fillHostArray(hostArrDst, N, 502);

  HIP_CHECK(hipSetDevice(0));

  HIP_CHECK(hipMemcpyPeerAsync(devArrDev1, 1,  // des
                               devArrDev0, 0,  // src
                               NBYTES, hipStreamLegacy));

  HIP_CHECK(hipSetDevice(1));

  HIP_CHECK(hipMemcpyAsync(hostArrDst, devArrDev1, NBYTES, hipMemcpyDeviceToHost, hipStreamLegacy));
  HIP_CHECK(hipStreamSynchronize(hipStreamLegacy));
  
  for (int i = 0; i < N; i++) {
    INFO("At index : " << i << " Got value : " << hostArrDst[i]
                       << " Expected value : 500 or 501 \n");
    REQUIRE(((hostArrDst[i] == 500) || (hostArrDst[i] == 501)));
  }

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipFree(devArrDev1));
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipFree(devArrDev0));
  delete[] hostArrDst;
}

/*
 * Local helper function to copy data from device 0 to device 1
 */
static void operationsInDev0(int* devArrDev0, int* devArrDev1) {
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipMemcpyPeerAsync(devArrDev1, 1,  // des
                               devArrDev0, 0,  // src
                               NBYTES, hipStreamLegacy));
}

/*
 * Local helper function to copy data from device to host
 */
static void operationsInDev1(int* devArrDev1, int* hostArrDst) {
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipMemcpyAsync(hostArrDst, devArrDev1, NBYTES, hipMemcpyDeviceToHost, hipStreamLegacy));
}

/**
 * Test Description
 * ------------------------
 *  - This test case tests the below scenario with two devices in two threads:-
 *  - In thread 1 Dev 0 : devArrDev0 -> devArrDev1 : memcpy from dev 0 to dev 1
 *  - In thread 2 Dev 1 : devArrDev1 -> hostArrDst : memcpy from dev 1 to host
 *  - The opeations in device 0 and device 1, thread 1 and thread 2
 *  - should run concurrently.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamLegacy_Ext.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipStreamLegacy_TwoThreadsInTwoDevicesEachOneDiffOperation") {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 2");
    return;
  }

  // Set arrays in device 0
  HIP_CHECK(hipSetDevice(0));

  int* devArrDev0 = nullptr;
  HIP_CHECK(hipMalloc(&devArrDev0, NBYTES));
  REQUIRE(devArrDev0 != nullptr);
  fillDeviceArray(devArrDev0, N, 999);

  // Set arrays in device 1
  HIP_CHECK(hipSetDevice(1));

  int* devArrDev1 = nullptr;
  HIP_CHECK(hipMalloc(&devArrDev1, NBYTES));
  REQUIRE(devArrDev1 != nullptr);
  fillDeviceArray(devArrDev1, N, 888);

  int* hostArrDst = new int[N];
  REQUIRE(hostArrDst != nullptr);
  fillHostArray(hostArrDst, N, 777);

  HIP_CHECK(hipSetDevice(0));

  std::thread dev0Thread(operationsInDev0, devArrDev0, devArrDev1);
  dev0Thread.join();
  std::thread dev1Thread(operationsInDev1, devArrDev1, hostArrDst);
  dev1Thread.join();
  HIP_CHECK(hipStreamSynchronize(hipStreamLegacy));

  for (int i = 0; i < N; i++) {
    INFO("At index : " << i << " Got value : " << hostArrDst[i]
                       << " Expected value : 999 or 888 \n");
    REQUIRE(((hostArrDst[i] == 999) || (hostArrDst[i] == 888)));
  }

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipFree(devArrDev1));
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipFree(devArrDev0));
  delete[] hostArrDst;
}

/**
 * Test Description
 * ------------------------
 *  - This test creates the child process and use the hipStreamLegacy flag
 *  - in the child process. The hipStreamLegacy flag should work properly
 *  - in child process.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamLegacy_Ext.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipStreamLegacy_InChildProcess") {
  hip::SpawnProc proc("hipStreamLegacy_exe", true);
  REQUIRE(proc.run() == 0);
}

/**
 * Test Description
 * ------------------------
 *  - This test case, tests the hipStreamLegacy with Kernel.
 *  - Kernel launch should success and should give proper result.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamLegacy_Ext.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipStreamLegacy_WithKernel") {
  int* hostArrSrc = new int[N];
  REQUIRE(hostArrSrc != nullptr);
  fillHostArray(hostArrSrc, N, 1);

  int* devArr = nullptr;
  HIP_CHECK(hipMalloc(&devArr, NBYTES));
  REQUIRE(devArr != nullptr);

  int* hostArrDst = new int[N];
  REQUIRE(hostArrDst != nullptr);
  fillHostArray(hostArrDst, N, 6);

  HIP_CHECK(hipMemcpyAsync(devArr, hostArrSrc, NBYTES, hipMemcpyHostToDevice, hipStreamLegacy));
  addOneKernel<<<1, 1, 0, hipStreamLegacy>>>(devArr, N);
  addOneKernel<<<1, 1, 0, hipStreamLegacy>>>(devArr, N);
  HIP_CHECK(hipMemcpyAsync(hostArrDst, devArr, NBYTES, hipMemcpyDeviceToHost, hipStreamLegacy));
  HIP_CHECK(hipStreamSynchronize(hipStreamLegacy));

  for (int i = 0; i < N; i++) {
    INFO("At index : " << i << " Got value : " << hostArrDst[i] << " Expected value : 3 \n");
    REQUIRE(hostArrDst[i] == 3);
  }

  delete[] hostArrSrc;
  delete[] hostArrDst;
  HIP_CHECK(hipFree(devArr));
}

/**
 * Test Description
 * ------------------------
 *  - This test case, tests the hipStreamSynchronize with
 *  - the hipStreamLegacy. It should work without any conflicts.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamLegacy_Ext.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */

TEST_CASE("Unit_hipStreamLegacy_hipStreamSynchronize") {
  int* hostArrSrc = new int[N];
  REQUIRE(hostArrSrc != nullptr);
  fillHostArray(hostArrSrc, N, 1);

  int* devArr = nullptr;
  HIP_CHECK(hipMalloc(&devArr, NBYTES));
  REQUIRE(devArr != nullptr);

  int* hostArrDst = new int[N];
  REQUIRE(hostArrDst != nullptr);
  fillHostArray(hostArrDst, N, 3);

  HIP_CHECK(hipMemcpyAsync(devArr, hostArrSrc, NBYTES, hipMemcpyHostToDevice, hipStreamLegacy));
  HIP_CHECK(hipStreamSynchronize(hipStreamLegacy));

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
