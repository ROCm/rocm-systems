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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <vector>
#include <thread>

static constexpr int N = 2 * 1024 * 1024;
static constexpr size_t NBYTES = N * sizeof(int);
static constexpr int ThreadCount = 5;
int hostArrSrc[ThreadCount][N];
int* devArr[ThreadCount];
int hostArrDst[ThreadCount][N];

/**
 * In addOneKernel function, all elements of the array a increased by 1
 */
static __global__ void addOneKernel(int* a, size_t size) {
  size_t offset = blockDim.x * blockIdx.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = offset; i < size; i += stride) {
    a[i] += 1;
  }
}

static void Copy_to_device() {
  unsigned int ele_size = (32 * 1024);  // 32KB
  int* A_h = nullptr;
  int* A_d = nullptr;

  HIP_CHECK(hipHostMalloc(&A_h, ele_size * sizeof(int)));
  HIP_CHECK(hipMalloc(&A_d, ele_size * sizeof(int)));

  for (unsigned int i = 0; i < ele_size; ++i) {
    A_h[i] = 123;
  }
  HIP_CHECK(
      hipMemcpyAsync(A_d, A_h, ele_size * sizeof(int), hipMemcpyHostToDevice, hipStreamPerThread));
  // Clean up
  HIP_CHECK(hipHostFree(A_h));
  HIP_CHECK(hipFree(A_d));
}

/*
hipStreamPerThread is an implicit stream which gets destroyed once thread is completed.
Scenario : App pushes Async task(s) into hipStreamPerThread and did not wait for it to complete.
Watch out : Incomplete task in hipStreamPerThread should not cause any crash due to thread exit.
 */
TEST_CASE("Unit_hipStreamPerThread_MultiThread") {
  constexpr unsigned int MAX_THREAD_CNT = 10;
  std::vector<std::thread> threads(MAX_THREAD_CNT);

  for (auto& th : threads) {
    th = std::thread(Copy_to_device);
  }

  for (auto& th : threads) {
    th.detach();
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test case, tests the behaviour of hipStreamPerThread
 *  - while the stream is capturing
 * Test source
 * ------------------------
 *  - unit/streamperthread/hipStreamPerThread_MultiThread.cc
 */
TEST_CASE("Unit_hipStreamPerthread_StreamCapture_Basic") {
  GENERATE_CAPTURE();
  hipStream_t stream = hipStreamPerThread;

  std::vector<int> hostArrSrc(N);
  std::fill(hostArrSrc.begin(), hostArrSrc.end(), 5);

  int* devArr = nullptr;
  HIP_CHECK(hipMalloc(&devArr, NBYTES));
  REQUIRE(devArr != nullptr);

  std::vector<int> hostArrDst(N);
  std::fill(hostArrDst.begin(), hostArrDst.end(), 0);

  BEGIN_CAPTURE(stream);

  HIP_CHECK(hipMemcpyAsync(devArr, hostArrSrc.data(), NBYTES, hipMemcpyHostToDevice, stream));
  addOneKernel<<<512, 512, 0, stream>>>(devArr, N);
  HIP_CHECK(hipMemcpyAsync(hostArrDst.data(), devArr, NBYTES, hipMemcpyDeviceToHost, stream));
  END_CAPTURE(stream);
  HIP_CHECK(hipStreamSynchronize(stream));

  for (int i = 0; i < N; i++) {
    if (hostArrDst[i] != 6) {
      std::cout << "At index : " << i << " Got value : " << hostArrDst[i]
                << " Expected value : 6 \n"
                << std::endl;
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipFree(devArr));
}

void launchFunction(const int threadId, hipStreamCaptureMode flags) {
  hipStream_t stream = hipStreamPerThread;
  HIP_CHECK_THREAD(hipStreamBeginCapture(stream, flags));

  HIP_CHECK_THREAD(hipMemcpyAsync(devArr[threadId], hostArrSrc[threadId], NBYTES,
                                  hipMemcpyHostToDevice, stream));
  addOneKernel<<<512, 512, 0, stream>>>(devArr[threadId], N);
  HIP_CHECK_THREAD(hipMemcpyAsync(hostArrDst[threadId], devArr[threadId], NBYTES,
                                  hipMemcpyDeviceToHost, stream));

  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK_THREAD(hipStreamEndCapture(stream, &graph));
  HIP_CHECK_THREAD(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK_THREAD(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK_THREAD(hipStreamSynchronize(stream));

  HIP_CHECK_THREAD(hipGraphExecDestroy(graph_exec));
  HIP_CHECK_THREAD(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - This test case, tests the behaviour of hipStreamPerThread
 *  - while stream is capturing with Multiple threads
 * Test source
 * ------------------------
 *  - unit/streamperthread/hipStreamPerThread_MultiThread.cc
 */
TEST_CASE("Unit_hipStreamPerthread_StreamCapture_MultipleThreads") {
  for (int r = 0; r < ThreadCount; r++) {
    for (int c = 0; c < N; c++) {
      hostArrSrc[r][c] = 5;
    }
  }

  for (int i = 0; i < ThreadCount; i++) {
    HIP_CHECK(hipMalloc(&devArr[i], NBYTES));
    REQUIRE(devArr[i] != nullptr);
  }

  /* Not capturing for hipStreamCaptureModeGlobal mode in case of
   * Multi-threading as hipStreamSynchronize used in launchFunction
   * and it cannot be called from different threads in Global mode
   */
  hipStreamCaptureMode flags =
      GENERATE(hipStreamCaptureModeThreadLocal, hipStreamCaptureModeRelaxed);

  std::vector<std::thread> threads;
  for (int t = 0; t < ThreadCount; t++) {
    threads.push_back(std::thread(launchFunction, t, flags));
  }

  for (int t = 0; (t < ThreadCount) && (t < threads.size()); t++) {
    threads[t].join();
  }
  HIP_CHECK_THREAD_FINALIZE();

  for (int r = 0; r < ThreadCount; r++) {
    for (int c = 0; c < N; c++) {
      if (hostArrDst[r][c] != 6) {
        std::cout << " Got value : " << hostArrDst[r][c]
                  << " Expected value : 6 "
                     " At r = "
                  << r << "c = " << c << std::endl;
        REQUIRE(false);
      }
    }
  }

  for (int r = 0; r < ThreadCount; r++) {
    HIP_CHECK(hipFree(devArr[r]));
  }
}

