/*Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <atomic>
#include <chrono>
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <unistd.h>
#include <vector>
#define HIP_CHECK_PERF(a)                                                                          \
  {                                                                                                \
    auto err = a;                                                                                  \
    if ((err != hipSuccess) && (err != hipErrorNotReady)) {                                        \
      printf(#a "= Error! %s\n", hipGetErrorString(err));                                          \
      exit(1);                                                                                     \
    }                                                                                              \
  }
/**
 * @addtogroup hipLaunchKernelGGL hipLaunchKernelGGL
 * @{
 * @ingroup PerformanceTest
 */
__global__ void empty_kernel() {
  __shared__ int temp[256];
  temp[threadIdx.x] = sinf(float(threadIdx.x));
}
void rocm_empty_gpu_job(void* stream) {
  hipLaunchKernelGGL(empty_kernel, 1, 256, 0, (hipStream_t)stream);
}
std::vector<std::vector<hipStream_t>> stream_pools;
std::atomic<int> count(0);
bool kill = false;
std::chrono::system_clock::time_point thread_report[16];
void thread_jobs(int dev, int virt) {
  HIP_CHECK_PERF(hipSetDevice(dev));
  hipStream_t exec_stream = stream_pools[dev][virt];
  uint64_t n = 0;
  while (!kill) {
    rocm_empty_gpu_job(exec_stream);
    n++;
    if ((n & 150) == 0) {
      HIP_CHECK_PERF(hipStreamSynchronize(exec_stream));
      thread_report[dev * 4 + virt] = std::chrono::system_clock::now();
    }
    count++;
  }
  HIP_CHECK_PERF(hipStreamSynchronize(exec_stream));
}
/**
 * Test Description
 * ------------------------
 * - This test case prints the number of jobs/Second.
 * - 1) Launch number of threads on each device.
 * - 2) Threads should have only dispatches.
 * - 3) In the main thread calculate the number of jobs/Second
 * - 4) Print the jobs/Second value.
 * Test source
 * ------------------------
 * - catch/perftests/event/hipKernelLookUpPerf.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_hipKernelLookUp_PerfTest") {
  int mgpu = 0;
  HIP_CHECK_PERF(hipGetDeviceCount(&mgpu));
  stream_pools.resize(mgpu);
  HIP_CHECK_PERF(hipSetDeviceFlags(hipDeviceScheduleSpin));
  for (int i = 0; i < mgpu; i++) {
    HIP_CHECK_PERF(hipSetDevice(i));
    stream_pools[i].resize(12);
    for (int j = 0; j < 12; j++)
      HIP_CHECK_PERF(hipStreamCreateWithFlags(&stream_pools[i][j], hipStreamNonBlocking));
  }
  for (int nDev = 1; nDev <= mgpu; nDev++) {
    count = 0;
    INFO("RUNNING ON " << nDev << " DEVICES\n");
    kill = false;
    std::vector<std::thread> threads;
    for (int i = 0; i < nDev * 4; i++) threads.push_back(std::thread(thread_jobs, i / 4, i % 4));
    usleep(1000000);
    auto t1 = std::chrono::system_clock::now();
    int counter = int(count);
    uint64_t total_count = 0;
    double total_time = 0;
    for (int t = 0; t < 10; t++) {
      usleep(1000000);
      auto t2 = std::chrono::system_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
      int counter2 = int(count);
      for (int i = 0; i < nDev * 4; i++) {
        if (std::chrono::duration_cast<std::chrono::microseconds>(t2 - thread_report[i]).count() >=
            1000000) {
          INFO("Thread " << i / 4 << "/" << i % 4 << " is stuck\n");
        }
      }
      total_count += counter2 - counter;
      total_time += duration * 1e-6;
      t1 = t2;
      counter = counter2;
    }
    INFO("AVERAGE: " << total_count << "/" << total_time << " = " << total_count / total_time);
    kill = true;
    for (auto& t : threads) t.join();
    for (int i = 0; i < nDev; i++) {
      HIP_CHECK_PERF(hipSetDevice(i));
      HIP_CHECK_PERF(hipDeviceSynchronize());
    }
  }
}
/**
 * End doxygen group PerformanceTest.
 * @}
 */
