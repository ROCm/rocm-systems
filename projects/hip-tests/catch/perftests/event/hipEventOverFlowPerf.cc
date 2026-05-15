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
 * @addtogroup hipEventRecord hipEventRecord
 * @{
 * @ingroup PerformanceTest
 * `hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream)`
 * - Record an event in the specified stream..
 */
__global__ void null_kernel() {
  __shared__ int temp[256];
  temp[threadIdx.x] = sinf(float(threadIdx.x));
}
void rocm_null_gpu_job(void* stream) {
  hipLaunchKernelGGL(null_kernel, 1, 256, 0, (hipStream_t)stream);
}
std::vector<std::vector<hipStream_t>> stream_pool;
std::atomic<int> counter(0);
bool do_kill = false;
std::chrono::system_clock::time_point thread_reports[16];
void thread_job(int dev, int virt) {
  HIP_CHECK_PERF(hipSetDevice(dev));  // use dev
  uint8_t* mem;
  HIP_CHECK_PERF(hipMalloc(&mem, 512));
  void* hmem2;
  HIP_CHECK_PERF(hipHostAlloc(&hmem2, 512, 0));
  uint8_t* hmem = (uint8_t*)hmem2;
  hipStream_t exec_stream = stream_pool[dev][virt];
  hipStream_t h2d_stream = stream_pool[dev][virt + 4];
  hipStream_t d2h_stream = stream_pool[dev][virt + 8];
  hipEvent_t eh2d, ed2h;
  HIP_CHECK_PERF(hipEventCreate(&eh2d));
  HIP_CHECK_PERF(hipEventCreate(&ed2h));
  uint64_t n = 0;
  while (!do_kill) {
    rocm_null_gpu_job(exec_stream);
    HIP_CHECK_PERF(hipMemcpyAsync(hmem, mem, 4, hipMemcpyDeviceToHost, d2h_stream));
    HIP_CHECK_PERF(hipMemcpyAsync(mem + 256, hmem + 256, 4, hipMemcpyHostToDevice, h2d_stream));
    HIP_CHECK_PERF(hipEventRecord(eh2d, h2d_stream));
    HIP_CHECK_PERF(hipEventRecord(ed2h, d2h_stream));
    HIP_CHECK_PERF(hipEventQuery(eh2d));
    HIP_CHECK_PERF(hipEventQuery(ed2h));
    n++;
    if ((n & 150) == 0) {
      HIP_CHECK_PERF(hipStreamSynchronize(exec_stream));
      HIP_CHECK_PERF(hipStreamSynchronize(h2d_stream));
      HIP_CHECK_PERF(hipStreamSynchronize(d2h_stream));
      thread_reports[dev * 4 + virt] = std::chrono::system_clock::now();
    }
    counter++;
  }
  HIP_CHECK_PERF(hipStreamSynchronize(exec_stream));
  HIP_CHECK_PERF(hipStreamSynchronize(h2d_stream));
  HIP_CHECK_PERF(hipStreamSynchronize(d2h_stream));
  HIP_CHECK_PERF(hipFree(mem));
  HIP_CHECK_PERF(hipHostFree(hmem2));
  HIP_CHECK_PERF(hipEventDestroy(eh2d));
  HIP_CHECK_PERF(hipEventDestroy(ed2h));
}
/**
 * Test Description
 * ------------------------
 * - This test case prints the number of jobs/Second.
 * - 1) Launch number of thread on each device.
 * - 2) In the thread do some operations like Kernel Launch, memCpy, event
 * record, event query etc.
 * - 3) In the main thread calculate the number of jobs/Second
 * - 4) Print the jobs/Second value.
 * Test source
 * ------------------------
 * - catch/perftests/event/hipEventOverFlowPerf.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_hipEventOverFlow_PerfTest") {
  int mgpu = 0;
  HIP_CHECK_PERF(hipGetDeviceCount(&mgpu));
  stream_pool.resize(mgpu);
  HIP_CHECK_PERF(hipSetDeviceFlags(hipDeviceScheduleSpin));
  std::vector<uint8_t*> memory_buffers[2];
  for (int i = 0; i < mgpu; i++) {
    HIP_CHECK_PERF(hipSetDevice(i));
    stream_pool[i].resize(12);
    memory_buffers[i].resize(128);
    for (int j = 0; j < 12; j++)
      HIP_CHECK_PERF(hipStreamCreateWithFlags(&stream_pool[i][j], hipStreamNonBlocking));
    for (int j = 0; j < 128; j++)
      HIP_CHECK_PERF(hipMalloc(&memory_buffers[i][j], 4096 * ((j & 1) + 1)));
  }
  for (int nDev = 1; nDev <= mgpu; nDev++) {
    counter = 0;
    printf("RUNNING ON %d DEVICES\n", nDev);
    do_kill = false;
    std::vector<std::thread> threads;
    for (int i = 0; i < nDev * 4; i++) threads.push_back(std::thread(thread_job, i / 4, i % 4));
    usleep(1000000);
    auto t1 = std::chrono::system_clock::now();
    int count = int(counter);
    uint64_t total_count = 0;
    double total_time = 0;
    for (int t = 0; t < 10; t++) {
      usleep(1000000);
      auto t2 = std::chrono::system_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
      int count2 = int(counter);
      for (int i = 0; i < nDev * 4; i++) {
        if (std::chrono::duration_cast<std::chrono::microseconds>(t2 - thread_reports[i]).count() >=
            1000000) {
          printf("Thread %d/%d is stuck\n", i / 4, i % 4);
        }
      }
      total_count += count2 - count;
      total_time += duration * 1e-6;
      t1 = t2;
      count = count2;
    }
    printf("AVERAGE: %ld / %f = %f job/s\n", total_count, total_time, total_count / total_time);
    do_kill = true;
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
