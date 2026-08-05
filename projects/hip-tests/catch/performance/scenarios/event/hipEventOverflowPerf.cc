/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <atomic>
#include <chrono>
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <unistd.h>
#include <vector>
#include "hipPerfCommon.hh"
/**
 * @addtogroup hipEventRecord hipEventRecord
 * @{
 * @ingroup PerformanceTestEvent
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
std::atomic<bool> do_kill{false};
std::vector<PaddedTimestamp> thread_reports;
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
      thread_reports[dev * 4 + virt].ns.store(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count(),
          std::memory_order_relaxed);
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
 * - performance/scenarios/event/hipEventOverflowPerf.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.5
 */
HIP_TEST_CASE(Performance_hipEventOverflow) {
  int mgpu = 0;
  HIP_CHECK_PERF(hipGetDeviceCount(&mgpu));
  stream_pool.resize(mgpu);
  HIP_CHECK_PERF(hipSetDeviceFlags(hipDeviceScheduleSpin));
  std::vector<std::vector<uint8_t*>> memory_buffers(mgpu);
  for (int i = 0; i < mgpu; i++) {
    HIP_CHECK_PERF(hipSetDevice(i));
    stream_pool[i].resize(12);
    memory_buffers[i].resize(128);
    for (int j = 0; j < 12; j++)
      HIP_CHECK_PERF(hipStreamCreateWithFlags(&stream_pool[i][j], hipStreamNonBlocking));
    for (int j = 0; j < 128; j++)
      HIP_CHECK_PERF(hipMalloc(&memory_buffers[i][j], 4096 * ((j & 1) + 1)));
  }
  // std::atomic is non-movable, so the vector cannot be resized — construct
  // it at the required size and move-assign it into the global.
  thread_reports = std::vector<PaddedTimestamp>(mgpu * 4);
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
      int64_t t2_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t2.time_since_epoch())
                          .count();
      for (int i = 0; i < nDev * 4; i++) {
        int64_t slot_ns = thread_reports[i].ns.load(std::memory_order_relaxed);
        if ((t2_ns - slot_ns) / 1000 >= 1000000) {
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
  HIP_CHECK_PERF(hipSetDevice(0));
  for (int i = 0; i < mgpu; i++) {
    for (auto* buf : memory_buffers[i]) HIP_CHECK_PERF(hipFree(buf));
    for (auto s : stream_pool[i]) HIP_CHECK_PERF(hipStreamDestroy(s));
  }
}
/**
 * End doxygen group PerformanceTest.
 * @}
 */
