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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip/hip_ext.h>
#include <hip_array_common.hh>
#define THREADS_PER_BLOCK 64  // 64 threads per wave on Mi300, 32 threads per wave on Navi31
#define MAXITERS 100000       // maximum iteration number in a thread
#define FUNC1(K, I) (K + K * I / 37 - I) / 1091;  // Shared by kernel and host

using namespace std;
constexpr int w1 = 34;

// #define SHOW_DETAILS // Show statics details
// #define VERIFY // Verify data

/**
 * @addtogroup kernelLaunch clock
 * @{
 * @ingroup PerformanceTest
 * Contains unit tests for clock, clock64, wall_clock64 and hipExtLaunchKernelGGL APIs
 */
__global__ void kernel1(uint64_t* out, size_t maxIter, uint64_t* clockCount,
                        uint64_t* wallClockCount) {
  uint64_t wallClock = wall_clock64();
  uint64_t clock = clock64();
  size_t tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  uint64_t k = tid;
  for (size_t i = 0; i < maxIter; i++) {
    k = FUNC1(k, i);
  }
  out[tid] = k;
  clockCount[tid] = clock64() - clock;               // GPU cycle count
  wallClockCount[tid] = wall_clock64() - wallClock;  // Wall clock cycle count
}

#ifdef VERIFY
static void host1(uint64_t* out, size_t maxIter, size_t totalThreadsSize) {
  for (size_t j = 0; j < totalThreadsSize; j++) {
    uint64_t k = j;
    for (size_t i = 0; i < maxIter; i++) {
      k = FUNC1(k, i);
    }
    out[j] = k;
  }
}
#endif

/*
 * Roughly query the variable gpu frequency
 */
static bool query_gpu_frequency(void (*kernel)(uint64_t* out, size_t maxIter, uint64_t* clockCount,
                                               uint64_t* wallClockCount),
                                const int wall_clock_rate, const uint32_t blocksMax,
                                const uint32_t blockSizeMax, const int CUs) {
  hipStream_t stream;
  hipEvent_t start_event, end_event;

  const size_t totalThreadsSize = static_cast<size_t>(blockSizeMax) * blocksMax;
  const size_t totalBytesSize = totalThreadsSize * sizeof(uint64_t);
  const size_t maxIter = MAXITERS;
  uint64_t* out;  // Data to verify kernel rightness
  uint64_t* clockCount;
  uint64_t* wallClockCount;

  HIP_CHECK(hipHostMalloc(&out, totalBytesSize));
  HIP_CHECK(hipMemset(out, 0, totalBytesSize));
  HIP_CHECK(hipMalloc(&clockCount, totalBytesSize));
  HIP_CHECK(hipMemset(clockCount, 0, totalBytesSize));
  HIP_CHECK(hipMalloc(&wallClockCount, totalBytesSize));
  HIP_CHECK(hipMemset(wallClockCount, 0, totalBytesSize));
  HIP_CHECK(hipEventCreate(&start_event));
  HIP_CHECK(hipEventCreate(&end_event));
  HIP_CHECK(hipStreamCreate(&stream));
  std::vector<uint64_t> hostClockCount(totalThreadsSize, 0);
  std::vector<uint64_t> hostWallClockCount(totalThreadsSize, 0);
  bool verified = true;
#ifdef VERIFY
  std::vector<uint64_t> hostOut(totalThreadsSize, 0);
  std::vector<uint64_t> hostOutExpected(totalThreadsSize, 0);
#endif
  for (uint32_t blocks = 1; blocks <= blocksMax; blocks++) {
    for (uint32_t blockSize = THREADS_PER_BLOCK; blockSize <= blockSizeMax; blockSize *= 2) {
      hipExtLaunchKernelGGL(kernel, dim3(blocks), dim3(blockSize), 0, stream, start_event,
                            end_event, 0, out, maxIter, clockCount, wallClockCount);
      HIP_CHECK(hipStreamSynchronize(stream));
      float totalGpuTime = 0;  // Total GPU time
      HIP_CHECK(hipEventElapsedTime(&totalGpuTime, start_event, end_event));

      const size_t curThreadsSize = static_cast<size_t>(blockSize) * blocks;
      const size_t curBytesSize = curThreadsSize * sizeof(uint64_t);
      HIP_CHECK(hipMemcpy(hostClockCount.data(), clockCount, curBytesSize, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(hostWallClockCount.data(), wallClockCount, curBytesSize,
                          hipMemcpyDeviceToHost));

      double clockMean = 0;
      double wallClockMean = 0;
#ifdef SHOW_DETAILS
      double clockDeviation = 0;
      double wallClockDeviation = 0;
      getStatics(hostClockCount.data(), curThreadsSize, clockMean, &clockDeviation);
      getStatics(hostWallClockCount.data(), curThreadsSize, wallClockMean, &wallClockDeviation);
#else
      getStatics(hostClockCount.data(), curThreadsSize, clockMean);
      getStatics(hostWallClockCount.data(), curThreadsSize, wallClockMean);
#endif

      double aveGpuTime = wallClockMean / wall_clock_rate;  // in ms, should be < totalGpuTime
      double avgGpuFrequency = clockMean / aveGpuTime;      // in KHz

      cout << setw(8) << blocks << setw(11) << blockSize << setw(22) << fixed << setprecision(3)
           << avgGpuFrequency / 1000. << setw(20) << fixed << setprecision(3) << aveGpuTime
           << setw(20) << fixed << setprecision(3) << totalGpuTime << setw(26) << fixed
           << setprecision(6) << curBytesSize / totalGpuTime / 1000. << setw(31) << fixed
           << setprecision(6) << curBytesSize / totalGpuTime / 1000. / CUs;

#ifdef SHOW_DETAILS
      cout << setw(15) << fixed << setprecision(3) << wallClockMean << setw(15) << fixed
           << setprecision(3) << wallClockDeviation << setw(15) << fixed << setprecision(3)
           << clockMean << setw(15) << fixed << setprecision(3) << clockDeviation;
#endif
      cout << endl;

#ifdef VERIFY
      HIP_CHECK(hipMemcpy(hostOut.data(), out, curBytesSize, hipMemcpyDeviceToHost));
      host1(hostOutExpected.data(), maxIter, curThreadsSize);
      verified = verify(hostOutExpected.data(), hostOut.data(), curThreadsSize);
#endif
      if (!verified) {
        cout << "Failed" << endl;
        break;
      }
    }
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipEventDestroy(start_event));
  HIP_CHECK(hipEventDestroy(end_event));
  HIP_CHECK(hipFree(out));
  HIP_CHECK(hipFree(clockCount));
  HIP_CHECK(hipFree(wallClockCount));
  return verified;
}

/**
 * Test Description
 * ------------------------
 *  - The test will roughly evaluate the variable GPU frequency in terms of
 * block sizes.
 * ------------------------
 *  - catch\performance\kernelLaunch\hipExtLaunchKernelGGLPerf.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipExtLaunchKernelGGL_QueryGPUFrequency") {
  HIP_CHECK(hipSetDevice(0));
  int clock_rate = 0;       // in kHz
  int wall_clock_rate = 0;  // in kHz
  int occupancyBlocks = 0;
  int occupancyBlockSize = 0;
  hipDeviceProp_t props{};
  HIP_CHECK(hipDeviceGetAttribute(&wall_clock_rate, hipDeviceAttributeWallClockRate, 0));
  HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&occupancyBlocks, &occupancyBlockSize, kernel1));
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  clock_rate = props.clockRate;

  cout << left;
  cout << setw(w1)
       << "--------------------------------------------------------------------------------"
       << endl;
  cout << setw(w1) << "device#" << 0 << endl;
  cout << setw(w1) << "Name: " << props.name << endl;
  cout << setw(w1) << "gcnArchName: " << props.gcnArchName << endl;
  cout << setw(w1) << "multiProcessorCount: " << props.multiProcessorCount << endl;
  cout << setw(w1) << "maxThreadsPerMultiProcessor: " << props.maxThreadsPerMultiProcessor << endl;
  cout << setw(w1) << "maxThreadsPerBlock: " << props.maxThreadsPerBlock << endl;
  cout << setw(w1) << "occupancyBlocks: " << occupancyBlocks << endl;
  cout << setw(w1) << "occupancyBlockSize: " << occupancyBlockSize << endl;
  cout << setw(w1) << "waveSize: " << props.warpSize << endl;
  cout << setw(w1) << "clockRate: " << clock_rate / 1000.0 << " Mhz" << endl;
  cout << setw(w1) << "wallClockRate: " << wall_clock_rate / 1000.0 << " Mhz" << endl;
  cout << setw(w1) << "memoryClockRate: " << props.memoryClockRate / 1000.0 << " Mhz" << endl;
  cout << setw(w1) << "totalGlobalMem: " << fixed << setprecision(2)
       << props.totalGlobalMem / 1000000000. << " GB" << endl;
  cout << setw(w1) << "sharedMemPerBlock: " << props.sharedMemPerBlock / 1024.0 << " KiB" << endl;
  cout << setw(w1) << "l2CacheSize: " << props.l2CacheSize << endl;

  cout << setw(8) << "Blocks " << setw(11) << "BlockSize" << setw(22) << "avgGpuFrequency(MHz)"
       << setw(20) << "aveGpuTime(ms)" << setw(20) << "totalGpuTime(ms)" << setw(26)
       << "processCapacity(Mbytes/s)" << setw(31) << "processCapacityPerCU(Mbytes/s)";
#ifdef SHOW_DETAILS
  cout << setw(15) << "mean wallClock" << setw(15) << "deviation" << setw(15) << "mean clock"
       << setw(15) << "deviation";
#endif
  cout << endl;

  int blocksMax = 64;
  int blockSizeMax = 64;
  bool passed = query_gpu_frequency(kernel1, wall_clock_rate, blocksMax, blockSizeMax,
                                    props.multiProcessorCount);
  REQUIRE(passed);
}
