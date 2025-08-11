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

/**
* @addtogroup hipMemcpyAsync
* @{
* @ingroup perfMemoryTest
* `hipError_t hipMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int
srcDevice, size_t sizeBytes, hipStream_t stream) ` -
* Copies data between devices.
*/

#include <cmd_options.hh>
#include <hip_array_common.hh>
#include <hip_test_common.hh>
#include <iomanip>
#include <iostream>
#include <sstream>
#define VERIFY_DATA
static constexpr size_t N = 1024 * 1024 * 256;
static constexpr size_t dataBytes = N * sizeof(int);  // 1Gb or 1073741824 bytes
static constexpr int nIters = 10;                     // interation number for test

/**
 * Test Description
 * ------------------------
 * - Verify  device 0 to all devices copy performance.
 *   It also verify the copy performance of indiviual devices like device 0
 *   to device 1, device 0 to device 2,...., device 0 to device 7.
 * Test source
 * ------------------------
 * - perftests/memory/hipPerfBufferCopyInterGpuPerformance.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.0
 */
TEST_CASE("Perf_PerfBufferCopySpeedAll2All_Inter_GPU") {
  int nGpus = 0;
  HIP_CHECK(hipGetDeviceCount(&nGpus));
  if (nGpus < 2) {
    fprintf(stderr, "Need at least 2 GPUs, skipped!\n");
    return;
  }
  int** ArrayOfDevicePointers = reinterpret_cast<int**>(malloc(nGpus * sizeof(int*)));

  for (int local = 0; local < nGpus; local++) {
    HIP_CHECK(hipSetDevice(local));
    HIP_CHECK(hipMalloc(&ArrayOfDevicePointers[local], dataBytes));
  }
  HIP_CHECK(hipSetDevice(0));
  hipStream_t* streams = reinterpret_cast<hipStream_t*>(malloc(nGpus * sizeof(hipStream_t)));
  for (int stream = 0; stream < nGpus; stream++) {
    HIP_CHECK(hipStreamCreateWithFlags(&streams[stream], hipStreamNonBlocking));
  }

  std::vector<int> srcData(dataBytes);
  std::fill_n(srcData.begin(), N, 9);
  HIP_CHECK(hipMemcpy(ArrayOfDevicePointers[0], srcData.data(), dataBytes, hipMemcpyHostToDevice));

  // warmup operation
  for (int dstDeviceId = 1; dstDeviceId < nGpus; dstDeviceId++) {
    HIP_CHECK(hipMemcpyPeerAsync(ArrayOfDevicePointers[dstDeviceId], dstDeviceId,
                                 ArrayOfDevicePointers[0], 0, dataBytes, streams[dstDeviceId]));
  }
  HIP_CHECK(hipDeviceSynchronize());

  for (int dstDeviceId = 1; dstDeviceId < nGpus; dstDeviceId++) {
    auto cpuStart = std::chrono::steady_clock::now();
    for (int it = 0; it < nIters; it++) {
      HIP_CHECK(hipMemcpyPeerAsync(ArrayOfDevicePointers[dstDeviceId], dstDeviceId,
                                   ArrayOfDevicePointers[0], 0, dataBytes, streams[dstDeviceId]));
    }
    HIP_CHECK(hipStreamSynchronize(streams[dstDeviceId]));
    std::chrono::duration<double, std::milli> cpuMS = std::chrono::steady_clock::now() - cpuStart;

    fprintf(stderr,
            "Time: %f ms, transfer of %f GB data and Data transfer rate : %f "
            "GB/s  from device : 0 to device : %d\n",
            cpuMS.count() / nIters, dataBytes / 1e9,
            (dataBytes * 1000) / ((cpuMS.count() / nIters) * 1024 * 1024 * 1024), dstDeviceId);
  }
  // copy device 0 to all the devices
  auto cpuStart = std::chrono::steady_clock::now();
  for (int it = 0; it < nIters; it++) {
    for (int dstDeviceId = 1; dstDeviceId < nGpus; dstDeviceId++) {
      HIP_CHECK(hipMemcpyPeerAsync(ArrayOfDevicePointers[dstDeviceId], dstDeviceId,
                                   ArrayOfDevicePointers[0], 0, dataBytes, streams[dstDeviceId]));
    }
    HIP_CHECK(hipDeviceSynchronize());
  }
  std::chrono::duration<double, std::milli> cpuMS = std::chrono::steady_clock::now() - cpuStart;

  fprintf(stderr,
          "Time: %f ms, transfer of %f GB data and Data transfer rate : %f "
          "GB/s  from device : 0 to All devices\n",
          cpuMS.count() / nIters, dataBytes / 1e9,
          (dataBytes * 1000) / ((cpuMS.count() / nIters) * 1024 * 1024 * 1024));

// validation
#ifdef VERIFY_DATA
  for (int dstDeviceId = 1; dstDeviceId < nGpus; dstDeviceId++) {
    HIP_CHECK(hipSetDevice(dstDeviceId));
    std::vector<int> dstData(dataBytes);
    std::fill_n(dstData.begin(), N, 0);
    HIP_CHECK(hipMemcpy(dstData.data(), ArrayOfDevicePointers[dstDeviceId], dataBytes,
                        hipMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
      if (dstData[i] != 9) std::cout << "data transfer failed\n";
    }
  }
#endif
  for (int i = 0; i < nGpus; i++) {
    HIP_CHECK(hipFree(ArrayOfDevicePointers[i]));
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
}

/**
 * End doxygen group perfMemoryTest.
 * @}
 */
