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

/**
* @addtogroup hipMemcpyAsync
* @{
* @ingroup perfMemoryTest
* `hipError_t hipMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice,
                              size_t sizeBytes, hipStream_t stream) ` -
* Copies data between devices.
*/

#include <hip_test_common.hh>
#include <hip_array_common.hh>
#include <cmd_options.hh>
#include <iostream>
#include <sstream>
#include <iomanip>
// #define VERIFY_DATA
using namespace std;
enum DEV_MEM_TYPE { COARSE_GRAINED, FINE_GRAINED, EXTENDED_FINE_GRAINED, UNKNOWN_MEM };

typedef long long T;  // You may change to any type

static constexpr int nWarmup = 1;  // warmup iteration number
static constexpr int nIters = 10;  // interation number for test
static constexpr size_t dataBytes = 1024 * 1024 * 1024;

template <typename T> static __global__ void copy_kernel(T* dst, T* src, size_t N) {
  const size_t off = blockDim.x * gridDim.x;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += off) dst[i] = src[i];
}

static string getMemType(DEV_MEM_TYPE memType) {
  switch (memType) {
    case COARSE_GRAINED:
      return "coarse";
    case FINE_GRAINED:
      return "fine";
    case EXTENDED_FINE_GRAINED:
      // Extended - Scope Fine Grained Memory: read is cached, write is not
      return "extended fine";
    default:
      return "unknown mem type";
  }
}

static void mallocDevBuf(void** pp, size_t size, DEV_MEM_TYPE memType) {
  switch (memType) {
    case COARSE_GRAINED:
      HIP_CHECK(hipMalloc(pp, size));
      break;
    case FINE_GRAINED:
#if HT_AMD
      HIP_CHECK(hipExtMallocWithFlags(pp, size, hipDeviceMallocFinegrained));
#else
      fprintf(stderr, "Unsupported memType for nvidia hardware: %d\n", memType);
      REQUIRE(false);
#endif
      break;
    case EXTENDED_FINE_GRAINED:
      // Extended - Scope Fine Grained Memory: read is cached, write is not
      // Perf gain compared with cacheable write
#if HT_AMD
      HIP_CHECK(hipExtMallocWithFlags(pp, size, hipDeviceMallocUncached));
#else
      fprintf(stderr, "Unsupported memType for nvidia hardware: %d\n", memType);
      REQUIRE(false);
#endif
      break;
    default:
      fprintf(stderr, "Unknown memType = %d\n", memType);
      REQUIRE(false);
      break;
  }
}

static void testCopyPerf(bool toRemote, bool kernelCopy, bool onOneGpu, DEV_MEM_TYPE srcType,
                         DEV_MEM_TYPE dstType) {
  int nGpus = 0;
  unsigned int threadsPerBlock = 1024;
  unsigned int blocks = 16;  // DEBUG_CLR_LIMIT_BLIT_WG
  HIP_CHECK(hipGetDeviceCount(&nGpus));
  if (nGpus < 2) {
    fprintf(stderr, "Need at least 2 GPUs, skipped!\n");
    return;
  }
#if 0
  if (kernelCopy) {
    int minGridSize = 0;
    int blockSize = 0;
    HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, copy_kernel<T>));
    blocks = minGridSize / nGpus;
    threadsPerBlock = blockSize;
    fprintf(stderr, "minGridSize %d, threadsPerBlock %u, blocks %u, nGpus %d\n",
      minGridSize, threadsPerBlock, blocks, nGpus);
  }
#endif
#ifdef VERIFY_DATA
  std::vector<char> hostMem0(dataBytes), hostMem1(dataBytes, 0);
  for (size_t n = 0; n < dataBytes; n++) initVal(hostMem0[n]);
#endif
  char** srcBuf = reinterpret_cast<char**>(malloc(nGpus * nGpus * sizeof(char*)));
  char** dstBuf = reinterpret_cast<char**>(malloc(nGpus * nGpus * sizeof(char*)));
  hipStream_t* streams = (hipStream_t*)malloc(nGpus * nGpus * sizeof(hipStream_t));
  for (int local = 0; local < nGpus; local++) {
    HIP_CHECK(hipSetDevice(local));
    for (int remote = 0; remote < nGpus; remote++) {
      if (local == remote) continue;
      mallocDevBuf((void**)(srcBuf + local * nGpus + remote), dataBytes, srcType);
      mallocDevBuf((void**)(dstBuf + local * nGpus + remote), dataBytes, dstType);
      HIP_CHECK(hipStreamCreateWithFlags(&streams[local * nGpus + remote], hipStreamNonBlocking));
      HIP_CHECK(hipDeviceEnablePeerAccess(remote, 0));
#ifdef VERIFY_DATA
      HIP_CHECK(hipMemcpy(srcBuf[local * nGpus + remote], hostMem0.data(), dataBytes,
                          hipMemcpyHostToDevice));
#endif
    }
  }

  unsigned N = dataBytes / sizeof(T);  // Number of T in buffer of dataBytes bytes.
  REQUIRE(N * sizeof(T) == dataBytes);

  auto test = [&](int iters) {
    for (int it = 0; it < iters; it++) {
      for (int local = 0; local < nGpus; local++) {
        HIP_CHECK(hipSetDevice(local));
        for (int i = 0; i < nGpus - 1; i++) {
          int remote = (local + i + 1) % nGpus;
          if (toRemote) {
            // local to remotes
            if (kernelCopy) {
              hipLaunchKernelGGL(copy_kernel<T>, dim3(blocks), dim3(threadsPerBlock), 0,
                                 streams[local * nGpus + remote],
                                 reinterpret_cast<T*>(dstBuf[remote * nGpus + local]),
                                 reinterpret_cast<T*>(srcBuf[local * nGpus + remote]),
                                 static_cast<size_t>(N));
              HIP_CHECK(hipGetLastError());
            } else {
              HIP_CHECK(hipMemcpyPeerAsync(dstBuf[remote * nGpus + local], remote,
                                           srcBuf[local * nGpus + remote], local, dataBytes,
                                           streams[local * nGpus + remote]));
            }
          } else {
            // remotes to local
            if (kernelCopy) {
              hipLaunchKernelGGL(copy_kernel<T>, dim3(blocks), dim3(threadsPerBlock), 0,
                                 streams[remote * nGpus + local],
                                 reinterpret_cast<T*>(dstBuf[local * nGpus + remote]),
                                 reinterpret_cast<T*>(srcBuf[remote * nGpus + local]),
                                 static_cast<size_t>(N));
              HIP_CHECK(hipGetLastError());
            } else {
              HIPCHECK(hipMemcpyPeerAsync(dstBuf[local * nGpus + remote], local,
                                          srcBuf[remote * nGpus + local], remote, dataBytes,
                                          streams[remote * nGpus + local]));
            }
          }
        }
        if (onOneGpu) break;
      }

      for (int local = 0; local < nGpus; local++) {
        for (int remote = 0; remote < nGpus; remote++) {
          if (local == remote) continue;
          HIP_CHECK(hipStreamSynchronize(streams[local * nGpus + remote]));
        }
      }
    }
  };

  string title = kernelCopy ? "kernel copy - " : "hipMemcpyPeerAsync - ";
  title += getMemType(srcType);
  title += " to ";
  title += getMemType(dstType);
  title += toRemote ? " - local to remotes " : " - remotes to local ";
  title += onOneGpu ? "on 1 GPU " : "";

  // warmup
  test(nWarmup);
  auto cpuStart = std::chrono::steady_clock::now();
  test(nIters);
  std::chrono::duration<double, std::milli> cpuMS = std::chrono::steady_clock::now() - cpuStart;
  fprintf(stderr, "%s: Time: %f ms/iter, AvgCopyBW: %f GB/s per GPU\n", title.c_str(),
          cpuMS.count() / nIters, (nGpus - 1) * dataBytes / cpuMS.count() * nIters / 1e6);

  // exit
  for (int local = 0; local < nGpus; local++) {
    HIP_CHECK(hipSetDevice(local));
    HIP_CHECK(hipDeviceSynchronize());
    for (int remote = 0; remote < nGpus; remote++) {
      if (local == remote) continue;
#ifdef VERIFY_DATA
      // Verify
      if (local == 0 && onOneGpu) {
        memset(hostMem1.data(), 0, dataBytes);
        if (toRemote) {
          HIP_CHECK(hipMemcpy(hostMem1.data(), dstBuf[remote * nGpus + local], dataBytes,
                              hipMemcpyDeviceToHost));
        } else {
          HIP_CHECK(hipMemcpy(hostMem1.data(), dstBuf[local * nGpus + remote], dataBytes,
                              hipMemcpyDeviceToHost));
        }
        REQUIRE(hostMem1 == hostMem0);
      } else if (!onOneGpu) {
        // All dstBuf will be enumed regardless of toRemote
        memset(hostMem1.data(), 0, dataBytes);
        HIP_CHECK(hipMemcpy(hostMem1.data(), dstBuf[local * nGpus + remote], dataBytes,
                            hipMemcpyDeviceToHost));
        REQUIRE(hostMem1 == hostMem0);
      }
#endif
      HIP_CHECK(hipFree(srcBuf[local * nGpus + remote]));
      HIP_CHECK(hipFree(dstBuf[local * nGpus + remote]));
      HIP_CHECK(hipStreamDestroy(streams[local * nGpus + remote]));
    }
  }
  free(streams);
  free(dstBuf);
  free(srcBuf);
  SUCCEED("");
}

static void testCopyPerf(bool toRemote, bool kernelCopy, bool onOneGpu) {
  fprintf(stderr, "**********************************************************\n");
#if HT_AMD
  for (int srcType = COARSE_GRAINED; srcType < UNKNOWN_MEM; srcType++) {
    for (int dstType = COARSE_GRAINED; dstType < UNKNOWN_MEM; dstType++) {
      testCopyPerf(toRemote, kernelCopy, onOneGpu, static_cast<DEV_MEM_TYPE>(srcType),
                   static_cast<DEV_MEM_TYPE>(dstType));
    }
  }
#else
  // Only support coarse grained memory allocation on nvidia GPUs
  testCopyPerf(toRemote, kernelCopy, onOneGpu, COARSE_GRAINED, COARSE_GRAINED);
#endif
}

/**
 * Test Description
 * ------------------------
 * - Verify all devices to all devices copy performance via hipMemcpyPeerAsync
 *    from remotes to local.
 *    To specify devices to be tested, export HIP_VISIBLE_DEVICES=gpuid0, gupid1,...
 *      For example, to test first 2 devices, export HIP_VISIBLE_DEVICES=0,1
 *    If GPU number is less than 2, the test will be skipped.
 * Test source
 * ------------------------
 * - perftests/memory/hipPerfBufferCopySpeedAll2All.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.0
 */
TEST_CASE("Perf_PerfBufferCopySpeedAll2All_test - hipMemcpyPeerAsync - remotes to local") {
  testCopyPerf(false, false, false);
}

/**
 * Test Description
 * ------------------------
 * - Verify all devices to all devices copy performance via hipMemcpyPeerAsync
 *    from local to remotes.
 *    To specify devices to be tested, export HIP_VISIBLE_DEVICES=gpuid0, gupid1,...
 *      For example, to test first 2 devices, export HIP_VISIBLE_DEVICES=0,1
 *    If GPU number is less than 2, the test will be skipped.
 * Test source
 * ------------------------
 * - perftests/memory/hipPerfBufferCopySpeedAll2All.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.0
 */
TEST_CASE("Perf_PerfBufferCopySpeedAll2All_test - hipMemcpyPeerAsync - local to remotes") {
  testCopyPerf(true, false, false);
}

/**
 * Test Description
 * ------------------------
 * - Verify all devices to all devices copy performance via kernel copy
 *    from remotes to local.
 *    To specify devices to be tested, export HIP_VISIBLE_DEVICES=gpuid0, gupid1,...
 *      For example, to test first 2 devices, export HIP_VISIBLE_DEVICES=0,1
 *    If GPU number is less than 2, the test will be skipped.
 *    If GPU number is bigger than 4, export GPU_MAX_HW_QUEUES="GPU number -1" to
 *    prevent HW queue serialization because GPU_MAX_HW_QUEUES=4 by default.
 * Test source
 * ------------------------
 * - perftests/memory/hipPerfBufferCopySpeedAll2All.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.0
 */
TEST_CASE("Perf_PerfBufferCopySpeedAll2All_test - kernel copy - remotes to local") {
  testCopyPerf(false, true, false);
}

/**
 * Test Description
 * ------------------------
 * - Verify all devices to all devices copy performance via kernel copy
 *    from local to remotes.
 *    To specify devices to be tested, export HIP_VISIBLE_DEVICES=gpuid0, gupid1,...
 *      For example, to test first 2 devices, export HIP_VISIBLE_DEVICES=0,1
 *    If GPU number is less than 2, the test will be skipped.
 *    If GPU number is bigger than 4, export GPU_MAX_HW_QUEUES="GPU number -1" to
 *    prevent HW queue serialization because GPU_MAX_HW_QUEUES=4 by default.
 * Test source
 * ------------------------
 * - perftests/memory/hipPerfBufferCopySpeedAll2All.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.0
 */
TEST_CASE("Perf_PerfBufferCopySpeedAll2All_test - kernel copy - local to remotes") {
  testCopyPerf(true, true, false);
}

/**
 * Test Description
 * ------------------------
 * - Verify all other devices to the first devices copy performance via hipMemcpyPeerAsync
 *    from remotes to local.
 *    To specify devices to be tested, export HIP_VISIBLE_DEVICES=gpuid0, gupid1,...
 *      For example, to test first 2 devices, export HIP_VISIBLE_DEVICES=0,1
 *    If GPU number is less than 2, the test will be skipped.
 * Test source
 * ------------------------
 * - perftests/memory/hipPerfBufferCopySpeedAll2All.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.0
 */
TEST_CASE("Perf_PerfBufferCopySpeedAll2One_test - hipMemcpyPeerAsync - remotes to local") {
  testCopyPerf(false, false, true);
}

/**
 * Test Description
 * ------------------------
 * - Verify the first device to all other devices copy performance via hipMemcpyPeerAsync
 *    from local to remotes.
 *    To specify devices to be tested, export HIP_VISIBLE_DEVICES=gpuid0, gupid1,...
 *      For example, to test first 2 devices, export HIP_VISIBLE_DEVICES=0,1
 *    If GPU number is less than 2, the test will be skipped.
 * Test source
 * ------------------------
 * - perftests/memory/hipPerfBufferCopySpeedAll2All.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.0
 */
TEST_CASE("Perf_PerfBufferCopySpeedOne2All_test - hipMemcpyPeerAsync - local to remotes") {
  testCopyPerf(true, false, true);
}

/**
 * Test Description
 * ------------------------
 * - Verify the all other devices to the first devices copy performance via kernel copy
 *    from remotes to local.
 *    To specify devices to be tested, export HIP_VISIBLE_DEVICES=gpuid0, gupid1,...
 *      For example, to test first 2 devices, export HIP_VISIBLE_DEVICES=0,1
 *    If GPU number is less than 2, the test will be skipped.
 * Test source
 * ------------------------
 * - perftests/memory/hipPerfBufferCopySpeedAll2All.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.0
 */
TEST_CASE("Perf_PerfBufferCopySpeedAll2One_test - kernel copy - remotes to local") {
  testCopyPerf(false, true, true);
}

/**
 * Test Description
 * ------------------------
 * - Verify the first device to all other devices copy performance via kernel copy
 *    from local to remotes.
 *    To specify devices to be tested, export HIP_VISIBLE_DEVICES=gpuid0, gupid1,...
 *      For example, to test first 2 devices, export HIP_VISIBLE_DEVICES=0,1
 *    If GPU number is less than 2, the test will be skipped.
 *    If GPU number is bigger than 4, export GPU_MAX_HW_QUEUES="GPU number -1" to
 *    prevent HW queue serialization because GPU_MAX_HW_QUEUES=4 by default.
 * Test source
 * ------------------------
 * - perftests/memory/hipPerfBufferCopySpeedAll2All.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.0
 */
TEST_CASE("Perf_PerfBufferCopySpeedOne2All_test - kernel copy - local to remotes") {
  testCopyPerf(true, true, true);
}

/**
 * End doxygen group perfMemoryTest.
 * @}
 */
