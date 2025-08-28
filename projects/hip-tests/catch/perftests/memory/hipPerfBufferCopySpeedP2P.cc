/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
 * `hipMemcpyAsync(void* dst, const void* src, size_t count,
 *                 hipMemcpyKind kind, hipStream_t stream = 0)` -
 * Copies data between devices.
 */

#include <hip_test_common.hh>
#include <hip_array_common.hh>
#include <cmd_options.hh>
#include <iostream>
#include <sstream>
#include <iomanip>
using namespace std;
typedef long long T;  // You may change to any type

// #define VERIFY_DATA
enum TIMING_MODE { TIMING_MODE_CPU, TIMING_MODE_GPU };

// -sizes are in bytes, +sizes are in kb, last size must be largest
#if 1
static constexpr int sizes[] = {-64,  -256,  -512,  1,     2,      4,     8,    16,
                                32,   64,    128,   256,   512,    1024,  2048, 4096,
                                8192, 16384, 32768, 65536, 131072, 262144};
#else
static constexpr int sizes[] = {262144};
#endif
static constexpr int nSizes = sizeof(sizes) / sizeof(sizes[0]);
static constexpr double megaSize = 1000000.;
static constexpr int defaultIterations = 200;
static constexpr unsigned threadsPerBlock = 1024;

template <typename T> static __global__ void copy_kernel(T* dst, T* src, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) dst[idx] = src[idx];  // We make sure idx < N
}

static size_t sizeToBytes(int size) { return (size < 0) ? -size : size * 1024; }

static string sizeToString(int size) {
  stringstream ss;
  if (size < 0) {
    ss << setfill('0') << setw(3) << -size / 1024.;
  } else {
    ss << size;
  }
  return ss.str();
}

static void checkP2PSupport() {
  int deviceCnt = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCnt));
  cout << "Total no. of  available gpu #" << deviceCnt << "\n" << endl;

  for (int deviceId = 0; deviceId < deviceCnt; deviceId++) {
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));
    cout << "for gpu#" << deviceId << " " << props.name << endl;
    cout << "    peer2peer supported : ";
    int PeerCnt = 0;
    for (int i = 0; i < deviceCnt; i++) {
      int isPeer;
      HIP_CHECK(hipDeviceCanAccessPeer(&isPeer, i, deviceId));
      if (isPeer) {
        cout << "gpu#" << i << " ";
        ++PeerCnt;
      }
    }
    if (PeerCnt == 0) cout << "NONE" << " ";

    cout << std::endl;
    cout << "    peer2peer not supported : ";
    int nonPeerCnt = 0;
    for (int i = 0; i < deviceCnt; i++) {
      int isPeer;
      HIP_CHECK(hipDeviceCanAccessPeer(&isPeer, i, deviceId));
      if (!isPeer && (i != deviceId)) {
        cout << "gpu#" << i << " ";
        ++nonPeerCnt;
      }
    }
    if (nonPeerCnt == 0) cout << "NONE" << " ";

    cout << "\n" << endl;
  }

  cout << "\nNote: For non-supported peer2peer devices, memcopy will use/follow the normal "
          "behaviour (GPU1-->host then host-->GPU2)\n\n"
       << endl;
}

static void outputMatrix(const string& title, const int numGPUs, const vector<double>& data,
                         const TIMING_MODE mode, const size_t dataSize, const int iterations) {
  fprintf(stderr, "%s, Timing %s, Data %zu KB, Iterations %d\n   ", title.c_str(),
          mode == TIMING_MODE_GPU ? "GPU" : "CPU", dataSize, iterations);
  for (int j = 0; j < numGPUs; j++) {
    fprintf(stderr, "%9d ", j);
  }
  fprintf(stderr, "\n");
  for (int i = 0; i < numGPUs; i++) {
    fprintf(stderr, "%6d ", i);
    for (int j = 0; j < numGPUs; j++) {
      fprintf(stderr, "%8.06f ", data[i * numGPUs + j]);
    }
    fprintf(stderr, "\n");
  }
}

static void testP2PUniDirMemPerf(const int iterations, const TIMING_MODE timingMode,
                                 const bool useHipMemcpyAsync) {
  const char* method = useHipMemcpyAsync ? "hipMemcpyAsync()" : "copy kernel";
  int gpuCount = 0;
  HIP_CHECK(hipGetDeviceCount(&gpuCount));
  if (gpuCount < 1) {
    fprintf(stderr, "Need at least 1 GPU, skipped!\n");
    return;
  }
  vector<double> timeMs(gpuCount * gpuCount, 0.);
  vector<double> bandWidth(gpuCount * gpuCount, 0.);

  size_t numMax = sizeToBytes(sizes[nSizes - 1]) * 2;
#ifdef VERIFY_DATA
  std::vector<char> hostMem0(numMax), hostMem1(numMax, 0);
  for (size_t n = 0; n < numMax; n++) initVal(hostMem0[n]);
#endif
  for (int currentGpu = 0; currentGpu < gpuCount; currentGpu++) {
    for (int peerGpu = 0; peerGpu < gpuCount; peerGpu++) {
      HIP_CHECK(hipSetDevice(currentGpu));

      fprintf(stderr, "Uni: Gpu%d -> Gpu%d by %s, Timing %s, Iterations %d\n", currentGpu, peerGpu,
              method, timingMode == TIMING_MODE_GPU ? "GPU" : "CPU", iterations);

      if (currentGpu != peerGpu) {
        int canAccessPeer = 0;
        HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, currentGpu, peerGpu));
        if (!canAccessPeer) {
          fprintf(stderr, "Gpu%d cannot access Gpu%d, Skipped\n", currentGpu, peerGpu);
          continue;
        }
        HIP_CHECK(hipDeviceEnablePeerAccess(peerGpu, 0));
      }
      fprintf(stderr, "Size(KB)          Time(ms)       Bandwidth(GB/s)\n");

      unsigned char *currentGpuMem = nullptr, *peerGpuMem = nullptr;

      HIP_CHECK(hipMalloc((void**)&currentGpuMem, numMax));
      HIP_CHECK(hipSetDevice(peerGpu));
      HIP_CHECK(hipMalloc((void**)&peerGpuMem, numMax));
      HIP_CHECK(hipSetDevice(currentGpu));
#ifdef VERIFY_DATA
      HIP_CHECK(hipMemcpy(currentGpuMem, hostMem0.data(), numMax, hipMemcpyHostToDevice));
#endif
      unsigned N = numMax / sizeof(T);   // Number of T in buffer of numMax bytes.
      REQUIRE(N * sizeof(T) == numMax);  // To prevent verification failure
      unsigned blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
      // Warmup
      if (useHipMemcpyAsync) {
        HIP_CHECK(hipMemcpyAsync(peerGpuMem, currentGpuMem, numMax, hipMemcpyDeviceToDevice, 0));
      } else {
        hipLaunchKernelGGL(copy_kernel<T>, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                           reinterpret_cast<T*>(peerGpuMem), reinterpret_cast<T*>(currentGpuMem),
                           static_cast<size_t>(N));
        HIP_CHECK(hipGetLastError());
      }
      HIP_CHECK(hipDeviceSynchronize());
#ifdef VERIFY_DATA
      // Verify
      HIP_CHECK(hipMemcpy(hostMem1.data(), peerGpuMem, numMax, hipMemcpyDeviceToHost));
      REQUIRE(hostMem1 == hostMem0);
#endif
      float t = 0;  // in ms
      auto cpuStart = std::chrono::steady_clock::now();
      hipEvent_t eventStart, eventStop;
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      if (timingMode == TIMING_MODE_GPU) {
        HIP_CHECK(hipEventCreate(&eventStart));
        HIP_CHECK(hipEventCreate(&eventStop));
      }

      for (int i = 0; i < nSizes; i++) {
        const int thisSize = sizes[i];
        const size_t nbytes = sizeToBytes(sizes[i]);
        N = nbytes / sizeof(T);  // Number of T in buffer of nbytes bytes
        blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        if (timingMode == TIMING_MODE_CPU) {
          HIP_CHECK(hipDeviceSynchronize());
          cpuStart = std::chrono::steady_clock::now();
        } else if (timingMode == TIMING_MODE_GPU) {
          HIP_CHECK(hipEventRecord(eventStart, stream));
        }

        for (size_t offsetEnd = numMax - nbytes, offset = 0, j = 0; j < iterations;
             j++, offset += nbytes) {
          if (offset > offsetEnd) offset = 0;
          if (useHipMemcpyAsync) {
            HIP_CHECK(hipMemcpyAsync(peerGpuMem + offset, currentGpuMem + offset, nbytes,
                                     hipMemcpyDeviceToDevice, stream));
          } else {
            hipLaunchKernelGGL(copy_kernel<T>, dim3(blocks), dim3(threadsPerBlock), 0, stream,
                               reinterpret_cast<T*>(peerGpuMem + offset),
                               reinterpret_cast<T*>(currentGpuMem + offset),
                               static_cast<size_t>(N));
            HIP_CHECK(hipGetLastError());
          }
        }
        if (timingMode == TIMING_MODE_GPU) {
          HIP_CHECK(hipEventRecord(eventStop, stream));
          HIP_CHECK(hipEventSynchronize(eventStop));
          HIP_CHECK(hipEventElapsedTime(&t, eventStart, eventStop));
        } else if (timingMode == TIMING_MODE_CPU) {
          HIP_CHECK(hipDeviceSynchronize());
          std::chrono::duration<double, std::milli> cpuMs =
              std::chrono::steady_clock::now() - cpuStart;
          t = cpuMs.count();
        }
        t /= iterations;
        double bandwidth = nbytes / megaSize / t;  // GByte/s

        fprintf(stderr, "%8s,         %-9.06lf,        %-4.08lf\n", sizeToString(thisSize).c_str(),
                t, bandwidth);
        if (i == (nSizes - 1)) {
          timeMs[currentGpu * gpuCount + peerGpu] = t;
          bandWidth[currentGpu * gpuCount + peerGpu] = bandwidth;
        }
      }
      if (currentGpu != peerGpu) {
        HIP_CHECK(hipDeviceDisablePeerAccess(peerGpu));
      }
      if (timingMode == TIMING_MODE_GPU) {
        HIP_CHECK(hipEventDestroy(eventStart));
        HIP_CHECK(hipEventDestroy(eventStop));
      }
#ifdef VERIFY_DATA
      // Verify again
      for (size_t n = 0; n < numMax; n++) hostMem1[n] = 0;
      HIP_CHECK(hipMemcpy(hostMem1.data(), peerGpuMem, numMax, hipMemcpyDeviceToHost));
      REQUIRE(hostMem1 == hostMem0);
#endif
      // Cleanup
      HIP_CHECK(hipFree((void*)currentGpuMem));
      HIP_CHECK(hipFree((void*)peerGpuMem));
      HIP_CHECK(hipStreamDestroy(stream));
    }
  }
  outputMatrix(string("Unidirectional ") + method + " Time Table(ms)", gpuCount, timeMs, timingMode,
               sizes[nSizes - 1], iterations);
  outputMatrix(string("Unidirectional ") + method + " Bandwith Table(GB/s)", gpuCount, bandWidth,
               timingMode, sizes[nSizes - 1], iterations);
}

static void testP2PBiDirMemPerf(const int iterations, const bool useHipMemcpyAsync) {
  const char* method = useHipMemcpyAsync ? "hipMemcpyAsync()" : "copy kernel";
  int gpuCount = 0;
  HIP_CHECK(hipGetDeviceCount(&gpuCount));
  if (gpuCount < 1) {
    fprintf(stderr, "Need at least 1 GPU, skipped!\n");
    return;
  }
  vector<double> timeMs(gpuCount * gpuCount, 0.);
  vector<double> bandWidth(gpuCount * gpuCount, 0.);
  size_t numMax = sizeToBytes(sizes[nSizes - 1]) * 2;
  for (int currentGpu = 0; currentGpu < gpuCount; currentGpu++) {
    for (int peerGpu = 0; peerGpu < gpuCount; peerGpu++) {
      HIP_CHECK(hipSetDevice(currentGpu));
      fprintf(stderr, "Bi: Gpu%d <-> Gpu%d by %s, Timing GPU, Iterations %d\n", currentGpu, peerGpu,
              method, iterations);

      if (currentGpu != peerGpu) {
        int canAccessPeer = 0;
        HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, currentGpu, peerGpu));
        if (!canAccessPeer) {
          fprintf(stderr, "currentGpu %d cannot access peerGpu %d\n", currentGpu, peerGpu);
          continue;
        }
        canAccessPeer = 0;
        HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, peerGpu, currentGpu));
        if (!canAccessPeer) {
          fprintf(stderr, "peerGpu %d cannot access currentGpu %d\n", peerGpu, currentGpu);
          continue;
        }
        HIP_CHECK(hipSetDevice(peerGpu));
        HIP_CHECK(hipDeviceEnablePeerAccess(currentGpu, 0));
        HIP_CHECK(hipSetDevice(currentGpu));
        HIP_CHECK(hipDeviceEnablePeerAccess(peerGpu, 0));
      }
      fprintf(stderr,
              "Gpu%d -> Gpu%d                                         *"
              "*      Gpu%d -> Gpu%d\n",
              currentGpu, peerGpu, peerGpu, currentGpu);
      fprintf(stderr,
              "Size(KB)          Time(ms)       Bandwidth(GB/s)      "
              "       Time(ms)       Bandwidth(GB/s)\n");

      unsigned char *currentGpuMem[2], *peerGpuMem[2];
      HIP_CHECK(hipMalloc((void**)&currentGpuMem[0], numMax));
      HIP_CHECK(hipMalloc((void**)&currentGpuMem[1], numMax));

      HIP_CHECK(hipSetDevice(peerGpu));
      // peerGpu is the current device
      HIP_CHECK(hipMalloc((void**)&peerGpuMem[0], numMax));
      HIP_CHECK(hipMalloc((void**)&peerGpuMem[1], numMax));

      HIP_CHECK(hipSetDevice(currentGpu));

      unsigned N = numMax / sizeof(T);  // Number of T in buffer of numMax bytes.
      REQUIRE(N * sizeof(T) == numMax);
      unsigned blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

      // Warmup. currentGpu is the current device
      if (useHipMemcpyAsync) {
        HIP_CHECK(
            hipMemcpyAsync(peerGpuMem[0], currentGpuMem[0], numMax, hipMemcpyDeviceToDevice, 0));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipSetDevice(peerGpu));
        HIP_CHECK(
            hipMemcpyAsync(currentGpuMem[1], peerGpuMem[1], numMax, hipMemcpyDeviceToDevice, 0));
        HIP_CHECK(hipDeviceSynchronize());
      } else {
        // Warmup
        hipLaunchKernelGGL(copy_kernel<T>, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                           reinterpret_cast<T*>(peerGpuMem[0]),
                           reinterpret_cast<T*>(currentGpuMem[0]), static_cast<size_t>(N));
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipSetDevice(peerGpu));
        hipLaunchKernelGGL(copy_kernel<T>, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                           reinterpret_cast<T*>(currentGpuMem[1]),
                           reinterpret_cast<T*>(peerGpuMem[1]), static_cast<size_t>(N));
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
      }

      hipStream_t stream[2];
      hipEvent_t eventStart[2], eventStop[2];
      HIP_CHECK(hipSetDevice(currentGpu));
      HIP_CHECK(hipStreamCreate(&stream[0]));
      HIP_CHECK(hipEventCreate(&eventStart[0]));
      HIP_CHECK(hipEventCreate(&eventStop[0]));
      HIP_CHECK(hipSetDevice(peerGpu));
      HIP_CHECK(hipStreamCreate(&stream[1]));
      HIP_CHECK(hipEventCreate(&eventStart[1]));
      HIP_CHECK(hipEventCreate(&eventStop[1]));

      HIP_CHECK(hipSetDevice(currentGpu));

      for (int i = 0; i < nSizes; i++) {
        const int thisSize = sizes[i];
        const size_t nbytes = sizeToBytes(thisSize);
        N = nbytes / sizeof(T);  // Number of T in buffer of nbytes bytes
        blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

        HIP_CHECK(hipEventRecord(eventStart[0], stream[0]));
        HIP_CHECK(hipEventRecord(eventStart[1], stream[1]));

        for (size_t offsetEnd = numMax - nbytes, offset = 0, j = 0; j < iterations;
             j++, offset += nbytes) {
          if (offset > offsetEnd) offset = 0;
          if (useHipMemcpyAsync) {
            HIP_CHECK(hipMemcpyAsync(peerGpuMem[0] + offset, currentGpuMem[0] + offset, nbytes,
                                     hipMemcpyDeviceToDevice, stream[0]));
            HIP_CHECK(hipMemcpyAsync(currentGpuMem[1] + offset, peerGpuMem[1] + offset, nbytes,
                                     hipMemcpyDeviceToDevice, stream[1]));
          } else {
            hipLaunchKernelGGL(copy_kernel<T>, dim3(blocks), dim3(threadsPerBlock), 0, stream[0],
                               reinterpret_cast<T*>(peerGpuMem[0] + offset),
                               reinterpret_cast<T*>(currentGpuMem[0] + offset),
                               static_cast<size_t>(N));
            hipLaunchKernelGGL(copy_kernel<T>, dim3(blocks), dim3(threadsPerBlock), 0, stream[1],
                               reinterpret_cast<T*>(currentGpuMem[1] + offset),
                               reinterpret_cast<T*>(peerGpuMem[1] + offset),
                               static_cast<size_t>(N));
          }
        }

        HIP_CHECK(hipEventRecord(eventStop[0], stream[0]));
        HIP_CHECK(hipEventRecord(eventStop[1], stream[1]));

        HIP_CHECK(hipEventSynchronize(eventStop[0]));
        HIP_CHECK(hipEventSynchronize(eventStop[1]));

        float t[2];
        double bandwidth[2];
        for (int n = 0; n < 2; n++) {
          HIP_CHECK(hipEventElapsedTime(&t[n], eventStart[n], eventStop[n]));
          t[n] /= iterations;
          bandwidth[n] = nbytes / megaSize / t[n];  // GByte/s
        }

        fprintf(stderr,
                "%8s,         %-9.06lf,        %-4.08lf,              "
                "%-9.06lf,        %-4.08lf\n",
                sizeToString(thisSize).c_str(), t[0], bandwidth[0], t[1], bandwidth[1]);

        if (i == (nSizes - 1)) {
          timeMs[currentGpu * gpuCount + peerGpu] = (t[0] + t[1]) / 2;
          bandWidth[currentGpu * gpuCount + peerGpu] = bandwidth[0] + bandwidth[1];
        }
      }
      if (currentGpu != peerGpu) {
        HIP_CHECK(hipSetDevice(peerGpu));
        HIPCHECK(hipDeviceDisablePeerAccess(currentGpu));
        HIP_CHECK(hipSetDevice(currentGpu));
        HIPCHECK(hipDeviceDisablePeerAccess(peerGpu));
      }
      for (int i = 0; i < 2; i++) {
        HIP_CHECK(hipEventDestroy(eventStart[i]));
        HIP_CHECK(hipEventDestroy(eventStop[i]));
        HIP_CHECK(hipStreamDestroy(stream[i]));
        HIP_CHECK(hipFree((void*)currentGpuMem[i]));
        HIP_CHECK(hipFree((void*)peerGpuMem[i]));
      }
    }
  }
  outputMatrix(string("Bidirectional ") + method + " Time Table(ms)", gpuCount, timeMs,
               TIMING_MODE_GPU, sizes[nSizes - 1], iterations);
  outputMatrix(string("Bidirectional ") + method + " Bandwith Table(GB/s)", gpuCount, bandWidth,
               TIMING_MODE_GPU, sizes[nSizes - 1], iterations);
}

/**
 * Test Description
 * ------------------------
 *  - Verify P2P uni-direction memcpy performance.
 *    To enable rocr kernel copying, export HSA_ENABLE_SDMA=0
 *    To enable SDMA copying, export HSA_ENABLE_SDMA=1 (by default)
 *    To specify devices to be tested, export HIP_VISIBLE_DEVICES=gpuid0, gupid1,...
 *      For example, to test first 2 devices, export HIP_VISIBLE_DEVICES=0,1
 * Test source
 * ------------------------
 *  - perftests/memory/hipPerfBufferCopySpeedP2P.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Perf_hipTestP2PUniDirMemcpyAsync_test - Timing CPU") {
  const int iterations =
      cmd_options.iterations == 1000 ? defaultIterations : cmd_options.iterations;
  testP2PUniDirMemPerf(iterations, TIMING_MODE_CPU, true);
}

TEST_CASE("Perf_hipTestP2PUniDirMemcpyAsync_test - Timing GPU") {
  const int iterations =
      cmd_options.iterations == 1000 ? defaultIterations : cmd_options.iterations;
  testP2PUniDirMemPerf(iterations, TIMING_MODE_GPU, true);
}

/**
 * Test Description
 * ------------------------
 *  - Verify P2P uni-direction kernel copy performance.
 *    To specify devices to be tested, export HIP_VISIBLE_DEVICES=gpuid0, gupid1,...
 *      For example, to test first 2 devices, export HIP_VISIBLE_DEVICES=0,1
 * Test source
 * ------------------------
 *  - perftests/memory/hipPerfBufferCopySpeedP2P.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Perf_hipTestP2PUniDirKernelCopy_test - Timing CPU") {
  const int iterations =
      cmd_options.iterations == 1000 ? defaultIterations : cmd_options.iterations;
  testP2PUniDirMemPerf(iterations, TIMING_MODE_CPU, false);
}

TEST_CASE("Perf_hipTestP2PUniDirKernelCopy_test - Timing GPU") {
  const int iterations =
      cmd_options.iterations == 1000 ? defaultIterations : cmd_options.iterations;
  testP2PUniDirMemPerf(iterations, TIMING_MODE_GPU, false);
}

/**
 * Test Description
 * ------------------------
 *  - Verify P2P bi-direction memcpy performance.
 *    To enable rocr kernel copying, export HSA_ENABLE_SDMA=0
 *    To enable SDMA copying, export HSA_ENABLE_SDMA=1 (by default)
 *    To specify devices to be tested, export HIP_VISIBLE_DEVICES=gpuid0, gupid1,...
 *      For example, to test first 2 devices, export HIP_VISIBLE_DEVICES=0,1
 * Test source
 * ------------------------
 *  - perftests/memory/hipPerfBufferCopySpeedP2P.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Perf_hipTestP2PBiDirMemcpyAsync_test") {
  const int iterations =
      cmd_options.iterations == 1000 ? defaultIterations : cmd_options.iterations;
  testP2PBiDirMemPerf(iterations, true);
}

/**
 * Test Description
 * ------------------------
 *  - Verify P2P bi-direction kernel copy performance.
 *    To specify devices to be tested, export HIP_VISIBLE_DEVICES=gpuid0, gupid1,...
 *      For example, to test first 2 devices, export HIP_VISIBLE_DEVICES=0,1
 * Test source
 * ------------------------
 *  - perftests/memory/hipPerfBufferCopySpeedP2P.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Perf_hipTestP2PBiDirKernelCopy_test") {
  const int iterations =
      cmd_options.iterations == 1000 ? defaultIterations : cmd_options.iterations;
  testP2PBiDirMemPerf(iterations, false);
}

/**
 * Test Description
 * ------------------------
 *  - Check support of peer to peer
 * Test source
 * ------------------------
 *  - perftests/memory/hipPerfBufferCopySpeedP2P.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Perf_hipCheckP2PSupport") { checkP2PSupport(); }

/**
 * End doxygen group perfMemoryTest.
 * @}
 */
