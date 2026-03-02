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
 * @addtogroup hipMemcpyAsync hipMemcpyAsync
 * @{
 * @ingroup perfMemoryTest
 * `hipMemcpyAsync(void* dst, const void* src, size_t count,
 *                 hipMemcpyKind kind, hipStream_t stream = 0)` -
 * Copies data between host and device, or device to device etc.
 */
#include <hip/hip_ext.h>
#include <hip_test_common.hh>
#include <cstdlib>
#include <iomanip>  // Add this at the top if not already included
#define ENABLE_DEBUG 1

#define NUM_SIZES 8
//  4KB, 8KB, 64KB, 256KB, 1 MB, 4MB, 16 MB, 16MB+10, 128 MB
static const unsigned int Sizes[NUM_SIZES] = {4096, 8192, 65536, 1048576, 4194304,
                                              16777216, 16777216 + 10, 134217728};
static const unsigned int Iterations[2] = {1, 1000};

#define BUF_TYPES 5
//  25 ways to combine 5 different buffer types
#define NUM_SUBTESTS (BUF_TYPES * BUF_TYPES)

static void setData(void* ptr, unsigned int size, char value) {
  char* ptr2 = reinterpret_cast<char*>(ptr);
  for (unsigned int i = 0; i < size; i++) {
    ptr2[i] = value;
  }
}

static void checkData(void* ptr, unsigned int size, char value) {
  char* ptr2 = reinterpret_cast<char*>(ptr);
  for (unsigned int i = 0; i < size; i++) {
    if (ptr2[i] != value) {
      INFO("Validation failed at " << i << " Got " << ptr2[i] << " Expected " << value);
      REQUIRE(false);
    }
  }
}

static bool hipPerfBufferCopySpeed_test(int p_tests) {
  int testIdx = 0;
  unsigned int numIter;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  // 1. Run all P2P for all sizes
  if (numDevices >= 2) {
    for (int sizeIdx = 0; sizeIdx < NUM_SIZES; ++sizeIdx) {
      if (p_tests != -1 && testIdx != p_tests) {
        ++testIdx;
        continue;
      }
      unsigned int bufSize_ = Sizes[sizeIdx];
      void* srcBuffer = NULL;
      void* dstBuffer = NULL;
      numIter = Iterations[1];
      HIP_CHECK(hipSetDevice(0));
      HIP_CHECK(hipMalloc(&srcBuffer, bufSize_));
      hipError_t errMemset = hipMemset(srcBuffer, 0xd0, bufSize_);
      if (errMemset != hipSuccess) {
        HIP_CHECK(hipFree(srcBuffer));
        continue;
      }
      HIP_CHECK(hipSetDevice(1));
      HIP_CHECK(hipMalloc(&dstBuffer, bufSize_));
      int canAccessPeer01 = 0, canAccessPeer10 = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer01, 0, 1));
      HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer10, 1, 0));
      if (!canAccessPeer01 || !canAccessPeer10) {
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipDeviceDisablePeerAccess(1));
        HIP_CHECK(hipSetDevice(1));
        HIP_CHECK(hipDeviceDisablePeerAccess(0));
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipFree(srcBuffer));
        HIP_CHECK(hipSetDevice(1));
        HIP_CHECK(hipFree(dstBuffer));
        HIP_CHECK(hipSetDevice(0));
        continue;
      }
      HIP_CHECK(hipSetDevice(0));
      hipError_t errPeer0 = hipDeviceEnablePeerAccess(1, 0);
      HIP_CHECK(hipSetDevice(1));
      hipError_t errPeer1 = hipDeviceEnablePeerAccess(0, 0);
      if (errPeer0 != hipSuccess || errPeer1 != hipSuccess) {
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipFree(srcBuffer));
        HIP_CHECK(hipSetDevice(1));
        HIP_CHECK(hipFree(dstBuffer));
        HIP_CHECK(hipSetDevice(0));
        continue;
      }
      HIP_CHECK(hipMemcpyPeer(dstBuffer, 1, srcBuffer, 0, bufSize_));
      auto all_start = std::chrono::steady_clock::now();
      for (unsigned int i = 0; i < numIter; i++) {
        HIP_CHECK(hipMemcpyPeerAsync(dstBuffer, 1, srcBuffer, 0, bufSize_, 0));
      }
      HIP_CHECK(hipSetDevice(1));
      HIP_CHECK(hipDeviceSynchronize());
      hipError_t syncErr = hipGetLastError();
      if (syncErr != hipSuccess) {
        DEBUG_PRINT("WARNING: hipDeviceSynchronize error: %s\n", hipGetErrorString(syncErr));
      }
      HIP_CHECK(hipDeviceSynchronize());
      auto all_end = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed_secs = all_end - all_start;
      DEBUG_PRINT("Elapsed seconds: %f\n", elapsed_secs.count());
      double bufSizeWithIter = static_cast<double>(bufSize_);
      DEBUG_PRINT("%f\n", bufSizeWithIter);
      double perf_pre = bufSizeWithIter / elapsed_secs.count();
      DEBUG_PRINT("%f\n", perf_pre);
      double perf = perf_pre * static_cast<double>(numIter);
      DEBUG_PRINT("%f\n", perf_pre);
      perf *= static_cast<double>(1e-09);
       CONSOLE_PRINT(
           "HIPPerfBufferCopySpeed[%3d]       (%10u bytes)        P2P    s:%-5s d:%-5s       i:%4u  (GB/s) perf     %f",
           testIdx, bufSize_, "dev0", "dev1", numIter, (float)perf);
      void* temp = malloc(bufSize_ + 4096);
      void* chkBuf = reinterpret_cast<void*>(temp);
      HIP_CHECK(hipMemcpy(chkBuf, dstBuffer, bufSize_, hipMemcpyDefault));
      checkData(chkBuf, bufSize_, 0xd0);
      free(temp);
      HIP_CHECK(hipSetDevice(0));
      HIP_CHECK(hipDeviceDisablePeerAccess(1));
      HIP_CHECK(hipSetDevice(1));
      HIP_CHECK(hipDeviceDisablePeerAccess(0));
      HIP_CHECK(hipSetDevice(0));
      HIP_CHECK(hipFree(srcBuffer));
      HIP_CHECK(hipSetDevice(1));
      HIP_CHECK(hipFree(dstBuffer));
      HIP_CHECK(hipSetDevice(0));
      ++testIdx;
    }
  }
  // 2. Run all NoCU (intra) for all sizes
  for (int sizeIdx = 0; sizeIdx < NUM_SIZES; ++sizeIdx) {
    if (p_tests != -1 && testIdx != p_tests) {
      ++testIdx;
      continue;
    }
    unsigned int bufSize_ = Sizes[sizeIdx];
    void* srcBuffer = NULL;
    void* dstBuffer = NULL;
    numIter = Iterations[1];
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMalloc(&srcBuffer, bufSize_));
    HIP_CHECK(hipMalloc(&dstBuffer, bufSize_));
    HIP_CHECK(hipMemset(srcBuffer, 0xd0, bufSize_));
    HIP_CHECK(hipMemcpy(dstBuffer, srcBuffer, bufSize_, hipMemcpyDeviceToDeviceNoCU));
    auto all_start = std::chrono::steady_clock::now();
    for (unsigned int i = 0; i < numIter; i++) {
      HIP_CHECK(hipMemcpyAsync(dstBuffer, srcBuffer, bufSize_, hipMemcpyDeviceToDeviceNoCU, NULL));
    }
    HIP_CHECK(hipDeviceSynchronize());
    hipError_t syncErr = hipGetLastError();
    if (syncErr != hipSuccess) {
      DEBUG_PRINT("WARNING: hipDeviceSynchronize error: %s\n", hipGetErrorString(syncErr));
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto all_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_secs = all_end - all_start;
    DEBUG_PRINT("Elapsed seconds: %f\n", elapsed_secs.count());
    double bufSizeWithIter = static_cast<double>(bufSize_);
    DEBUG_PRINT("%f\n", bufSizeWithIter);
    double perf_pre = bufSizeWithIter / elapsed_secs.count();
    DEBUG_PRINT("%f\n", perf_pre);
    double perf = perf_pre * static_cast<double>(numIter);
    DEBUG_PRINT("%f\n", perf_pre);
    perf *= static_cast<double>(1e-09);
     CONSOLE_PRINT(
         "HIPPerfBufferCopySpeed[%3d]       (%10u bytes)        NoCU   s:%-5s d:%-5s       i:%4u  (GB/s) perf     %f",
         testIdx, bufSize_, "dev0", "dev0", numIter, (float)perf);
    void* temp = malloc(bufSize_ + 4096);
    void* chkBuf = reinterpret_cast<void*>(temp);
    HIP_CHECK(hipMemcpy(chkBuf, dstBuffer, bufSize_, hipMemcpyDefault));
    checkData(chkBuf, bufSize_, 0xd0);
    free(temp);
    HIP_CHECK(hipFree(srcBuffer));
    HIP_CHECK(hipFree(dstBuffer));
    ++testIdx;
  }

  // 3. Run all buffer type (default) for all sizes

  for (int srcTest = 0; srcTest < BUF_TYPES; ++srcTest) {
    for (int dstTest = 0; dstTest < BUF_TYPES; ++dstTest) {
      for (int sizeIdx = 0; sizeIdx < NUM_SIZES; ++sizeIdx) {
        if (p_tests != -1 && testIdx != p_tests) {
          ++testIdx;
          continue;
        }
        unsigned int bufSize_ = Sizes[sizeIdx];
        bool hostMalloc[2] = {false};
        bool hostRegister[2] = {false};
        bool unpinnedMalloc[2] = {false};
        bool deviceMallocUncached[2] = {false};
        void* memptr[2] = {NULL};
        void* alignedmemptr[2] = {NULL};
        void* srcBuffer = NULL;
        void* dstBuffer = NULL;
        numIter = Iterations[1];
        if (srcTest == 4) {
          deviceMallocUncached[0] = true;
        } else if (srcTest == 3) {
          hostRegister[0] = true;
        } else if (srcTest == 2) {
          hostMalloc[0] = true;
        } else if (srcTest == 1) {
          unpinnedMalloc[0] = true;
        }
        if (dstTest == 1) {
          unpinnedMalloc[1] = true;
        } else if (dstTest == 2) {
          hostMalloc[1] = true;
        } else if (dstTest == 3) {
          hostRegister[1] = true;
        } else if (dstTest == 4) {
          deviceMallocUncached[1] = true;
        }
        if (deviceMallocUncached[0]) {
          HIP_CHECK(hipExtMallocWithFlags(&srcBuffer, bufSize_, hipDeviceMallocUncached));
          HIP_CHECK(hipMemset(srcBuffer, 0xd0, bufSize_));
        } else if (hostMalloc[0]) {
          HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&srcBuffer), bufSize_, 0));
          setData(srcBuffer, bufSize_, 0xd0);
        } else if (hostRegister[0]) {
          memptr[0] = malloc(bufSize_ + 4096);
          uintptr_t raw = reinterpret_cast<uintptr_t>(memptr[0]);
          uintptr_t aligned = (raw + 4095) & ~static_cast<uintptr_t>(4095);
          alignedmemptr[0] = reinterpret_cast<void*>(aligned);
          srcBuffer = alignedmemptr[0];
          setData(srcBuffer, bufSize_, 0xd0);
          HIP_CHECK(hipHostRegister(srcBuffer, bufSize_, 0));
        } else if (unpinnedMalloc[0]) {
          memptr[0] = malloc(bufSize_ + 4096);
          uintptr_t raw = reinterpret_cast<uintptr_t>(memptr[0]);
          uintptr_t aligned = (raw + 4095) & ~static_cast<uintptr_t>(4095);
          alignedmemptr[0] = reinterpret_cast<void*>(aligned);
          srcBuffer = alignedmemptr[0];
          setData(srcBuffer, bufSize_, 0xd0);
        } else {
          HIP_CHECK(hipMalloc(&srcBuffer, bufSize_));
          HIP_CHECK(hipMemset(srcBuffer, 0xd0, bufSize_));
        }
        if (deviceMallocUncached[1]) {
          HIP_CHECK(hipExtMallocWithFlags(&dstBuffer, bufSize_, hipDeviceMallocUncached));
        } else if (hostMalloc[1]) {
          HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&dstBuffer), bufSize_, 0));
        } else if (hostRegister[1]) {
          memptr[1] = malloc(bufSize_ + 4096);
          uintptr_t raw = reinterpret_cast<uintptr_t>(memptr[1]);
          uintptr_t aligned = (raw + 4095) & ~static_cast<uintptr_t>(4095);
          alignedmemptr[1] = reinterpret_cast<void*>(aligned);
          dstBuffer = alignedmemptr[1];
          HIP_CHECK(hipHostRegister(dstBuffer, bufSize_, 0));
        } else if (unpinnedMalloc[1]) {
          memptr[1] = malloc(bufSize_ + 4096);
          uintptr_t raw = reinterpret_cast<uintptr_t>(memptr[1]);
          uintptr_t aligned = (raw + 4095) & ~static_cast<uintptr_t>(4095);
          alignedmemptr[1] = reinterpret_cast<void*>(aligned);
          dstBuffer = alignedmemptr[1];
        } else {
          HIP_CHECK(hipMalloc(&dstBuffer, bufSize_));
        }
        HIP_CHECK(hipMemcpy(dstBuffer, srcBuffer, bufSize_, hipMemcpyDefault));
        auto all_start = std::chrono::steady_clock::now();
        for (unsigned int i = 0; i < numIter; i++) {
          HIP_CHECK(hipMemcpyAsync(dstBuffer, srcBuffer, bufSize_, hipMemcpyDefault, NULL));
        }
        HIP_CHECK(hipDeviceSynchronize());
        hipError_t syncErr = hipGetLastError();
        if (syncErr != hipSuccess) {
          DEBUG_PRINT("WARNING: hipDeviceSynchronize error: %s\n", hipGetErrorString(syncErr));
        }
        HIP_CHECK(hipDeviceSynchronize());
        auto all_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_secs = all_end - all_start;
        DEBUG_PRINT("Elapsed seconds: %f\n", elapsed_secs.count());
        double bufSizeWithIter = static_cast<double>(bufSize_);
        DEBUG_PRINT("%f\n", bufSizeWithIter);
        double perf_pre = bufSizeWithIter / elapsed_secs.count();
        DEBUG_PRINT("%f\n", perf_pre);
        double perf = perf_pre * static_cast<double>(numIter);
        DEBUG_PRINT("%f\n", perf_pre);
        perf *= static_cast<double>(1e-09);
        const char* strSrc = NULL;
        const char* strDst = NULL;
        if (deviceMallocUncached[0])
          strSrc = "hMUC";
        else if (hostMalloc[0])
          strSrc = "hHM";
        else if (hostRegister[0])
          strSrc = "hHR";
        else if (unpinnedMalloc[0])
          strSrc = "unp";
        else
          strSrc = "hM";
        if (deviceMallocUncached[1])
          strDst = "hMUC";
        else if (hostMalloc[1])
          strDst = "hHM";
        else if (hostRegister[1])
          strDst = "hHR";
        else if (unpinnedMalloc[1])
          strDst = "unp";
        else
          strDst = "hM";
        if ((!hostMalloc[0] && !hostRegister[0] && !unpinnedMalloc[0]) &&
            (!hostMalloc[1] && !hostRegister[1] && !unpinnedMalloc[1]))
          perf *= 2.0;
        if ((hostMalloc[0] || hostRegister[0] || unpinnedMalloc[0]) &&
            (hostMalloc[1] || hostRegister[1] || unpinnedMalloc[1]))
          perf *= 2.0;
        CONSOLE_PRINT(
            "HIPPerfBufferCopySpeed[%3d]       (%10u bytes)        %-5s  s:%-5s d:%-5s       i:%4u  (GB/s) perf     %f",
            testIdx, bufSize_, "     ", strSrc, strDst, numIter, (float)perf);
        void* temp = malloc(bufSize_ + 4096);
        void* chkBuf = reinterpret_cast<void*>(temp);
        HIP_CHECK(hipMemcpy(chkBuf, dstBuffer, bufSize_, hipMemcpyDefault));
        checkData(chkBuf, bufSize_, 0xd0);
        free(temp);
        if (deviceMallocUncached[0]) {
          HIP_CHECK(hipFree(srcBuffer));
        } else if (hostMalloc[0]) {
          HIP_CHECK(hipHostFree(srcBuffer));
        } else if (hostRegister[0]) {
          HIP_CHECK(hipHostUnregister(srcBuffer));
          free(memptr[0]);
        } else if (unpinnedMalloc[0]) {
          free(memptr[0]);
        } else {
          HIP_CHECK(hipFree(srcBuffer));
        }
        if (deviceMallocUncached[1]) {
          HIP_CHECK(hipFree(dstBuffer));
        } else if (hostMalloc[1]) {
          HIP_CHECK(hipHostFree(dstBuffer));
        } else if (hostRegister[1]) {
          HIP_CHECK(hipHostUnregister(dstBuffer));
          free(memptr[1]);
        } else if (unpinnedMalloc[1]) {
          free(memptr[1]);
        } else {
          HIP_CHECK(hipFree(dstBuffer));
        }
      }
      ++testIdx;
    }
  }
  return true;
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipPerfBufferCopySpeed status.
 * Test source
 * ------------------------
 *  - perftests/memory/hipPerfBufferCopySpeed.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Perf_hipPerfBufferCopySpeed_test") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices <= 0) {
    SUCCEED(
        "Skipped testcase hipPerfBufferCopySpeed as"
        "there is no device to test.");
  } else {
    int deviceId = 0;
    HIP_CHECK(hipSetDevice(deviceId));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

    CONSOLE_PRINT(
        "hipPerfBufferCopySpeed - info: Set device to %d : %s\nLegend: unp - unpinned(malloc), hM "
        "- hipMalloc(device), hHR - hipHostRegister(pinned), hHM - hipHostMalloc(prePinned), hMUC "
        "- hipMallocUncached\n",
        deviceId, props.name);

    CONSOLE_PRINT("Usage: Set HIP_PERF_TEST_NUM environment variable to run a specific subtest (e.g., HIP_PERF_TEST_NUM=5)\n");

    // Run the test with all sizes and buffer types, alter p_tests to run a specific test
    // Method 1: Set HIP_PERF_TEST_NUM environment variable (e.g., HIP_PERF_TEST_NUM=5)
    // Method 2: Directly change the value below (e.g., int p_tests = 5;)
    char* testNumStr = getenv("HIP_PERF_TEST_NUM");
    int p_tests = -1;  // Default: run all tests
    if (testNumStr != nullptr) {
      p_tests = atoi(testNumStr);
      CONSOLE_PRINT("Running specific test: %d\n", p_tests);
    }
    REQUIRE(true == hipPerfBufferCopySpeed_test(p_tests));
  }
}

/**
 * End doxygen group perfMemoryTest.
 * @}
 */
