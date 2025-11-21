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
 * @addtogroup hipMemcpy2DAsync hipMemcpy2DAsync
 * @{
 * @ingroup perfMemoryTest
 * `hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
 *     size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream = 0)` -
 * Copies data between host and device.
 */

#include <hip_test_common.hh>
// #define ENABLE_DEBUG 1

#define NUM_SIZES 8
//  4KB, 8KB, 64KB, 256KB, 1 MB, 4MB, 16 MB, 16MB+10
static const unsigned int Sizes[NUM_SIZES] = {4096,    8192,    65536,    262144,
                                              1048576, 4194304, 16777216, 16777216 + 10};

static const unsigned int Iterations[2] = {1, 1000};

#define BUF_TYPES 4
//  16 ways to combine 4 different buffer types
#define NUM_SUBTESTS (BUF_TYPES * BUF_TYPES)

static void setData(void* ptr, unsigned int size, char value) {
  char* ptr2 = reinterpret_cast<char*>(ptr);
  for (unsigned int i = 0; i < size; i++) {
    ptr2[i] = value;
  }
}

static bool hipPerfBufferCopyRectSpeed_test(int p_tests) {
  unsigned int bufSize_;
  unsigned int numIter;
  bool hostMalloc[2] = {false};
  bool hostRegister[2] = {false};
  bool unpinnedMalloc[2] = {false};
  void* memptr[2] = {NULL};
  void* alignedmemptr[2] = {NULL};
  void* srcBuffer = NULL;
  void* dstBuffer = NULL;

  int numTests = (p_tests == -1) ? (NUM_SIZES * NUM_SUBTESTS * 2 - 1) : p_tests;
  int test = (p_tests == -1) ? 0 : p_tests;

  for (; test <= numTests; test++) {
    unsigned int srcTest = (test / NUM_SIZES) % BUF_TYPES;
    unsigned int dstTest = (test / (NUM_SIZES * BUF_TYPES)) % BUF_TYPES;
    bufSize_ = Sizes[test % NUM_SIZES];
    hostMalloc[0] = hostMalloc[1] = false;
    hostRegister[0] = hostRegister[1] = false;
    unpinnedMalloc[0] = unpinnedMalloc[1] = false;
    srcBuffer = dstBuffer = 0;
    memptr[0] = memptr[1] = 0;
    alignedmemptr[0] = alignedmemptr[1] = NULL;

    size_t width = static_cast<size_t>(sqrt(static_cast<float>(bufSize_)));

    if (srcTest == 3) {
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
    }

    numIter = Iterations[test / (NUM_SIZES * NUM_SUBTESTS)];

    if (hostMalloc[0]) {
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&srcBuffer), bufSize_, 0));
      setData(srcBuffer, bufSize_, 0xd0);
    } else if (hostRegister[0]) {
      memptr[0] = malloc(bufSize_ + 4096);
      alignedmemptr[0] = reinterpret_cast<void*>(memptr[0]);
      srcBuffer = alignedmemptr[0];
      setData(srcBuffer, bufSize_, 0xd0);
      HIP_CHECK(hipHostRegister(srcBuffer, bufSize_, 0));
    } else if (unpinnedMalloc[0]) {
      memptr[0] = malloc(bufSize_ + 4096);
      alignedmemptr[0] = reinterpret_cast<void*>(memptr[0]);
      srcBuffer = alignedmemptr[0];
      setData(srcBuffer, bufSize_, 0xd0);
    } else {
      HIP_CHECK(hipMalloc(&srcBuffer, bufSize_));
      HIP_CHECK(hipMemset(srcBuffer, 0xd0, bufSize_));
    }

    if (hostMalloc[1]) {
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&dstBuffer), bufSize_, 0));
    } else if (hostRegister[1]) {
      memptr[1] = malloc(bufSize_ + 4096);
      alignedmemptr[1] = reinterpret_cast<void*>(memptr[0]);
      dstBuffer = alignedmemptr[1];
      HIP_CHECK(hipHostRegister(dstBuffer, bufSize_, 0));
    } else if (unpinnedMalloc[1]) {
      memptr[1] = malloc(bufSize_ + 4096);
      alignedmemptr[1] = reinterpret_cast<void*>(memptr[0]);
      dstBuffer = alignedmemptr[1];
    } else {
      HIP_CHECK(hipMalloc(&dstBuffer, bufSize_));
    }

    //  warm up
    HIP_CHECK(hipMemcpy2D(dstBuffer, width, srcBuffer, width, width, width, hipMemcpyDefault));

    // measure performance based on host time
    auto all_start = std::chrono::steady_clock::now();

    for (unsigned int i = 0; i < numIter; i++) {
      HIP_CHECK(hipMemcpy2DAsync(dstBuffer, width, srcBuffer, width, width, width, hipMemcpyDefault,
                                 NULL));
    }
    HIP_CHECK(hipDeviceSynchronize());

    auto all_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_secs = all_end - all_start;

    // read speed in GB/s
    double perf = (static_cast<double>(bufSize_ * numIter) * static_cast<double>(1e-09)) /
                  elapsed_secs.count();

    const char* strSrc = NULL;
    const char* strDst = NULL;
    if (hostMalloc[0])
      strSrc = "hHM";
    else if (hostRegister[0])
      strSrc = "hHR";
    else if (unpinnedMalloc[0])
      strSrc = "unp";
    else
      strSrc = "hM";

    if (hostMalloc[1])
      strDst = "hHM";
    else if (hostRegister[1])
      strDst = "hHR";
    else if (unpinnedMalloc[1])
      strDst = "unp";
    else
      strDst = "hM";

    // Double results when src and dst are both on device
    if ((!hostMalloc[0] && !hostRegister[0] && !unpinnedMalloc[0]) &&
        (!hostMalloc[1] && !hostRegister[1] && !unpinnedMalloc[1]))
      perf *= 2.0;
    // Double results when src and dst are both in sysmem
    if ((hostMalloc[0] || hostRegister[0] || unpinnedMalloc[0]) &&
        (hostMalloc[1] || hostRegister[1] || unpinnedMalloc[1]))
      perf *= 2.0;

    CONSOLE_PRINT("hipPerfBufferCopyRectSpeed[%d]\t( %u )\ts:%s d:%s\ti:%u\t(GB/s) perf\t%.2f\n",
                  test, bufSize_, strSrc, strDst, numIter, (float)perf);

    //  Free src
    if (hostMalloc[0]) {
      HIP_CHECK(hipHostFree(srcBuffer));
    } else if (hostRegister[0]) {
      HIP_CHECK(hipHostUnregister(srcBuffer));
      free(memptr[0]);
    } else if (unpinnedMalloc[0]) {
      free(memptr[0]);
    } else {
      HIP_CHECK(hipFree(srcBuffer));
    }

    //  Free dst
    if (hostMalloc[1]) {
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
  return true;
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipPerfBufferCopy status.
 * Test source
 * ------------------------
 *  - perftests/memory/hipPerfBufferCopyRectSpeed.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Perf_hipPerfBufferCopyRectSpeed_test") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices <= 0) {
    SUCCEED(
        "Skipped testcase hipPerfBufferCopyRectSpeed"
        "as there is no device to test.");
  } else {
    int deviceId = 0;
    HIP_CHECK(hipSetDevice(deviceId));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

    CONSOLE_PRINT(
        "hipPerfBufferCopyRectSpeed - info: Set device to %d : %s Legend: unp - unpinned(malloc), "
        "hM - hipMalloc(device)\n        hHR - hipHostRegister(pinned), hHM - "
        "hipHostMalloc(prePinned)\n",
        deviceId, props.name);

    REQUIRE(true == hipPerfBufferCopyRectSpeed_test(1));
  }
}

/**
 * End doxygen group perfMemoryTest.
 * @}
 */
