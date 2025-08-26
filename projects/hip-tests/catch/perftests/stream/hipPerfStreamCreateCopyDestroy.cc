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
 * @addtogroup hipPerfStreamCreateCopyDestroy hipPerfStreamCreateCopyDestroy
 * @{
 * @ingroup perfStreamTest
 * `hipError_t hipStreamCreate(hipStream_t* stream)` -
 * Create an asynchronous stream.
 */

#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>

#define BufSize 0x1000
#define Iterations 0x100
#define TotalStreams 4
#define TotalBufs 4


class hipPerfStreamCreateCopyDestroy {
 private:
  unsigned int numBuffers_;
  unsigned int numStreams_;
  const size_t totalStreams_[TotalStreams];
  const size_t totalBuffers_[TotalBufs];

 public:
  hipPerfStreamCreateCopyDestroy()
      : numBuffers_(0),
        numStreams_(0),
        totalStreams_{1, 2, 4, 8},
        totalBuffers_{1, 100, 1000, 5000} {};
  ~hipPerfStreamCreateCopyDestroy() {};
  bool open(int deviceID);
  bool run(unsigned int testNumber);
};

bool hipPerfStreamCreateCopyDestroy::open(int deviceId) {
  int nGpu = 0;
  HIP_CHECK(hipGetDeviceCount(&nGpu));
  if (nGpu < 1) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 1");
  }
  HIP_CHECK(hipSetDevice(deviceId));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

  CONSOLE_PRINT("info: running on bus 0x%x %s with %d CUs and device id: %d\n", props.pciBusID,
                props.name, props.multiProcessorCount, deviceId);
  return true;
}

bool hipPerfStreamCreateCopyDestroy::run(unsigned int testNumber) {
  numStreams_ = totalStreams_[testNumber % TotalStreams];
  size_t iter =
      Iterations / (numStreams_ * (static_cast<size_t>(1) << (testNumber / TotalBufs + 1)));
  hipStream_t* streams = new hipStream_t[numStreams_];

  numBuffers_ = totalBuffers_[testNumber / TotalBufs];
  float** dSrc = new float*[numBuffers_];
  size_t nBytes = BufSize * sizeof(float);

  for (size_t b = 0; b < numBuffers_; ++b) {
    HIP_CHECK(hipMalloc(&dSrc[b], nBytes));
  }

  float* hSrc;
  hSrc = new float[nBytes];
  HIP_CHECK(hSrc == 0 ? hipErrorOutOfMemory : hipSuccess);
  for (size_t i = 0; i < BufSize; i++) {
    hSrc[i] = 1.618f + i;
  }

  auto start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < iter; ++i) {
    for (size_t s = 0; s < numStreams_; ++s) {
      HIP_CHECK(hipStreamCreate(&streams[s]));
    }

    for (size_t s = 0; s < numStreams_; ++s) {
      for (size_t b = 0; b < numBuffers_; ++b) {
        HIP_CHECK(hipMemcpyWithStream(dSrc[b], hSrc, nBytes, hipMemcpyHostToDevice, streams[s]));
      }
    }

    for (size_t s = 0; s < numStreams_; ++s) {
      HIP_CHECK(hipStreamDestroy(streams[s]));
    }
  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff = end - start;

  auto time = static_cast<float>(diff.count() * 1000 / (iter * numStreams_));

  CONSOLE_PRINT(
      "Create+Copy+Destroy time for %u streams and %u buffers and %zu iterations %.6f (ms)\n",
      numStreams_, numBuffers_, iter, time);

  delete[] hSrc;
  for (size_t b = 0; b < numBuffers_; ++b) {
    HIP_CHECK(hipFree(dSrc[b]));
  }

  delete[] streams;
  delete[] dSrc;
  return true;
}

/**
 * Test Description
 * ------------------------
 *  - Verify the Create+Copy+Destroy time for different stream.
 * Test source
 * ------------------------
 *  - perftests/stream/hipPerfDeviceConcurrency.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Perf_hipPerfStreamCreateCopyDestroy") {
  hipPerfStreamCreateCopyDestroy streamCCD;
  int deviceId = 0;
  REQUIRE(true == streamCCD.open(deviceId));

  for (auto testCase = 0; testCase < TotalStreams * TotalBufs; testCase++) {
    REQUIRE(true == streamCCD.run(testCase));
  }
}

/**
 * End doxygen group perfStreamTest.
 * @}
 */
