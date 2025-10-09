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
 * @addtogroup hipMemsetKernel hipMemsetKernel
 * @{
 * @ingroup perfMemoryTest
 * `hipMemset(void* devPtr, int  value, size_t count)` -
 * Initializes or sets device memory to a value.
 */
// #define ENABLE_DEBUG 1
#include <hip_test_common.hh>

static unsigned int sizeList[] = {
    256, 512, 1024, 2048, 4096, 8192,
};

static unsigned int eleNumList[] = {0x100,    0x400,    0x1000,   0x4000,   0x10000,
                                    0x20000,  0x40000,  0x80000,  0x100000, 0x200000,
                                    0x400000, 0x800000, 0x1000000};

typedef struct _dataType {
  char memsetval = 0x42;
  char memsetD8val = 0xDE;
  int16_t memsetD16val = 0xDEAD;
  int memsetD32val = 0xDEADBEEF;
} dataType;

#define NUM_ITER 1000

enum MemsetType {
  hipMemsetTypeDefault,
  hipMemsetTypeD8,
  hipMemsetTypeD16,
  hipMemsetTypeD32,
  hipMemsetTypeMax

};

class hipPerfMemset {
 private:
  uint64_t bufSize_;
  unsigned int num_elements_;
  unsigned int testNumEle_;
  unsigned int _numSubTests = 0;
  unsigned int _numSubTests2D = 0;
  unsigned int _numSubTests3D = 0;
  unsigned int num_sizes_ = 0;

 public:
  hipPerfMemset() {
    num_elements_ = sizeof(eleNumList) / sizeof(unsigned int);
    _numSubTests = num_elements_ * hipMemsetTypeMax;

    num_sizes_ = sizeof(sizeList) / sizeof(unsigned int);
    _numSubTests2D = num_sizes_;
    _numSubTests3D = _numSubTests2D;
  }

  ~hipPerfMemset() {}

  bool open(int deviceID);

  template <typename T>
  void run1D(unsigned int test, T memsetval, enum MemsetType type, bool async);

  template <typename T>
  void run2D(unsigned int test, T memsetval, enum MemsetType type, bool async);

  template <typename T>
  void run3D(unsigned int test, T memsetval, enum MemsetType type, bool async);

  uint getNumTests() { return _numSubTests; }

  uint getNumTests2D() { return _numSubTests2D; }
  uint getNumTests3D() { return _numSubTests3D; }
};

bool hipPerfMemset::open(int deviceId) {
  int nGpu = 0;
  HIP_CHECK(hipGetDeviceCount(&nGpu));
  if (nGpu < 1) {
    return false;
  }

  HIP_CHECK(hipSetDevice(deviceId));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, deviceId));
  CONSOLE_PRINT("info: running on bus 0x%x %s with %d CUs and device id: %d\n", props.pciBusID,
                props.name, props.multiProcessorCount, deviceId);
  return true;
}

template <typename T>
void hipPerfMemset::run1D(unsigned int test, T memsetval, enum MemsetType type, bool async) {
  T *A_h, *A_d;

  testNumEle_ = eleNumList[test % num_elements_];

  bufSize_ = testNumEle_ * sizeof(uint32_t);

  HIP_CHECK(hipMalloc(&A_d, bufSize_));

  A_h = reinterpret_cast<T*>(malloc(bufSize_));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  // Warm-up
  if (async) {
    HIP_CHECK(hipMemsetAsync((void*)A_d, memsetval, bufSize_, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  } else {
    HIP_CHECK(hipMemset((void*)A_d, memsetval, bufSize_));
    HIP_CHECK(hipDeviceSynchronize());
  }

  auto start = std::chrono::steady_clock::now();

  for (uint i = 0; i < NUM_ITER; i++) {
    if (type == hipMemsetTypeDefault && !async) {
      HIP_CHECK(hipMemset(reinterpret_cast<void*>(A_d), memsetval, bufSize_));
    } else if (type == hipMemsetTypeDefault && async) {
      HIP_CHECK(hipMemsetAsync(A_d, memsetval, bufSize_, stream));
    } else if (type == hipMemsetTypeD8 && !async) {
      HIP_CHECK(hipMemsetD8((hipDeviceptr_t)A_d, memsetval, bufSize_));
    } else if (type == hipMemsetTypeD8 && async) {
      HIP_CHECK(hipMemsetD8Async((hipDeviceptr_t)A_d, memsetval, bufSize_, stream));
    } else if (type == hipMemsetTypeD16 && !async) {
      HIP_CHECK(hipMemsetD16((hipDeviceptr_t)A_d, memsetval, bufSize_ / sizeof(T)));
    } else if (type == hipMemsetTypeD16 && async) {
      HIP_CHECK(hipMemsetD16Async((hipDeviceptr_t)A_d, memsetval, bufSize_ / sizeof(T), stream));
    } else if (type == hipMemsetTypeD32 && !async) {
      HIP_CHECK(hipMemsetD32((hipDeviceptr_t)A_d, memsetval, bufSize_ / sizeof(T)));
    } else if (type == hipMemsetTypeD32 && async) {
      HIP_CHECK(hipMemsetD32Async((hipDeviceptr_t)A_d, memsetval, bufSize_ / sizeof(T), stream));
    }
  }
  if (async) {
    HIPCHECK(hipStreamSynchronize(stream));
  } else {
    HIPCHECK(hipDeviceSynchronize());
  }

  auto end = std::chrono::steady_clock::now();

  HIP_CHECK(hipMemcpy(A_h, A_d, bufSize_, hipMemcpyDeviceToHost));

  for (int i = 0; i < bufSize_ / sizeof(T); i++) {
    if (A_h[i] != memsetval) {
      DEBUG_PRINT("mismatch at index %d computed: %d, memsetval: %d\n", i, static_cast<int>(A_h[i]),
                  static_cast<int>(memsetval));
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipFree(A_d));
  free(A_h);

  std::chrono::duration<double> diff = end - start;

  auto sec = diff.count();
  auto perf = static_cast<double>((bufSize_ * NUM_ITER * (1e-09)) / sec);

  std::cout << "[" << std::setw(2) << test << "] " << std::setw(5) << bufSize_ / 1024 << " Kb "
            << std::setw(4) << " typeSize " << sizeof(T) << " : " << std::setw(7) << perf
            << " GB/s \n";
}

template <typename T>
void hipPerfMemset::run2D(unsigned int test, T memsetval, enum MemsetType type, bool async) {
  bufSize_ = sizeList[test % num_sizes_];
  size_t numH = bufSize_;
  size_t numW = bufSize_;
  size_t pitch_A;
  size_t width = numW * sizeof(char);
  size_t sizeElements = width * numH;
  size_t elements = numW * numH;

  T *A_h, *A_d;

  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width, numH));
  A_h = reinterpret_cast<char*>(malloc(sizeElements));

  for (size_t i = 0; i < elements; i++) {
    A_h[i] = 1;
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  // Warm-up
  if (async) {
    HIP_CHECK(hipMemset2DAsync(A_d, pitch_A, memsetval, numW, numH, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  } else {
    HIP_CHECK(hipMemset2D(A_d, pitch_A, memsetval, numW, numH));
    HIP_CHECK(hipDeviceSynchronize());
  }
  auto start = std::chrono::steady_clock::now();

  for (uint i = 0; i < NUM_ITER; i++) {
    if (type == hipMemsetTypeDefault && !async) {
      HIP_CHECK(hipMemset2D(A_d, pitch_A, memsetval, numW, numH));
    } else if (type == hipMemsetTypeDefault && async) {
      HIP_CHECK(hipMemset2DAsync(A_d, pitch_A, memsetval, numW, numH, stream));
    }
  }

  if (async) {
    HIP_CHECK(hipStreamSynchronize(stream));
  } else {
    HIP_CHECK(hipDeviceSynchronize());
  }

  auto end = std::chrono::steady_clock::now();

  HIP_CHECK(hipMemcpy2D(A_h, width, A_d, pitch_A, numW, numH, hipMemcpyDeviceToHost));

  for (int i = 0; i < elements; i++) {
    if (A_h[i] != memsetval) {
      DEBUG_PRINT("mismatch at index %d computed: %d, memsetval: %d\n", i, static_cast<int>(A_h[i]),
                  static_cast<int>(memsetval));
      REQUIRE(false);
    }
  }

  std::chrono::duration<double> diff = end - start;

  auto sec = diff.count();
  auto perf = static_cast<double>((sizeElements * NUM_ITER * (1e-09)) / sec);

  std::cout << "hipPerf2DMemset" << (async ? "Async" : "     ") << "[" << test << "] " << "  "
            << "(GB/s) for " << std::setw(5) << bufSize_ << " x " << std::setw(5) << bufSize_
            << " bytes : " << std::setw(7) << perf << "\n";

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(A_d));
  free(A_h);
}

template <typename T>
void hipPerfMemset::run3D(unsigned int test, T memsetval, enum MemsetType type, bool async) {
  bufSize_ = sizeList[test % num_sizes_];

  size_t numH = bufSize_;
  size_t numW = bufSize_;
  size_t depth = 10;
  size_t width = numW * sizeof(char);
  size_t sizeElements = width * numH * depth;
  size_t elements = numW * numH * depth;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  T* A_h;

  hipExtent extent = make_hipExtent(width, numH, depth);
  hipPitchedPtr devPitchedPtr;

  HIP_CHECK(hipMalloc3D(&devPitchedPtr, extent));
  A_h = reinterpret_cast<char*>(malloc(sizeElements));
  HIPASSERT(A_h != NULL);

  for (size_t i = 0; i < elements; i++) {
    A_h[i] = 1;
  }

  // Warm up
  if (async) {
    HIP_CHECK(hipMemset3DAsync(devPitchedPtr, memsetval, extent, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  } else {
    HIP_CHECK(hipMemset3D(devPitchedPtr, memsetval, extent));
    HIP_CHECK(hipDeviceSynchronize());
  }

  auto start = std::chrono::steady_clock::now();

  for (uint i = 0; i < NUM_ITER; i++) {
    if (type == hipMemsetTypeDefault && !async) {
      HIP_CHECK(hipMemset3D(devPitchedPtr, memsetval, extent));
    } else if (type == hipMemsetTypeDefault && async) {
      HIP_CHECK(hipMemset3DAsync(devPitchedPtr, memsetval, extent, stream));
    }
  }

  if (async) {
    HIP_CHECK(hipStreamSynchronize(stream));
  } else {
    HIP_CHECK(hipDeviceSynchronize());
  }

  auto end = std::chrono::steady_clock::now();

  hipMemcpy3DParms myparms;
  myparms.srcArray = nullptr;
  myparms.dstArray = nullptr;
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.dstPtr = make_hipPitchedPtr(A_h, width, numW, numH);
  myparms.srcPtr = devPitchedPtr;
  myparms.extent = extent;

  myparms.kind = hipMemcpyDeviceToHost;

  HIP_CHECK(hipMemcpy3D(&myparms));

  for (int i = 0; i < elements; i++) {
    if (A_h[i] != memsetval) {
      DEBUG_PRINT("mismatch at index %d computed: %d, memsetval: %d\n", i, static_cast<int>(A_h[i]),
                  static_cast<int>(memsetval));
      REQUIRE(false);
    }
  }

  std::chrono::duration<double> diff = end - start;

  auto sec = diff.count();
  auto perf = static_cast<double>((sizeElements * NUM_ITER * (1e-09)) / sec);

  CONSOLE_PRINT("hipPerf3DMemset%s[%d] (GB/s) for %5zu x %5zu x %zu bytes : %7.2f\n",
                (async ? "Async" : "     "), test, bufSize_, bufSize_, depth, perf);
  HIP_CHECK(hipFree(devPitchedPtr.ptr));
  free(A_h);
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipPerfMemset status.
 * Test source
 * ------------------------
 *  - perftests/memory/hipPerfMemset.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Perf_hipPerfMemset_test") {
  hipPerfMemset hipPerfMemset;

  int deviceId = 0;
  REQUIRE(hipPerfMemset.open(deviceId));

  dataType pattern;

  int numTests = hipPerfMemset.getNumTests();
  int numTests2D = hipPerfMemset.getNumTests2D();
  int numTests3D = hipPerfMemset.getNumTests3D();

  bool async = false;

  for (uint i = 0; i < 2; i++) {
    CONSOLE_PRINT("--------------------- 1D buffer -------------------\n");
    for (auto testCase = 0; testCase < numTests; testCase++) {
      if (testCase < sizeof(eleNumList) / sizeof(uint32_t)) {
        CONSOLE_PRINT("hipMemsetD8%s", (async ? "Async " : "      "));
        hipPerfMemset.run1D(testCase, pattern.memsetval, hipMemsetTypeD8, async);
      } else if (testCase < 2 * sizeof(eleNumList) / sizeof(uint32_t)) {
        CONSOLE_PRINT("hipMemsetD16%s", (async ? "Async" : "     "));
        hipPerfMemset.run1D(testCase, pattern.memsetD16val, hipMemsetTypeD16, async);
      } else if (testCase < 3 * sizeof(eleNumList) / sizeof(uint32_t)) {
        CONSOLE_PRINT("hipMemsetD32%s", (async ? "Async" : "     "));
        hipPerfMemset.run1D(testCase, pattern.memsetD32val, hipMemsetTypeD32, async);
      } else {
        CONSOLE_PRINT("hipMemset%s", (async ? "Async   " : "        "));
        hipPerfMemset.run1D(testCase, pattern.memsetval, hipMemsetTypeDefault, async);
      }
    }
    async = true;
  }

  CONSOLE_PRINT("\n");
  CONSOLE_PRINT("\n------------------ 2D buffer arrays ---------------\n");

  async = false;
  for (uint i = 0; i < 2; i++) {
    CONSOLE_PRINT("\n");
    for (uint test = 0; test < numTests2D; test++) {
      hipPerfMemset.run2D(test, pattern.memsetval, hipMemsetTypeDefault, async);
    }
    async = true;
  }

  CONSOLE_PRINT("\n");
  CONSOLE_PRINT("\n------------------ 3D buffer arrays ---------------\n");

  async = false;
  for (uint i = 0; i < 2; i++) {
    CONSOLE_PRINT("\n");
    for (uint test = 0; test < numTests3D; test++) {
      hipPerfMemset.run3D(test, pattern.memsetval, hipMemsetTypeDefault, async);
    }
    async = true;
  }
}

/**
 * End doxygen group perfMemoryTest.
 * @}
 */
