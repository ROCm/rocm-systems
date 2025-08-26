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
 * @addtogroup hipMemcpyKernel hipMemcpyKernel
 * @{
 * @ingroup perfMemoryTest
 * `hipMemcpy(void* dst, const void* src, size_t count, hipMemcpyKind kind)` -
 * Copies data between host and device.
 */

#include <hip_test_common.hh>
// #define ENABLE_DEBUG 1
#define NUM_TYPES 3
std::vector<std::string> types = {"float", "float2", "float4"};
std::vector<unsigned int> typeSizes = {4, 8, 16};

#define NUM_SIZES 12
std::vector<unsigned int> sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

#define NUM_BUFS 6
#define MAX_BUFS (1 << (NUM_BUFS - 1))

#ifdef __HIP_PLATFORM_NVIDIA__
__host__ __device__ void operator+=(float2& a, float2 b) {  // NOLINT
  a.x += b.x;
  a.y += b.y;
}

__host__ __device__ void operator+=(float4& a, float4 b) {  // NOLINT
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
#endif

template <typename T> __global__ void sampleRate(T* outBuffer, unsigned int inBufSize,
                                                 unsigned int writeIt, T** inBuffer, int numBufs) {
  uint gid = (blockIdx.x * blockDim.x + threadIdx.x);
  uint inputIdx = gid % inBufSize;

  T tmp;
  memset(&tmp, 0, sizeof(T));
  for (int i = 0; i < numBufs; i++) {
    tmp += *(*(inBuffer + i) + inputIdx);
  }

  if (writeIt * (unsigned int)tmp.x) {
    outBuffer[gid] = tmp;
  }
}

template <typename T> __global__ void sampleRateFloat(T* outBuffer, unsigned int inBufSize,
                                                      unsigned int writeIt, T** inBuffer,
                                                      int numBufs) {
  uint gid = (blockIdx.x * blockDim.x + threadIdx.x);
  uint inputIdx = gid % inBufSize;

  T tmp = (T)0.0f;

  for (int i = 0; i < numBufs; i++) {
    tmp += *((*inBuffer + i) + inputIdx);
  }

  if (writeIt * (unsigned int)tmp) {
    outBuffer[gid] = tmp;
  }
}

class hipPerfSampleRate {
 public:
  hipPerfSampleRate();
  ~hipPerfSampleRate();

  bool open(void);
  void run(unsigned int testCase);
  void close(void);

  // array of funtion pointers
  typedef void (hipPerfSampleRate::*funPtr)(void* outBuffer, unsigned int inBufSize,
                                            unsigned int writeIt, void** inBuffer, int numBufs,
                                            int grids, int blocks);

  // Wrappers
  void float_kernel(void* outBuffer, unsigned int inBufSize, unsigned int writeIt, void** inBuffer,
                    int numBufs, int grids, int blocks);

  void float2_kernel(void* outBuffer, unsigned int inBufSize, unsigned int writeIt, void** inBuffer,
                     int numBufs, int grids, int blocks);

  void float4_kernel(void* outBuffer, unsigned int inBufSize, unsigned int writeIt, void** inBuffer,
                     int numBufs, int grids, int blocks);

 private:
  void setData(void* ptr, unsigned int value);
  void checkData(uint* ptr);

  unsigned int width_;
  unsigned int bufSize_;
  int numCUs;

  unsigned int outBufSize_;
  static const unsigned int MAX_ITERATIONS = 25;
  unsigned int numBufs_;
  unsigned int typeIdx_;
};

hipPerfSampleRate::hipPerfSampleRate() {}
hipPerfSampleRate::~hipPerfSampleRate() {}
void hipPerfSampleRate::close() {}

bool hipPerfSampleRate::open(void) {
  int nGpu = 0;
  HIP_CHECK(hipGetDeviceCount(&nGpu));
  if (nGpu < 1) {
    return false;
  }

  int deviceId = 0;
  hipDeviceProp_t props;
  HIP_CHECK(hipSetDevice(deviceId));
  HIP_CHECK(hipGetDeviceProperties(&props, deviceId));
  CONSOLE_PRINT("info: running on bus 0x%x %s with %d CUs and device id: %d\n", props.pciBusID,
                props.name, props.multiProcessorCount, deviceId);
  numCUs = props.multiProcessorCount;
  return true;
}

// Wrappers for the kernel launches
void hipPerfSampleRate::float_kernel(void* outBuffer, unsigned int inBufSize, unsigned int writeIt,
                                     void** inBuffer, int numBufs, int grids, int blocks) {
  hipLaunchKernelGGL(sampleRateFloat<float>, dim3(grids, grids, grids), dim3(blocks), 0, 0,
                     reinterpret_cast<float*>(outBuffer), inBufSize, writeIt,
                     reinterpret_cast<float**>(inBuffer), numBufs);
}

void hipPerfSampleRate::float2_kernel(void* outBuffer, unsigned int inBufSize, unsigned int writeIt,
                                      void** inBuffer, int grids, int blocks, int numBufs) {
  hipLaunchKernelGGL(sampleRate<float2>, dim3(grids, grids, grids), dim3(blocks), 0, 0,
                     reinterpret_cast<float2*>(outBuffer), inBufSize, writeIt,
                     reinterpret_cast<float2**>(inBuffer), numBufs);
}

void hipPerfSampleRate::float4_kernel(void* outBuffer, unsigned int inBufSize, unsigned int writeIt,
                                      void** inBuffer, int grids, int blocks, int numBufs) {
  hipLaunchKernelGGL(sampleRate<float4>, dim3(grids, grids, grids), dim3(blocks), 0, 0,
                     reinterpret_cast<float4*>(outBuffer), inBufSize, writeIt,
                     reinterpret_cast<float4**>(inBuffer), numBufs);
}

void hipPerfSampleRate::run(unsigned int test) {
  funPtr p[] = {&hipPerfSampleRate::float_kernel, &hipPerfSampleRate::float2_kernel,
                &hipPerfSampleRate::float4_kernel};

  // We compute a square domain
  width_ = sizes[test % NUM_SIZES];
  typeIdx_ = (test / NUM_SIZES) % NUM_TYPES;
  bufSize_ = width_ * width_ * typeSizes[typeIdx_];
  numBufs_ = (1 << (test / (NUM_SIZES * NUM_TYPES)));

  void** dPtr;
  void* hOutPtr;
  void* dOutPtr;
  void** hInPtr = new void*[numBufs_];
  void** dInPtr = new void*[numBufs_];

  outBufSize_ = sizes[NUM_SIZES - 1] * sizes[NUM_SIZES - 1] * typeSizes[NUM_TYPES - 1];

  // Allocate memory on the host and device
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&hOutPtr), outBufSize_, hipHostMallocDefault));
  setData(reinterpret_cast<void*>(hOutPtr), 0xdeadbeef);
  HIP_CHECK(hipMalloc(reinterpret_cast<uint**>(&dOutPtr), outBufSize_));

  // Allocate 2D array in Device
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dPtr), numBufs_ * sizeof(void*)));

  for (uint i = 0; i < numBufs_; i++) {
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&hInPtr[i]), bufSize_, hipHostMallocDefault));
    HIP_CHECK(hipMalloc(reinterpret_cast<uint**>(&dInPtr[i]), bufSize_));
    setData(hInPtr[i], 0x3f800000);
  }

  // Populate array of pointers with array addresses
  HIP_CHECK(hipMemcpy(dPtr, dInPtr, numBufs_ * sizeof(void*), hipMemcpyHostToDevice));

  // Copy memory from host to device
  for (uint i = 0; i < numBufs_; i++) {
    HIP_CHECK(hipMemcpy(dInPtr[i], hInPtr[i], bufSize_, hipMemcpyHostToDevice));
  }

  HIP_CHECK(hipMemcpy(dOutPtr, hOutPtr, outBufSize_, hipMemcpyHostToDevice));

  // Prepare kernel launch parameters
  // outBufSize_/sizeof(uint) - Grid size in 3D
  int grids = 64;
  int blocks = 64;

  unsigned int maxIter = MAX_ITERATIONS * (MAX_BUFS / numBufs_);
  unsigned int sizeDW = width_ * width_;
  unsigned int writeIt = 0;

  int idx = 0;

  if (!types[typeIdx_].compare("float")) {
    idx = 0;
  } else if (!types[typeIdx_].compare("float2")) {
    idx = 1;
  } else if (!types[typeIdx_].compare("float4")) {
    idx = 2;
  }

  // Time the kernel execution
  auto all_start = std::chrono::steady_clock::now();
  for (uint i = 0; i < maxIter; i++) {
    (this->*p[idx])(reinterpret_cast<void*>(dOutPtr), sizeDW, writeIt, dPtr, numBufs_, grids,
                    blocks);
  }

  HIP_CHECK(hipDeviceSynchronize());
  auto all_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> all_kernel_time = all_end - all_start;

  double perf =
      (static_cast<double>(outBufSize_ * numBufs_ * maxIter * (1e-09))) / all_kernel_time.count();

  CONSOLE_PRINT("Domain %u x %u bufs %u %s %u x %u (GB/s) %f\n", sizes[NUM_SIZES - 1],
                sizes[NUM_SIZES - 1], numBufs_, types[typeIdx_].c_str(), width_, width_, perf);

  HIP_CHECK(hipFree(dOutPtr));

  // Free host and device memory
  for (uint i = 0; i < numBufs_; i++) {
    HIP_CHECK(hipHostFree(hInPtr[i]));
    HIP_CHECK(hipFree(dInPtr[i]));
  }
  HIP_CHECK(hipHostFree(hOutPtr));
  HIP_CHECK(hipFree(dPtr));
  delete[] hInPtr;
  delete[] dInPtr;
}


void hipPerfSampleRate::setData(void* ptr, unsigned int value) {
  unsigned int* ptr2 = (unsigned int*)ptr;
  for (unsigned int i = 0; i < bufSize_ / sizeof(unsigned int); i++) {
    ptr2[i] = value;
  }
}


void hipPerfSampleRate::checkData(uint* ptr) {
  for (unsigned int i = 0; i < outBufSize_ / sizeof(float); i++) {
    if (ptr[i] != static_cast<float>(numBufs_)) {
      DEBUG_PRINT("Data validation failed at %u Got %u, expected %f\n", i, ptr[i], (float)numBufs_);
      REQUIRE(false);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipPerfSampleRate status.
 * Test source
 * ------------------------
 *  - perftests/memory/hipPerfSampleRate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Perf_hipPerfSampleRate_test") {
  hipPerfSampleRate sampleTypes;

  REQUIRE(true == sampleTypes.open());

  for (unsigned int testCase = 0; testCase < 216; testCase += 36) {
    sampleTypes.run(testCase);
  }
}

/**
 * End doxygen group perfMemoryTest.
 * @}
 */
