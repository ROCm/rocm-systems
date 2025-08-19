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
 * @addtogroup hipPerfStreamConcurrency hipPerfStreamConcurrency
 * @{
 * @ingroup perfComputeTest
 * `hipError_t hipStreamCreate(hipStream_t* stream)` -
 * Create an asynchronous stream.
 */

#include <hip_test_common.hh>
#include <hip/hip_vector_types.h>

#ifdef __HIP_PLATFORM_NVIDIA__
inline __device__ float4 operator*(float s, float4 a) {
  return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline __device__ float4 operator*(float4 a, float4 b) {
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __device__ float4 operator+(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __device__ float4 operator-(float4 a, float4 b) {
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
#endif

typedef struct {
  double x;
  double y;
  double width;
} coordRec;

static coordRec coords[] = {
    {0.0, 0.0, 0.00001},  // All black
};

static unsigned int numCoords = sizeof(coords) / sizeof(coordRec);

__global__ static void mandelbrot(uint* out, uint width, float xPos, float yPos, float xStep,
                                  float yStep, uint maxIter) {
  int tid = (blockIdx.x * blockDim.x + threadIdx.x);
  int i = tid % (width / 4);
  int j = tid / (width / 4);
  int4 veci = make_int4(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3);
  int4 vecj = make_int4(j, j, j, j);
  float4 x0;
  x0.x = static_cast<float>(xPos + xStep * veci.x);
  x0.y = static_cast<float>(xPos + xStep * veci.y);
  x0.z = static_cast<float>(xPos + xStep * veci.z);
  x0.w = static_cast<float>(xPos + xStep * veci.w);
  float4 y0;
  y0.x = static_cast<float>(yPos + yStep * vecj.x);
  y0.y = static_cast<float>(yPos + yStep * vecj.y);
  y0.z = static_cast<float>(yPos + yStep * vecj.z);
  y0.w = static_cast<float>(yPos + yStep * vecj.w);
  float4 x = x0;
  float4 y = y0;
  uint iter = 0;
  float4 tmp;
  int4 stay;
  int4 ccount = make_int4(0, 0, 0, 0);
  float4 savx = x;
  float4 savy = y;
  stay.x = (x.x * x.x + y.x * y.x) <= static_cast<float>(4.0f);
  stay.y = (x.y * x.y + y.y * y.y) <= static_cast<float>(4.0f);
  stay.z = (x.z * x.z + y.z * y.z) <= static_cast<float>(4.0f);
  stay.w = (x.w * x.w + y.w * y.w) <= static_cast<float>(4.0f);
  for (iter = 0; (stay.x | stay.y | stay.z | stay.w) && (iter < maxIter); iter += 16) {
    x = savx;
    y = savy;
    // Two iterations
    tmp = x * x + x0 - y * y;
    y = 2.0f * x * y + y0;
    x = tmp * tmp + x0 - y * y;
    y = 2.0f * tmp * y + y0;
    // Two iterations
    tmp = x * x + x0 - y * y;
    y = 2.0f * x * y + y0;
    x = tmp * tmp + x0 - y * y;
    y = 2.0f * tmp * y + y0;
    // Two iterations
    tmp = x * x + x0 - y * y;
    y = 2.0f * x * y + y0;
    x = tmp * tmp + x0 - y * y;
    y = 2.0f * tmp * y + y0;
    // Two iterations
    tmp = x * x + x0 - y * y;
    y = 2.0f * x * y + y0;
    x = tmp * tmp + x0 - y * y;
    y = 2.0f * tmp * y + y0;
    // Two iterations
    tmp = x * x + x0 - y * y;
    y = 2.0f * x * y + y0;
    x = tmp * tmp + x0 - y * y;
    y = 2.0f * tmp * y + y0;
    // Two iterations
    tmp = x * x + x0 - y * y;
    y = 2.0f * x * y + y0;
    x = tmp * tmp + x0 - y * y;
    y = 2.0f * tmp * y + y0;
    // Two iterations
    tmp = x * x + x0 - y * y;
    y = 2.0f * x * y + y0;
    x = tmp * tmp + x0 - y * y;
    y = 2.0f * tmp * y + y0;
    stay.x = (x.x * x.x + y.x * y.x) <= static_cast<float>(4.0f);
    stay.y = (x.y * x.y + y.y * y.y) <= static_cast<float>(4.0f);
    stay.z = (x.z * x.z + y.z * y.z) <= static_cast<float>(4.0f);
    stay.w = (x.w * x.w + y.w * y.w) <= static_cast<float>(4.0f);
    savx.x = static_cast<bool>(stay.x ? x.x : savx.x);
    savx.y = static_cast<bool>(stay.y ? x.y : savx.y);
    savx.z = static_cast<bool>(stay.z ? x.z : savx.z);
    savx.w = static_cast<bool>(stay.w ? x.w : savx.w);
    savy.x = static_cast<bool>(stay.x ? y.x : savy.x);
    savy.y = static_cast<bool>(stay.y ? y.y : savy.y);
    savy.z = static_cast<bool>(stay.z ? y.z : savy.z);
    savy.w = static_cast<bool>(stay.w ? y.w : savy.w);
    ccount.x -= stay.x * 16;
    ccount.y -= stay.y * 16;
    ccount.z -= stay.z * 16;
    ccount.w -= stay.w * 16;
  }
  // Handle remainder
  if (!(stay.x & stay.y & stay.z & stay.w)) {
    iter = 16;
    do {
      x = savx;
      y = savy;
      stay.x = ((x.x * x.x + y.x * y.x) <= 4.0f) && (ccount.x < maxIter);
      stay.y = ((x.y * x.y + y.y * y.y) <= 4.0f) && (ccount.y < maxIter);
      stay.z = ((x.z * x.z + y.z * y.z) <= 4.0f) && (ccount.z < maxIter);
      stay.w = ((x.w * x.w + y.w * y.w) <= 4.0f) && (ccount.w < maxIter);
      tmp = x;
      x = x * x + x0 - y * y;
      y = 2.0f * tmp * y + y0;
      ccount.x += stay.x;
      ccount.y += stay.y;
      ccount.z += stay.z;
      ccount.w += stay.w;
      iter--;
      savx.x = (stay.x ? x.x : savx.x);
      savx.y = (stay.y ? x.y : savx.y);
      savx.z = (stay.z ? x.z : savx.z);
      savx.w = (stay.w ? x.w : savx.w);
      savy.x = (stay.x ? y.x : savy.x);
      savy.y = (stay.y ? y.y : savy.y);
      savy.z = (stay.z ? y.z : savy.z);
      savy.w = (stay.w ? y.w : savy.w);
    } while ((stay.x | stay.y | stay.z | stay.w) && iter);
  }
  uint4* vecOut = reinterpret_cast<uint4*>(out);
  vecOut[tid].x = (uint)(ccount.x);
  vecOut[tid].y = (uint)(ccount.y);
  vecOut[tid].z = (uint)(ccount.z);
  vecOut[tid].w = (uint)(ccount.w);
}

class hipPerfStreamConcurrency {
 public:
  hipPerfStreamConcurrency();
  ~hipPerfStreamConcurrency();

  void setNumKernels(unsigned int num) { numKernels = num; }
  void setNumStreams(unsigned int num) { numStreams = num; }
  unsigned int getNumStreams() { return numStreams; }

  unsigned int getNumKernels() { return numKernels; }

  bool open(int deviceID);
  bool run(unsigned int testCase, unsigned int deviceId);
  void close(void);

 private:
  void setData(void* ptr, unsigned int value);
  void checkData(uint* ptr);

  unsigned int numKernels;
  unsigned int numStreams;

  unsigned int width_;
  unsigned int bufSize;
  unsigned int maxIter;
  unsigned int coordIdx;
  unsigned long long totalIters;
  int numCUs;
};

hipPerfStreamConcurrency::hipPerfStreamConcurrency() {}

hipPerfStreamConcurrency::~hipPerfStreamConcurrency() {}

bool hipPerfStreamConcurrency::open(int deviceId) {
  int nGpu = 0;
  HIP_CHECK(hipGetDeviceCount(&nGpu));
  if (nGpu < 1) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 1");
  }

  HIP_CHECK(hipSetDevice(deviceId));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, deviceId));
  CONSOLE_PRINT("info: running on bus 0x%x %s with %d CUs and device ID: %d", props.pciBusID,
                props.name, props.multiProcessorCount, deviceId);

  numCUs = props.multiProcessorCount;
  return true;
}

void hipPerfStreamConcurrency::close() {}

bool hipPerfStreamConcurrency::run(unsigned int testCase, unsigned int deviceId) {
  int clkFrequency = 0;
  unsigned int numStreams = getNumStreams();
  unsigned int numKernels = getNumKernels();

  HIP_CHECK(hipDeviceGetAttribute(&clkFrequency, hipDeviceAttributeClockRate, deviceId));
  if (clkFrequency == 0) {
    CONSOLE_PRINT("clkFrequency = 0, set it to 1000000\n");
    clkFrequency = 1000000;
  }
  clkFrequency = (unsigned int)clkFrequency / 1000;

  // Maximum iteration count
  // maxIter = 8388608 * (engine_clock / 1000).serial execution
  maxIter = (unsigned int)(((8388608 * (static_cast<float>(clkFrequency) / 1000)) * numCUs) / 128);
  maxIter = (maxIter + 15) & ~15;
  hipStream_t* streams = new hipStream_t[numStreams];
  uint** hPtr = new uint*[numKernels];
  uint** dPtr = new uint*[numKernels];

  // Width is divisible by 4 because the mandelbrot kernel
  // processes 4 pixels at once.
  width_ = 256;
  bufSize = width_ * sizeof(uint);
  // Create streams for concurrency
  for (uint i = 0; i < numStreams; i++) {
    HIP_CHECK(hipStreamCreate(&streams[i]));
  }

  // Allocate memory on the host and device
  for (uint i = 0; i < numKernels; i++) {
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&hPtr[i]), bufSize, hipHostMallocDefault));
    setData(hPtr[i], 0xdeadbeef);
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dPtr[i]), bufSize))
  }

  // Prepare kernel launch parameters
  int threads = (bufSize / sizeof(uint));
  int threads_per_block = 64;
  int blocks = (threads / threads_per_block) + (threads % threads_per_block);
  coordIdx = testCase % numCoords;
  float xStep = static_cast<float>(coords[coordIdx].width / static_cast<double>(width_));
  float yStep = static_cast<float>(-coords[coordIdx].width / static_cast<double>(width_));
  float xPos = static_cast<float>(coords[coordIdx].x - 0.5 * coords[coordIdx].width);
  float yPos = static_cast<float>(coords[coordIdx].y + 0.5 * coords[coordIdx].width);

  // Copy memory asynchronously and concurrently from host to device
  for (uint i = 0; i < numKernels; i++) {
    HIP_CHECK(hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(dPtr[i]), hPtr[i], bufSize,
                                 streams[i % numStreams]));
  }

  // Synchronize to make sure all the copies are completed
  for (uint i = 0; i < numStreams; i++) {
    HIP_CHECK(hipStreamSynchronize(streams[i]));
  }
  // Warm-up kernel with lower iteration
  if (testCase == 0) {
    maxIter = 256;
  }
  // Time the kernel execution
  auto all_start = std::chrono::steady_clock::now();

  for (uint i = 0; i < numKernels; i++) {
    hipLaunchKernelGGL(mandelbrot, dim3(blocks), dim3(threads_per_block), 0,
                       streams[i % numStreams], dPtr[i], width_, xPos, yPos, xStep, yStep, maxIter);
  }

  // Synchronize all the concurrent streans to have completed execution
  for (uint i = 0; i < numStreams; i++) {
    HIP_CHECK(hipStreamSynchronize(streams[i]));
  }

  auto all_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> all_kernel_time = all_end - all_start;

  // Copy data back from device to the host
  for (uint i = 0; i < numKernels; i++) {
    HIP_CHECK(hipMemcpyDtoHAsync(hPtr[i], reinterpret_cast<hipDeviceptr_t>(dPtr[i]), bufSize,
                                 streams[i % numStreams]));
  }

  if (testCase != 0) {
    CONSOLE_PRINT("Measured time for %d kernels (s) on %d stream(s): %e\n", numKernels, numStreams,
                  all_kernel_time.count());
  }

  for (uint i = 0; i < numStreams; i++) {
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }

  // Free host and device memory
  for (uint i = 0; i < numKernels; i++) {
    HIP_CHECK(hipHostFree(hPtr[i]));
    HIP_CHECK(hipFree(dPtr[i]));
  }

  delete[] streams;
  delete[] hPtr;
  delete[] dPtr;
  return true;
}

void hipPerfStreamConcurrency::setData(void* ptr, unsigned int value) {
  unsigned int* ptr2 = (unsigned int*)ptr;
  for (unsigned int i = 0; i < width_; i++) {
    ptr2[i] = value;
  }
}

void hipPerfStreamConcurrency::checkData(uint* ptr) {
  totalIters = 0;
  for (unsigned int i = 0; i < width_; i++) {
    totalIters += ptr[i];
  }
}

/**
 * Test Description
 * ------------------------
 *  - Verify the different levels of stream concurrency.
 * Test source
 * ------------------------
 *  - perftests/stream/hipPerfStreamConcurrency.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Perf_hipPerfStreamConcurrency") {
  hipPerfStreamConcurrency streamConcurrency;
  int deviceId = 0;
  REQUIRE(true == streamConcurrency.open(deviceId));

  for (unsigned int testCase = 0; testCase < 5; testCase++) {
    switch (testCase) {
      case 0:
        // Warm-up kernel
        streamConcurrency.setNumStreams(1);
        streamConcurrency.setNumKernels(1);
        break;

      case 1:
        // default stream executes serially
        streamConcurrency.setNumStreams(1);
        streamConcurrency.setNumKernels(1);
        break;

      case 2:
        // 2-way concurrency
        streamConcurrency.setNumStreams(2);
        streamConcurrency.setNumKernels(2);
        break;

      case 3:
        // 4-way concurrency
        streamConcurrency.setNumStreams(4);
        streamConcurrency.setNumKernels(4);
        break;

      case 4:
        streamConcurrency.setNumStreams(2);
        streamConcurrency.setNumKernels(4);
        break;

      case 5:
        break;

      default:
        break;
    }
    REQUIRE(true == streamConcurrency.run(testCase, deviceId));
  }
}

/**
 * End doxygen group perfComputeTest.
 * @}
 */
