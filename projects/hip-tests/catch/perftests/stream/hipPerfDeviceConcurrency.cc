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
 * @addtogroup hipPerfDeviceConcurrency hipPerfDeviceConcurrency
 * @{
 * @ingroup perfStreamTest
 * `hipError_t hipStreamCreate(hipStream_t* stream)` -
 * Create an asynchronous stream.
 */

#include <hip_test_common.hh>

typedef struct {
  double x;
  double y;
  double width;
} coordRec;

static coordRec coords[] = {
    {0.0, 0.0, 0.00001},  // All black
};

static unsigned int numCoords = sizeof(coords) / sizeof(coordRec);

__global__ void mandelbrot(uint* out, uint width, float xPos, float yPos, float xStep, float yStep,
                           uint maxIter) {
  int tid = (blockIdx.x * blockDim.x + threadIdx.x);
  int i = tid % width;
  int j = tid / width;
  float x0 = static_cast<float>(xPos + xStep * i);
  float y0 = static_cast<float>(yPos + yStep * j);

  float x = x0;
  float y = y0;

  uint iter = 0;
  float tmp;
  for (iter = 0; (x * x + y * y <= 4.0f) && (iter < maxIter); iter++) {
    tmp = x;
    x = fma(-y, y, fma(x, x, x0));
    y = fma(2.0f * tmp, y, y0);
  }
  out[tid] = iter;
};

class hipPerfDeviceConcurrency {
 public:
  hipPerfDeviceConcurrency();
  ~hipPerfDeviceConcurrency();

  void setNumGpus(unsigned int num) { numDevices = num; }
  unsigned int getNumGpus() { return numDevices; }

  void open(void);
  void close(void);
  bool run(unsigned int testCase, int numGpus);

 private:
  void setData(void* ptr, unsigned int value);
  void checkData(uint* ptr);

  unsigned int numDevices;
  unsigned int width_;
  unsigned int bufSize;
  unsigned int coordIdx;
  unsigned long long totalIters = 0;
};

hipPerfDeviceConcurrency::hipPerfDeviceConcurrency() {}

hipPerfDeviceConcurrency::~hipPerfDeviceConcurrency() {}

void hipPerfDeviceConcurrency::open(void) {
  int nGpu = 0;
  HIP_CHECK(hipGetDeviceCount(&nGpu));
  setNumGpus(nGpu);
  if (nGpu < 1) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 1");
  }
}

void hipPerfDeviceConcurrency::close() {}

bool hipPerfDeviceConcurrency::run(unsigned int testCase, int numGpus) {
  static int deviceId;
  uint** hPtr = new uint*[numGpus];
  uint** dPtr = new uint*[numGpus];
  hipStream_t* streams = new hipStream_t[numGpus];
  int* numCUs = new int[numGpus];
  unsigned int* maxIter = new unsigned int[numGpus];
  unsigned long long* expectedIters = new unsigned long long[numGpus];

  int threads, threads_per_block, blocks;
  float xStep, yStep, xPos, yPos;

  for (int i = 0; i < numGpus; i++) {
    if (testCase != 0) {
      deviceId = i;
    }

    HIP_CHECK(hipSetDevice(deviceId));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, i));
    if (testCase != 0) {
      CONSOLE_PRINT("info: running on bus 0x%x %s with %d CUs and device ID: %d", props.pciBusID,
                    props.name, props.multiProcessorCount, i);
    }
    numCUs[i] = props.multiProcessorCount;
    int clkFrequency = 0;
    HIP_CHECK(hipDeviceGetAttribute(&clkFrequency, hipDeviceAttributeClockRate, i));
    if (clkFrequency == 0) {
      CONSOLE_PRINT("clkFrequency = 0, set it to 1000000");
      clkFrequency = 1000000;
    }
    clkFrequency = (unsigned int)clkFrequency / 1000;

    // Maximum iteration count
    // maxIter = 8388608 * (engine_clock / 1000).serial execution
    maxIter[i] = (unsigned int)(((8388608 * ((float)clkFrequency / 1000)) * numCUs[i]) / 128);
    maxIter[i] = (maxIter[i] + 15) & ~15;

    // Width is divisible by 4 because the mandelbrot
    // kernel processes 4 pixels at once.
    width_ = 256;
    bufSize = width_ * width_ * sizeof(uint);
    // Create streams for concurrency
    HIP_CHECK(hipStreamCreate(&streams[i]));

    // Allocate memory on the host and device
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&hPtr[i]), bufSize, hipHostMallocDefault));
    setData(hPtr[i], 0xdeadbeef);
    HIP_CHECK(hipMalloc(reinterpret_cast<uint**>(&dPtr[i]), bufSize))

    // Prepare kernel launch parameters
    threads = (bufSize / sizeof(uint));
    threads_per_block = 64;
    blocks = (threads / threads_per_block) + (threads % threads_per_block);

    coordIdx = testCase % numCoords;
    xStep = static_cast<float>(coords[coordIdx].width / static_cast<double>(width_));
    yStep = static_cast<float>(-coords[coordIdx].width / static_cast<double>(width_));
    xPos = static_cast<float>(coords[coordIdx].x - 0.5 * coords[coordIdx].width);
    yPos = static_cast<float>(coords[coordIdx].y + 0.5 * coords[coordIdx].width);

    // Copy memory from host to device
    HIP_CHECK(hipMemcpy(dPtr[i], hPtr[i], bufSize, hipMemcpyHostToDevice));
  }

  // Time the kernel execution
  auto all_start = std::chrono::steady_clock::now();
  for (int i = 0; i < numGpus; i++) {
    if (testCase != 0) {
      deviceId = i;
    }

    HIP_CHECK(hipSetDevice(deviceId));
    hipLaunchKernelGGL(mandelbrot, dim3(blocks), dim3(threads_per_block), 0, streams[i], dPtr[i],
                       width_, xPos, yPos, xStep, yStep, maxIter[i]);
  }
  for (int i = 0; i < numGpus; i++) {
    HIP_CHECK(hipStreamSynchronize(0));
  }

  auto all_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> all_kernel_time = all_end - all_start;

  for (int i = 0; i < numGpus; i++) {
    if (testCase != 0) {
      deviceId = i;
    }
    HIP_CHECK(hipSetDevice(deviceId));

    // Copy data back from device to the host
    HIP_CHECK(hipMemcpy(hPtr[i], dPtr[i], bufSize, hipMemcpyDeviceToHost));
    checkData(hPtr[i]);
    expectedIters[i] = width_ * width_ * (unsigned long long)maxIter[i];
    if (testCase != 0) {
      checkData(hPtr[i]);
      if (totalIters != expectedIters[i]) {
        CONSOLE_PRINT("Incorrect iteration count detected");
      }
    }

    HIP_CHECK(hipStreamDestroy(streams[i]));
    // Free host and device memory
    HIP_CHECK(hipHostFree(hPtr[i]));
    HIP_CHECK(hipFree(dPtr[i]));
  }

  if (testCase != 0) {
    CONSOLE_PRINT("\nMeasured time for kernel computation on %d device(s): %.6f (s)\n", numGpus,
                  all_kernel_time.count());
  }

  if (testCase == 0) {
    deviceId++;
  }
  delete[] hPtr;
  delete[] dPtr;
  delete[] streams;
  delete[] numCUs;
  delete[] maxIter;
  delete[] expectedIters;
  return true;
}

void hipPerfDeviceConcurrency::setData(void* ptr, unsigned int value) {
  unsigned int* ptr2 = (unsigned int*)ptr;
  for (unsigned int i = 0; i < width_ * width_; i++) {
    ptr2[i] = value;
  }
}

void hipPerfDeviceConcurrency::checkData(uint* ptr) {
  totalIters = 0;
  for (unsigned int i = 0; i < width_ * width_; i++) {
    totalIters += ptr[i];
  }
}

/**
 * Test Description
 * ------------------------
 *  - Verify the different levels of device concurrency.
 * Test source
 * ------------------------
 *  - perftests/stream/hipPerfDeviceConcurrency.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Perf_hipPerfDeviceConcurrency") {
  hipPerfDeviceConcurrency deviceConcurrency;
  deviceConcurrency.open();
  int nGpu = deviceConcurrency.getNumGpus();

  // testCase = 0 refers to warmup kernel run
  int testCase = 0;
  for (int i = 0; i < nGpu; i++) {
    // Warm-up kernel on all devices
    REQUIRE(true == deviceConcurrency.run(testCase, 1));
  }

  // Time for kernel on 1 device
  REQUIRE(true == deviceConcurrency.run(++testCase, 1));

  // Time for kernel on all available devices
  REQUIRE(true == deviceConcurrency.run(++testCase, nGpu));
}

/**
 * End doxygen group perfStreamTest.
 * @}
 */
