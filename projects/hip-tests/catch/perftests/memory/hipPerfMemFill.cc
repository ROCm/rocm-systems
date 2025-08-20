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

#define SIMPLY_ASSIGN 0
#define USE_HIPTEST_SETNUMBLOCKS 0

template <class T> __global__ void vec_fill(T* x, T coef, int N) {
  const int istart = threadIdx.x + blockIdx.x * blockDim.x;
  const int ishift = blockDim.x * gridDim.x;
  for (int i = istart; i < N; i += ishift) {
#if SIMPLY_ASSIGN
    x[i] = coef;
#else
    x[i] = coef * i;
#endif
  }
}

__device__ void print_log(int i, double value, double expected) {
  printf("failed at %d: val=%g, expected=%g\n", i, value, expected);
}

__device__ void print_log(int i, int value, int expected) {
  printf("failed at %d: val=%d, expected=%d\n", i, value, expected);
}

template <class T> __global__ void vec_verify(T* x, T coef, int N) {
  const int istart = threadIdx.x + blockIdx.x * blockDim.x;
  const int ishift = blockDim.x * gridDim.x;
  for (int i = istart; i < N; i += ishift) {
#if SIMPLY_ASSIGN
    if (x[i] != coef) {
      print_log(i, x[i], coef);
    }
#else
    if (x[i] != coef * i) {
      print_log(i, x[i], coef * i);
    }
#endif
  }
}

template <class T>
__global__ void daxpy(T* __restrict__ x, T* __restrict__ y, const T coef, int Niter, int N) {
  const int istart = threadIdx.x + blockIdx.x * blockDim.x;
  const int ishift = blockDim.x * gridDim.x;
  for (int iter = 0; iter < Niter; ++iter) {
    T iv = coef * iter;
    for (int i = istart; i < N; i += ishift) y[i] = iv * x[i] + y[i];
  }
}

template <class T> class hipPerfMemFill {
 private:
  static constexpr int NUM_START = 27;
  static constexpr int NUM_SIZE = 4;
  static constexpr int NUM_ITER = 10;
  static constexpr double NUM_1GB = 1024.0 * 1024.0 * 1024.0;
  size_t totalSizes_[NUM_SIZE];
  hipDeviceProp_t props_;
  const T coef_ = getCoefficient(3.14159);
  const unsigned int threadsPerBlock_ = 64;
  unsigned int blocksPerCU_;

 public:
  hipPerfMemFill() {
    for (int i = 0; i < NUM_SIZE; i++) {
      // 128M, 256M, 512M, 1024M
      totalSizes_[i] = 1ull << (i + NUM_START);
    }
  }

  ~hipPerfMemFill() {}

  bool supportLargeBar() { return props_.isLargeBar != 0; }

  bool supportManagedMemory() { return props_.managedMemory != 0; }

  const T getCoefficient(double val) { return static_cast<T>(val); }

  void setHostBuffer(T* A, T val, size_t size) {
    size_t len = size / sizeof(T);
    for (int i = 0; i < len; i++) {
      A[i] = val;
    }
  }

  bool open(int deviceId) {
    int nGpu = 0;
    HIP_CHECK(hipGetDeviceCount(&nGpu));
    if (nGpu < 1) {
      printf("No GPU!");
      return false;
    } else if (deviceId >= nGpu) {
      printf("Info: wrong GPU Id %d\n", deviceId);
      return false;
    }

    HIP_CHECK(hipSetDevice(deviceId));
    memset(&props_, 0, sizeof(props_));
    HIP_CHECK(hipGetDeviceProperties(&props_, deviceId));
    blocksPerCU_ = props_.multiProcessorCount * 4;

    std::cout << "Info: running on device: id: " << deviceId << ", bus: 0x" << props_.pciBusID
              << " " << props_.name << " with " << props_.multiProcessorCount
              << " CUs, large bar: " << supportLargeBar()
              << ", managed memory: " << supportManagedMemory()
              << ", DeviceMallocFinegrained: " << supportDeviceMallocFinegrained() << std::endl;
    return true;
  }

  void log_host(const char* title, double GBytes, double sec) {
    std::cout << title << " [" << std::setw(7) << GBytes << " GB]: cost " << std::setw(10) << sec
              << " s in bandwidth " << std::setw(10) << GBytes / sec << " [GB/s]" << std::endl;
  }

  void log_kernel(const char* title, double GBytes, double sec, double sec_hv, double sec_kv) {
    std::cout << title << " [" << std::setw(7) << GBytes << " GB]: cost " << std::setw(10) << sec
              << " s in bandwidth " << std::setw(10) << GBytes / sec << " [GB/s]"
              << ", hostVerify cost " << std::setw(10) << sec_hv << " s in bandwidth "
              << std::setw(10) << GBytes / sec_hv << " [GB/s]" << ", kernelVerify cost "
              << std::setw(10) << sec_kv << " s in bandwidth " << std::setw(10) << GBytes / sec_kv
              << " [GB/s]" << std::endl;
  }

  void hostFill(size_t size, T* data, T coef, double* sec) {
    size_t num = size / sizeof(T);  // Size of elements
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < num; ++i) {
#if SIMPLY_ASSIGN
      data[i] = coef;
#else
      data[i] = coef * i;
#endif
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;  // in second
    *sec = diff.count();
  }

  void kernelFill(size_t size, T* data, T coef, double* sec) {
    size_t num = size / sizeof(T);  // Size of elements
    unsigned blocks = setNumBlocks(num);

    // kernel will be loaded first time
    hipLaunchKernelGGL(HIP_KERNEL_NAME(vec_fill<T>), dim3(blocks), dim3(threadsPerBlock_), 0, 0,
                       data, 0, num);
    HIP_CHECK(hipDeviceSynchronize());

    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < NUM_ITER; ++iter) {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(vec_fill<T>), dim3(blocks), dim3(threadsPerBlock_), 0, 0,
                         data, coef, num);
    }
    HIP_CHECK(hipDeviceSynchronize());

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;  // in second
    *sec = diff.count() / NUM_ITER;                    // in second
  }

  void hostVerify(size_t size, T* data, T coef, double* sec) {
    size_t num = size / sizeof(T);  // Size of elements
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < num; ++i) {
#if SIMPLY_ASSIGN
      if (data[i] != coef) {
        std::cout << "hostVerify failed: i=" << i << ", data[i]=" << data[i]
                  << ", expected=" << coef << std::endl;
        REQUIRE(false);
      }
#else
      if (data[i] != coef * i) {
        std::cout << "hostVerify failed: i=" << i << ", data[i]=" << data[i]
                  << ", expected=" << coef * i << std::endl;
        REQUIRE(false);
      }
#endif
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;  // in second
    *sec = diff.count();
  }

  void kernelVerify(size_t size, T* data, T coef, double* sec) {
    size_t num = size / sizeof(T);  // Size of elements
    unsigned blocks = setNumBlocks(num);

    // kernel will be loaded first time
    hipLaunchKernelGGL(HIP_KERNEL_NAME(vec_verify<T>), dim3(blocks), dim3(threadsPerBlock_), 0, 0,
                       data, coef, num);
    HIP_CHECK(hipDeviceSynchronize());

    // Now all data verified. The following is to test bandwidth.
    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < NUM_ITER; ++iter) {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(vec_verify<T>), dim3(blocks), dim3(threadsPerBlock_), 0, 0,
                         data, coef, num);
    }
    HIP_CHECK(hipDeviceSynchronize());

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;  // in second
    *sec = diff.count() / NUM_ITER;                    // in second
  }

  bool testLargeBarDeviceMemoryHostFill(size_t size) {
    if (!supportLargeBar()) {
      return false;
    }

    double GBytes = static_cast<double>(size) / NUM_1GB;

    T* A;
    HIP_CHECK(hipMalloc(&A, size));
    double sec = 0;
    hostFill(size, A, coef_, &sec);  // Cpu can access device mem in LB
    HIP_CHECK(hipFree(A));

    log_host("Largebar: host   fill", GBytes, sec);
    return true;
  }

  bool testLargeBar() {
    if (!supportLargeBar()) {
      return false;
    }

    std::cout << "Test large bar device memory host filling" << std::endl;
    for (int i = 0; i < NUM_SIZE; i++) {
      if (!testLargeBarDeviceMemoryHostFill(totalSizes_[i])) {
        return false;
      }
    }

    return true;
  }

  bool testManagedMemoryHostFill(size_t size) {
    if (!supportManagedMemory()) {
      return false;
    }
    double GBytes = static_cast<double>(size) / NUM_1GB;

    T* A;
    HIP_CHECK(hipMallocManaged(&A, size));
    double sec = 0;
    hostFill(size, A, coef_, &sec);  // Cpu can access HMM mem
    HIP_CHECK(hipFree(A));

    log_host("Managed: host   fill", GBytes, sec);
    return true;
  }

  bool testManagedMemoryKernelFill(size_t size) {
    if (!supportManagedMemory()) {
      return false;
    }
    double GBytes = static_cast<double>(size) / NUM_1GB;

    T* A;
    HIP_CHECK(hipMallocManaged(&A, size));

    double sec = 0, sec_hv = 0, sec_kv = 0;
    kernelFill(size, A, coef_, &sec);
    // Managed memory can be verified by host
    hostVerify(size, A, coef_, &sec_hv);
    kernelVerify(size, A, coef_, &sec_kv);
    HIP_CHECK(hipFree(A));

    log_kernel("Managed: kernel fill", GBytes, sec, sec_hv, sec_kv);

    return true;
  }

  bool testManagedMemory() {
    if (!supportManagedMemory()) {
      return false;
    }

    std::cout << "Test managed memory host filling" << std::endl;
    for (int i = 0; i < NUM_SIZE; i++) {
      if (!testManagedMemoryHostFill(totalSizes_[i])) {
        return false;
      }
    }

    std::cout << "Test managed memory kernel filling" << std::endl;
    for (int i = 0; i < NUM_SIZE; i++) {
      if (!testManagedMemoryKernelFill(totalSizes_[i])) {
        return false;
      }
    }

    return true;
  }

  bool testHostMemoryHostFill(size_t size, unsigned int flags) {
    double GBytes = static_cast<double>(size) / NUM_1GB;
    T* A;
    HIP_CHECK(hipHostMalloc(&A, size, flags));
    double sec = 0;
    hostFill(size, A, coef_, &sec);
    HIP_CHECK(hipHostFree(A));

    log_host("Host: host   fill", GBytes, sec);
    return true;
  }

  bool testHostMemoryKernelFill(size_t size, unsigned int flags) {
    double GBytes = static_cast<double>(size) / NUM_1GB;

    T* A;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A), size, flags));
    double sec = 0, sec_hv = 0, sec_kv = 0;
    kernelFill(size, A, coef_, &sec);
    hostVerify(size, A, coef_, &sec_hv);
    kernelVerify(size, A, coef_, &sec_kv);
    HIP_CHECK(hipHostFree(A));

    log_kernel("Host: kernel fill", GBytes, sec, sec_hv, sec_kv);
    return true;
  }

  bool testHostMemory() {
    std::cout << "Test coherent host memory host filling" << std::endl;
    for (int i = 0; i < NUM_SIZE; i++) {
      if (!testHostMemoryHostFill(totalSizes_[i], hipHostMallocCoherent)) {
        return false;
      }
    }

    std::cout << "Test non-coherent host memory host filling" << std::endl;
    for (int i = 0; i < NUM_SIZE; i++) {
      if (!testHostMemoryHostFill(totalSizes_[i], hipHostMallocNonCoherent)) {
        return false;
      }
    }

    std::cout << "Test coherent host memory kernel filling" << std::endl;
    for (int i = 0; i < NUM_SIZE; i++) {
      if (!testHostMemoryKernelFill(totalSizes_[i], hipHostMallocCoherent)) {
        return false;
      }
    }

    std::cout << "Test non-coherent host memory kernel filling" << std::endl;
    for (int i = 0; i < NUM_SIZE; i++) {
      if (!testHostMemoryKernelFill(totalSizes_[i], hipHostMallocNonCoherent)) {
        return false;
      }
    }

    return true;
  }

  /* This function should be via device attribute query*/
  bool supportDeviceMallocFinegrained() {
#ifdef __HIP_PLATFORM_AMD__
    T* A = nullptr;
    hipError_t err;

    err =
        hipExtMallocWithFlags(reinterpret_cast<void**>(&A), sizeof(T), hipDeviceMallocFinegrained);
    if (err || !A) {
      return false;
    }
    HIP_CHECK(hipFree(A));
    return true;
#else
    return false;
#endif
  }

  unsigned int setNumBlocks(size_t size) {
    size_t num = size / sizeof(T);

#if USE_HIPTEST_SETNUMBLOCKS
    return HipTest::setNumBlocks(blocksPerCU_, threadsPerBlock_, num);
#else
    return (num + threadsPerBlock_ - 1) / threadsPerBlock_;
#endif
  }

#ifdef __HIP_PLATFORM_AMD__
  bool testExtDeviceMemoryHostFill(size_t size, unsigned int flags) {
    double GBytes = static_cast<double>(size) / NUM_1GB;

    T* A = nullptr;
    HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&A), size, flags));
    if (!A) {
      std::cout << "failed hipExtMallocWithFlags() with size =" << size << " flags=" << std::hex
                << flags << std::endl;
      return false;
    }

    double sec = 0;
    hostFill(size, A, coef_, &sec);  // Cpu can access this mem
    HIP_CHECK(hipFree(A));

    log_host("ExtDevice: host   fill", GBytes, sec);
    return true;
  }

  bool testExtDeviceMemoryKernelFill(size_t size, unsigned int flags) {
    double GBytes = static_cast<double>(size) / NUM_1GB;

    T* A = nullptr;
    HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&A), size, flags));
    if (!A) {
      std::cout << "failed hipExtMallocWithFlags() with size =" << size << " flags=" << std::hex
                << flags << std::endl;
      return false;
    }

    double sec = 0, sec_hv = 0, sec_kv = 0;
    kernelFill(size, A, coef_, &sec);
    // Fine grained device memory can be verified by host
    hostVerify(size, A, coef_, &sec_hv);
    kernelVerify(size, A, coef_, &sec_kv);
    HIP_CHECK(hipFree(A));

    log_kernel("ExtDevice: kernel fill", GBytes, sec, sec_hv, sec_kv);

    return true;
  }

  bool testExtDeviceMemory() {
    std::cout << "Test fine grained device memory host filling" << std::endl;
    for (int i = 0; i < NUM_SIZE; i++) {
      if (!testExtDeviceMemoryHostFill(totalSizes_[i], hipDeviceMallocFinegrained)) {
        return false;
      }
    }

    std::cout << "Test fine grained device memory kernel filling" << std::endl;
    for (int i = 0; i < NUM_SIZE; i++) {
      if (!testExtDeviceMemoryKernelFill(totalSizes_[i], hipDeviceMallocFinegrained)) {
        return false;
      }
    }

    return true;
  }
#endif

  bool run() {
    if (supportLargeBar()) {
      if (!testLargeBar()) {
        return false;
      }
    }

    if (supportManagedMemory()) {
      if (!testManagedMemory()) {
        return false;
      }
    }

    if (!testHostMemory()) {
      return false;
    }

#ifdef __HIP_PLATFORM_AMD__
    if (supportDeviceMallocFinegrained()) {
      if (!testExtDeviceMemory()) {
        return false;
      }
    }
#endif
    return true;
  }
};

/**
 * Test Description
 * ------------------------
 *  - Verify hipPerfMemFill status.
 * Test source
 * ------------------------
 *  - perftests/memory/hipPerfMemFill.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Perf_hipPerfMemFill_test") {
  std::cout << "Test int" << std::endl;
  hipPerfMemFill<int> hipPerfMemFillInt;
  REQUIRE(true == hipPerfMemFillInt.open(0));
  REQUIRE(true == hipPerfMemFillInt.run());

  std::cout << "Test double" << std::endl;
  hipPerfMemFill<double> hipPerfMemFillDouble;
  REQUIRE(true == hipPerfMemFillDouble.open(0));
  REQUIRE(true == hipPerfMemFillDouble.run());
}

/**
 * End doxygen group perfMemoryTest.
 * @}
 */
