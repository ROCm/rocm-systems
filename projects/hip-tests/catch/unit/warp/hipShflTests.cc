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

#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip/hip_fp16.h>

#define WIDTH 4

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

// Device (Kernel) function, it must be void
template <typename T> __global__ void matrixTranspose(T* out, T* in, const int width) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  T val = in[x];
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) out[i * width + j] = __shfl(val, j * width + i);
  }
}

// CPU implementation of matrix transpose
template <typename T>
void matrixTransposeCPUReference(T* output, T* input, const unsigned int width) {
  for (unsigned int j = 0; j < width; j++) {
    for (unsigned int i = 0; i < width; i++) {
      output[i * width + j] = input[j * width + i];
    }
  }
}

static void getFactor(int* fact) { *fact = 101; }
static void getFactor(unsigned int* fact) { *fact = static_cast<unsigned int>(INT32_MAX) + 1; }
static void getFactor(float* fact) { *fact = 2.5; }
static void getFactor(__half* fact) { *fact = 2.5; }
static void getFactor(double* fact) { *fact = 2.5; }
static void getFactor(int64_t* fact) { *fact = 303; }
static void getFactor(uint64_t* fact) { *fact = static_cast<uint64_t>(__LONG_LONG_MAX__) + 1; }

template <typename T> int compare(T* TransposeMatrix, T* cpuTransposeMatrix) {
  int errors = 0;
  for (int i = 0; i < NUM; i++) {
    if (TransposeMatrix[i] != cpuTransposeMatrix[i]) {
      errors++;
    }
  }
  return errors;
}

template <> int compare<__half>(__half* TransposeMatrix, __half* cpuTransposeMatrix) {
  int errors = 0;
  for (int i = 0; i < NUM; i++) {
    if (__half2float(TransposeMatrix[i]) != __half2float(cpuTransposeMatrix[i])) {  // NOLINT
      errors++;
    }
  }
  return errors;
}

template <typename T> void init(T* Matrix) {
  // initialize the input data
  T factor;
  getFactor(&factor);
  for (int i = 0; i < NUM; i++) {
    Matrix[i] = (T)i + factor;
  }
}

template <> void init(__half* Matrix) {
  // initialize the input data
  __half factor;
  getFactor(&factor);
  for (int i = 0; i < NUM; i++) {
    Matrix[i] = i + __half2float(factor);
  }
}

template <typename T> static void runTest() {
  T* Matrix;
  T* TransposeMatrix;
  T* cpuTransposeMatrix;

  T* gpuMatrix;
  T* gpuTransposeMatrix;

  hipDeviceProp_t devProp;
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  int errors = 0;

  Matrix = reinterpret_cast<T*>(malloc(NUM * sizeof(T)));
  TransposeMatrix = reinterpret_cast<T*>(malloc(NUM * sizeof(T)));
  cpuTransposeMatrix = reinterpret_cast<T*>(malloc(NUM * sizeof(T)));

  init(Matrix);

  // allocate the memory on the device side
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&gpuMatrix), NUM * sizeof(T)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&gpuTransposeMatrix), NUM * sizeof(T)));

  // Memory transfer from host to device
  HIP_CHECK(hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(T), hipMemcpyHostToDevice));

  // Lauching kernel from host
  hipLaunchKernelGGL(matrixTranspose<T>, dim3(1), dim3(THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y),
                     0, 0, gpuTransposeMatrix, gpuMatrix, WIDTH);

  // Memory transfer from device to host
  HIP_CHECK(hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(T), hipMemcpyDeviceToHost));

  // CPU MatrixTranspose computation
  matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, WIDTH);

  // verify the results
  REQUIRE(errors == compare(TransposeMatrix, cpuTransposeMatrix));
  // free the resources on device side
  HIP_CHECK(hipFree(gpuMatrix));
  HIP_CHECK(hipFree(gpuTransposeMatrix));

  // free the resources on host side
  free(Matrix);
  free(TransposeMatrix);
  free(cpuTransposeMatrix);
}

/**
 * @addtogroup __shfl __shfl
 * @{
 * @ingroup ShflTest
 * `T  __shfl(T var, int srcLane, int width=warpSize)` -
 * Contains wrap __shfl functions.
 * @}
 */

/**
 * Test Description
 * ------------------------
 * - Test case to verify __shfl warp functions for different datatypes.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipShflTests.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipShflTests") {
  SECTION("run test for int") { runTest<int>(); }
  SECTION("run test for float") { runTest<float>(); }
  SECTION("run test for double") { runTest<double>(); }
  // Test added to support half datatype.
  SECTION("run test for __half") { runTest<__half>(); }
  SECTION("run test for int64_t") { runTest<int64_t>(); }
  SECTION("run test for unsigned int") { runTest<unsigned int>(); }
  SECTION("run test for uint64_t") { runTest<uint64_t>(); }
}


/**
 * Test Description
 * ------------------------
 * - Test case to verify __shfl broadcast vals with undef args.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipShflTests.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */

__global__ void testShflWithUndefArgs(unsigned long total_out_strings, unsigned long* str_counter,
                                      uint32_t* out) {
  constexpr auto BLOCK_SIZE = warpSize;
  unsigned long lane = threadIdx.x % BLOCK_SIZE;

  auto get_next_string = [&]() {
    unsigned long istring;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
    if (lane == 0) {
      istring = atomicAdd(str_counter, 1);
    }
#pragma clang diagnostic pop
    return __shfl(istring, 0, warpSize);
  };

  for (unsigned long istring = get_next_string(); istring < total_out_strings;
       istring = get_next_string()) {
    out[lane]++;
  }
}
TEST_CASE("Unit_hipShflUndefArgs") {
  hipDeviceProp_t prop;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  size_t count = prop.warpSize;
  unsigned long* str_ctr;
  uint32_t* str_ctr1;
  HIP_CHECK(hipMalloc(&str_ctr, sizeof(unsigned long)));
  HIP_CHECK(hipMemset(str_ctr, 0, sizeof(unsigned long)));
  HIP_CHECK(hipMalloc(&str_ctr1, sizeof(uint32_t) * count));
  HIP_CHECK(hipMemset(str_ctr1, 0, sizeof(uint32_t) * count));
  testShflWithUndefArgs<<<1, count>>>(12, str_ctr, str_ctr1);
  uint32_t* out = new uint32_t[count];
  for (int i = 0; i < count; i++) {
    out[i] = 0;
  }
  HIP_CHECK(hipMemcpy(out, str_ctr1, count * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  for (int i = 0; i < count; i++) {
    REQUIRE(out[i] == 12);
  }
  HIP_CHECK(hipFree(str_ctr));
  HIP_CHECK(hipFree(str_ctr1));
  delete[] out;
}

/**
 * End doxygen group ShflTest.
 * @}
 */
