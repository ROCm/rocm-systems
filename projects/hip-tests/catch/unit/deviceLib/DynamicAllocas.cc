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
// Testcase Description:
// Verifies working of following Dyanmic Allocations:
// 1. Dynamic Allocas in kernels,
// 2. Dynamic Allocas in device functions,
// 3. Dynamic Allocas with over-alignment,
// 4. Multiple Dynamic Allocas in a function,
// 5. Dynamic Allocas in functions with parameter passing

#include <hip_test_common.hh>
using namespace std;

////////////////////////////////////////////////////////////////////////////////
// Auto-Verification Code
////////////////////////////////////////////////////////////////////////////////

// Auto-verification API
template <typename T> bool verifyResult(T* output, T* expected) {
  int expected_length = sizeof(*expected) / sizeof(expected[0]);
  int actual_length = sizeof(*output) / sizeof(output[0]);
  if (expected_length != actual_length) return false;
  for (int i = 0; i < 10; ++i)
    if (expected[i] != output[i]) return false;
  return true;
}

// Kernels exercising dynamic allocations
extern "C" __global__ void test_kernel_uniform_dynamic_alloca(int* a, int* b, int N) {
  volatile int* tmp = (int*)__builtin_alloca(N * 4);
  for (int i = 0; i < N; i++) {
    tmp[i] = a[b[i]];
  }
  for (int i = 0; i < N; i++) {
    a[i] = tmp[i] + a[i];
  }
}

extern "C" __global__ void validate_kernel_uniform_dynamic_alloca(int* a, int* b) {
  volatile int* tmp = (int*)__builtin_alloca(10 * 4);
  for (int i = 0; i < 10; i++) {
    tmp[i] = a[b[i]];
  }
  for (int i = 0; i < 10; i++) {
    a[i] = tmp[i] + a[i];
  }
}

extern "C" __global__ void test_kernel_uniform_dynamic_alloca_over_aligned(int* a, int* b, int N) {
  alignas(1024) volatile int tmp[N];
  volatile int* tmp2 = (int*)__builtin_alloca(10 * 4);
  uintptr_t addr = reinterpret_cast<std::uintptr_t>(tmp);
  for (int i = 0; i < N; i++) {
    tmp[i] = a[b[i]];
    tmp2[i] = addr % 1024;
  }
  for (int i = 0; i < N; i++) {
    a[i] = tmp[i] + a[i];
    b[i] = tmp2[i];
  }
}

extern "C" __global__ void validate_kernel_uniform_dynamic_alloca_over_aligned(int* a, int* b) {
  alignas(1024) volatile int tmp[10];
  for (int i = 0; i < 10; i++) {
    tmp[i] = a[b[i]];
  }
  for (int i = 0; i < 10; i++) {
    a[i] = tmp[i] + a[i];
  }
}

extern "C" __global__ void test_kernel_divergent_dynamic_alloca(int* a, int* b) {
  int N = threadIdx.x + 10;
  volatile int* tmp = (int*)__builtin_alloca(N * 4);
  for (int i = 0; i < N; i++) {
    tmp[i] = a[b[i]];
  }
  for (int i = 0; i < N; i++) {
    a[i] = tmp[i] + a[i];
  }
}

extern "C" __global__ void validate_kernel_divergent_dynamic_alloca(int* a, int* b) {
  int N = threadIdx.x + 10;
  volatile int* tmp = (int*)__builtin_alloca(10 * 4);
  for (int i = 0; i < N; i++) {
    tmp[i] = a[b[i]];
  }
  for (int i = 0; i < N; i++) {
    a[i] = tmp[i] + a[i];
  }
}

extern "C" __global__ void test_kernel_divergent_dynamic_alloca_over_aligned(int* a, int* b) {
  int N = threadIdx.x + 10;
  alignas(512) volatile int tmp[N];
  uintptr_t addr = reinterpret_cast<std::uintptr_t>(tmp);
  for (int i = 0; i < N; i++) {
    tmp[i] = a[b[i]];
    b[i] = addr % 512;
  }
  for (int i = 0; i < N; i++) {
    a[i] = tmp[i] + a[i];
  }
}

extern "C" __global__ void validate_kernel_divergent_dynamic_alloca_over_aligned(int* a, int* b) {
  int N = threadIdx.x + 10;
  alignas(512) volatile int tmp[10];
  for (int i = 0; i < N; i++) {
    tmp[i] = a[b[i]];
  }
  for (int i = 0; i < N; i++) {
    a[i] = tmp[i] + a[i];
  }
}

extern "C" __global__ void test_kernel_multiple_dynamic_alloca(int* a, int* b, int* c, int N) {
  int X = threadIdx.x + 10;
  int Y = threadIdx.x + N;
  volatile int* tmp = (int*)__builtin_alloca(N * 4);
  volatile int* tmp2 = (int*)__builtin_alloca(X * 4);
  volatile int* tmp3 = (int*)__builtin_alloca(Y * 4);
  for (int i = 0; i < N; i++) {
    tmp[i] = a[b[i]];
  }
  for (int i = 0; i < N; i++) {
    a[i] = tmp[i] + a[i];
  }
  for (int i = 0; i < N; i++) {
    tmp2[i] = b[c[i]];
    tmp3[i] = c[a[i]];
  }
  for (int i = 0; i < N; i++) {
    b[i] = tmp2[i];
    c[i] = tmp3[i] + threadIdx.x;
  }
}

extern "C" __global__ void validate_kernel_multiple_dynamic_alloca(int* a, int* b, int* c) {
  volatile int* tmp = (int*)__builtin_alloca(10 * 4);
  volatile int* tmp2 = (int*)__builtin_alloca(10 * 4);
  volatile int* tmp3 = (int*)__builtin_alloca(10 * 4);
  for (int i = 0; i < 10; i++) {
    tmp[i] = a[b[i]];
  }
  for (int i = 0; i < 10; i++) {
    a[i] = tmp[i] + a[i];
  }
  for (int i = 0; i < 10; i++) {
    tmp2[i] = b[c[i]];
    tmp3[i] = c[a[i]];
  }
  for (int i = 0; i < 10; i++) {
    b[i] = tmp2[i];
    c[i] = tmp3[i] + threadIdx.x;
  }
}

extern "C" __global__ void test_kernel_multiple_dynamic_alloca_over_aligned(int* a, int* b, int* c,
                                                                            int* d, int N) {
  int X = threadIdx.x + 10;
  int Y = threadIdx.x + N;
  alignas(32) volatile int tmp[N];
  alignas(128) volatile int tmp2[X];
  alignas(512) volatile int tmp3[Y];
  uintptr_t addr = reinterpret_cast<std::uintptr_t>(tmp);
  uintptr_t addr2 = reinterpret_cast<std::uintptr_t>(tmp2);
  uintptr_t addr3 = reinterpret_cast<std::uintptr_t>(tmp3);
  for (int i = 0; i < N; i++) {
    tmp[i] = a[b[i]];
  }
  for (int i = 0; i < N; i++) {
    a[i] = tmp[i] + a[i];
  }
  for (int i = 0; i < N; i++) {
    tmp2[i] = b[c[i]];
    tmp3[i] = c[a[i]];
  }
  for (int i = 0; i < N; i++) {
    b[i] = tmp2[i];
    c[i] = tmp3[i] + threadIdx.x;
  }
  for (int i = 0; i < N; i++) {
    d[i] = (addr % 32) + (addr2 % 128) + (addr3 % 512);
  }
}

extern "C" __global__ void validate_kernel_multiple_dynamic_alloca_over_aligned(int* a, int* b,
                                                                                int* c) {
  alignas(32) volatile int tmp[10];
  alignas(128) volatile int tmp2[10];
  alignas(512) volatile int tmp3[10];
  for (int i = 0; i < 10; i++) {
    tmp[i] = a[b[i]];
  }
  for (int i = 0; i < 10; i++) {
    a[i] = tmp[i] + a[i];
  }
  for (int i = 0; i < 10; i++) {
    tmp2[i] = b[c[i]];
    tmp3[i] = c[a[i]];
  }
  for (int i = 0; i < 10; i++) {
    b[i] = tmp2[i];
    c[i] = tmp3[i] + threadIdx.x;
  }
}

static void runKernelDynamicAllocasTest() {
  size_t size = 20 * sizeof(int);
  std::unique_ptr<int> num_elements = std::make_unique<int>();
  // allocate mem for the result on host side
  size_t no_array_elements = 20;
  std::unique_ptr<int[]> h1 = std::make_unique<int[]>(no_array_elements);
  std::unique_ptr<int[]> h2 = std::make_unique<int[]>(no_array_elements);
  std::unique_ptr<int[]> h3 = std::make_unique<int[]>(no_array_elements);
  std::unique_ptr<int[]> r1 = std::make_unique<int[]>(no_array_elements);
  std::unique_ptr<int[]> r2 = std::make_unique<int[]>(no_array_elements);
  std::unique_ptr<int[]> r3 = std::make_unique<int[]>(no_array_elements);
  std::unique_ptr<int[]> r4 = std::make_unique<int[]>(no_array_elements);
  std::unique_ptr<int[]> r5 = std::make_unique<int[]>(no_array_elements);
  std::unique_ptr<int[]> r6 = std::make_unique<int[]>(no_array_elements);
  std::unique_ptr<int[]> r7 = std::make_unique<int[]>(no_array_elements);
  std::unique_ptr<int[]> align = std::make_unique<int[]>(no_array_elements);
  // initialize the memory
  for (int i = 0; i < 20; i++) {
    *(h1.get() + i) = i;
    *(h2.get() + i) = 19 - i;
    *(h3.get() + i) = 5;
    *(align.get() + i) = 0;
  }
  *num_elements = 10;
  // allocate device memory for result
  int* d1 = nullptr;
  int* d2 = nullptr;
  int* d3 = nullptr;
  int* d4 = nullptr;
  int* size_kernel;
  HIP_CHECK(hipMalloc(&d1, size));
  HIP_CHECK(hipMalloc(&d2, size));
  HIP_CHECK(hipMalloc(&d3, size));
  HIP_CHECK(hipMalloc(&d4, size));
  HIP_CHECK(hipMalloc(&size_kernel, sizeof(int)));
  size_t stackSize = 10;
  HIP_CHECK(hipDeviceSetLimit(hipLimitStackSize, stackSize));
  // kernel uniform dynamic alloca
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(size_kernel, num_elements.get(), sizeof(int), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_kernel_uniform_dynamic_alloca, dim3(1), dim3(1), 0, 0, d1, d2,
                     *size_kernel);
  HIP_CHECK(hipMemcpy(r1.get(), d1, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(validate_kernel_uniform_dynamic_alloca, dim3(1), dim3(1), 0, 0, d1, d2);
  HIP_CHECK(hipMemcpy(r3.get(), d1, size, hipMemcpyDeviceToHost));
  REQUIRE(verifyResult(r1.get(), r3.get()));
  // kernel uniform dynamic alloca over-aligned
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(size_kernel, num_elements.get(), sizeof(int), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_kernel_uniform_dynamic_alloca_over_aligned, dim3(1), dim3(1), 0, 0, d1,
                     d2, *size_kernel);
  HIP_CHECK(hipMemcpy(r1.get(), d1, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r2.get(), d2, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(validate_kernel_uniform_dynamic_alloca_over_aligned, dim3(1), dim3(1), 0, 0,
                     d1, d2);
  HIP_CHECK(hipMemcpy(r3.get(), d1, size, hipMemcpyDeviceToHost));
  REQUIRE(verifyResult(r1.get(), r3.get()));
  REQUIRE(verifyResult(r2.get(), align.get()));
  // kernel divergent dynamic alloca
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_kernel_divergent_dynamic_alloca, dim3(1), dim3(1), 0, 0, d1, d2);
  HIP_CHECK(hipMemcpy(r1.get(), d1, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(validate_kernel_divergent_dynamic_alloca, dim3(1), dim3(1), 0, 0, d1, d2);
  HIP_CHECK(hipMemcpy(r2.get(), d1, size, hipMemcpyDeviceToHost));
  REQUIRE(verifyResult(r1.get(), r2.get()));
  // kernel divergent dynamic alloca over aligned
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_kernel_divergent_dynamic_alloca_over_aligned, dim3(1), dim3(1), 0, 0, d1,
                     d2);
  HIP_CHECK(hipMemcpy(r1.get(), d1, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r2.get(), d2, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(validate_kernel_divergent_dynamic_alloca_over_aligned, dim3(1), dim3(1), 0, 0,
                     d1, d2);
  HIP_CHECK(hipMemcpy(r3.get(), d1, size, hipMemcpyDeviceToHost));
  REQUIRE(verifyResult(r1.get(), r3.get()));
  REQUIRE(verifyResult(r2.get(), align.get()));
  // kernel multiple dynamic alloca
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d3, h3.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(size_kernel, num_elements.get(), sizeof(int), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_kernel_multiple_dynamic_alloca, dim3(1), dim3(1), 0, 0, d1, d2, d3,
                     *size_kernel);
  HIP_CHECK(hipMemcpy(r1.get(), d1, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r2.get(), d2, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r3.get(), d3, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d3, h3.get(), size, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(validate_kernel_multiple_dynamic_alloca, dim3(1), dim3(1), 0, 0, d1, d2, d3);
  HIP_CHECK(hipMemcpy(r4.get(), d1, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r5.get(), d2, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r6.get(), d3, size, hipMemcpyDeviceToHost));
  REQUIRE(verifyResult(r1.get(), r4.get()));
  REQUIRE(verifyResult(r2.get(), r5.get()));
  REQUIRE(verifyResult(r3.get(), r6.get()));
  // kernel multiple dynamic alloca over aligned
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d3, h3.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(size_kernel, num_elements.get(), sizeof(int), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_kernel_multiple_dynamic_alloca_over_aligned, dim3(1), dim3(1), 0, 0, d1,
                     d2, d3, d4, *size_kernel);
  HIP_CHECK(hipMemcpy(r1.get(), d1, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r2.get(), d2, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r3.get(), d3, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r4.get(), d4, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d3, h3.get(), size, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(validate_kernel_multiple_dynamic_alloca_over_aligned, dim3(1), dim3(1), 0, 0,
                     d1, d2, d3);
  HIP_CHECK(hipMemcpy(r5.get(), d1, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r6.get(), d2, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r7.get(), d3, size, hipMemcpyDeviceToHost));
  REQUIRE(verifyResult(r1.get(), r5.get()));
  REQUIRE(verifyResult(r2.get(), r6.get()));
  REQUIRE(verifyResult(r3.get(), r7.get()));
  REQUIRE(verifyResult(r4.get(), align.get()));
  HIP_CHECK(hipFree(d1));
  HIP_CHECK(hipFree(d2));
  HIP_CHECK(hipFree(d3));
  HIP_CHECK(hipFree(size_kernel));
}

// kernels and device functions exercizing dynamic allocations
__noinline__ extern "C" __device__ void process_array_with_aligned_32_byte_buffer(int* a, int* b,
                                                                                  int* c, int* d,
                                                                                  int X) {
  alignas(32) volatile int tmp[X];
  uintptr_t addr = reinterpret_cast<std::uintptr_t>(tmp);
  for (int i = 0; i < (X + 0); i++) {
    tmp[i] = a[c[i]];
  }
  for (int i = 0; i < (X + 0); i++) {
    c[i] = tmp[i];
    d[i] += addr % 32;
  }
}

__noinline__ extern "C" __device__ void process_array_with_aligned_1024_byte_buffer(int* a, int* b,
                                                                                    int* c, int* d,
                                                                                    int X, int N) {
  alignas(1024) volatile int tmp[N];
  uintptr_t addr = reinterpret_cast<std::uintptr_t>(tmp);
  for (int i = 0; i < (N + 0); i++) {
    tmp[i] = a[b[i]];
  }
  for (int i = 0; i < (N + 0); i++) {
    b[i] = tmp[i];
    d[i] += addr % 1024;
  }
}

extern "C" __global__ void test_multiple_device_functions_nested(int* a, int* b, int* c, int* d,
                                                                 int N) {
  int X = threadIdx.x + 10;
  alignas(1024) volatile int tmp[X];
  uintptr_t addr = reinterpret_cast<std::uintptr_t>(tmp);
  process_array_with_aligned_1024_byte_buffer(a, b, c, d, X, N);
  for (int i = 0; i < X; i++) {
    tmp[i] = a[b[i]];
  }
  for (int i = 0; i < X; i++) {
    a[i] = tmp[i];
    d[i] += addr % 1024;
  }
  process_array_with_aligned_32_byte_buffer(a, b, c, d, X);
}

extern "C" __global__ void validate_multiple_device_functions_nested(int* a, int* b, int* c) {
  int X = threadIdx.x + 10;
  alignas(1024) volatile int tmp[10];
  alignas(1024) volatile int tmp2[10];
  for (int i = 0; i < 10; i++) {
    tmp2[i] = a[b[i]];
  }
  for (int i = 0; i < 10; i++) {
    b[i] = tmp2[i];
  }
  for (int i = 0; i < X; i++) {
    tmp[i] = a[b[i]];
  }
  for (int i = 0; i < X; i++) {
    a[i] = tmp[i];
  }
  alignas(32) volatile int tmp3[10];
  for (int i = 0; i < (X + 0); i++) {
    tmp3[i] = a[c[i]];
  }
  for (int i = 0; i < (X + 0); i++) {
    c[i] = tmp3[i];
  }
}

__noinline__ extern "C" __device__ void process_array_with_nested_1024_byte_alignment(
    int* a, int* b, int* c, int* d, int X, int N) {
  alignas(1024) volatile int tmp[X];
  uintptr_t addr = reinterpret_cast<std::uintptr_t>(tmp);
  process_array_with_aligned_32_byte_buffer(a, b, c, d, X);
  for (int i = 0; i < X; i++) {
    tmp[i] = a[b[i]];
  }
  for (int i = 0; i < X; i++) {
    b[i] = tmp[i];
    d[i] += addr % 1024;
  }
}

extern "C" __global__ void test_multiple_device_functions(int* a, int* b, int* c, int* d, int N) {
  int X = threadIdx.x + 10;
  alignas(1024) volatile int tmp[X];
  uintptr_t addr = reinterpret_cast<std::uintptr_t>(tmp);
  process_array_with_nested_1024_byte_alignment(a, b, c, d, X, N);
  for (int i = 0; i < X; i++) {
    tmp[i] = a[b[i]];
  }
  for (int i = 0; i < X; i++) {
    a[i] = tmp[i];
    d[i] += addr % 1024;
  }
}

extern "C" __global__ void validate_multiple_device_functions(int* a, int* b, int* c) {
  int X = threadIdx.x + 10;
  alignas(1024) volatile int tmp1[10];
  alignas(1024) volatile int tmp2[10];
  alignas(32) volatile int tmp3[10];
  for (int i = 0; i < (X + 0); i++) {
    tmp3[i] = a[c[i]];
  }
  for (int i = 0; i < (X + 0); i++) {
    c[i] = tmp3[i];
  }
  for (int i = 0; i < X; i++) {
    tmp2[i] = a[b[i]];
  }
  for (int i = 0; i < X; i++) {
    b[i] = tmp2[i];
  }
  for (int i = 0; i < X; i++) {
    tmp1[i] = a[b[i]];
  }
  for (int i = 0; i < X; i++) {
    a[i] = tmp1[i];
  }
}

__noinline__ extern "C" __device__ void process_array_with_unaligned_buffer(
    int* a, int* b, int* c, int X, int a0, int a1, int a2, int a3, int a4, int a5, int a6, int a7,
    int a8, int a9, int a10, int a11, int a12, int a13, int a14, int a15, int a16, int a17, int a18,
    int a19, int a20, int a21, int a22, int a23, int a24, int a25, int a26, int a27, int a28,
    int a29, int a30, int a31, int a32, int a33, int a34, int a35, int a36, int a37, int a38,
    int a39, int a40) {
  int* tmp = (int*)__builtin_alloca(X * 4);
  int sum = 0;
  sum += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9;
  sum += a10 + a11 + a12 + a13 + a14 + a15 + a16 + a17 + a18 + a19;
  sum += a20 + a21 + a22 + a23 + a24 + a25 + a26 + a27 + a28 + a29;
  sum += a30 + a31 + a32 + a33 + a34 + a35 + a36 + a37 + a38 + a39 + a40;
  for (int i = 0; i < X; i++) {
    tmp[i] = a[c[i]] + sum;
  }
  for (int i = 0; i < X; i++) {
    c[i] = tmp[i];
  }
}

__noinline__ extern "C" __device__ void process_array_with_aligned_64_byte_buffer(int* a, int* b,
                                                                                  int* c, int* d,
                                                                                  int X, int N) {
  alignas(64) volatile int tmp[X];
  uintptr_t addr = reinterpret_cast<std::uintptr_t>(tmp);
  process_array_with_unaligned_buffer(a, b, c, X, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                                      a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16],
                                      a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25],
                                      a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34],
                                      a[35], a[36], a[37], a[38], a[39], a[40]);
  for (int i = 0; i < X; i++) {
    tmp[i] = a[b[i]];
  }
  for (int i = 0; i < X; i++) {
    b[i] = tmp[i];
    d[i] += addr % 64;
  }
}

extern "C" __global__ void test_function_calls_parameter_passing(int* a, int* b, int* c, int* d,
                                                                 int N) {
  int X = threadIdx.x + 10;
  alignas(32) volatile int tmp[X];
  uintptr_t addr = reinterpret_cast<std::uintptr_t>(tmp);
  for (int i = 0; i < X; i++) {
    tmp[i] = a[b[i]];
  }
  process_array_with_aligned_64_byte_buffer(a, b, c, d, X, N);
  for (int i = 0; i < X; i++) {
    a[i] = tmp[i];
    d[i] += addr % 32;
  }
}

extern "C" __global__ void validate_function_calls_parameter_passing(int* a, int* b, int* c) {
  int X = threadIdx.x + 10;
  alignas(32) volatile int tmp[10];
  for (int i = 0; i < X; i++) {
    tmp[i] = a[b[i]];
  }
  alignas(64) volatile int tmp2[10];
  int* tmp3 = (int*)__builtin_alloca(10 * 4);
  int sum = 0;
  for (int i = 0; i <= 40; i++) sum += a[i];
  for (int i = 0; i < X; i++) {
    tmp3[i] = a[c[i]] + sum;
  }
  for (int i = 0; i < X; i++) {
    c[i] = tmp3[i];
  }
  for (int i = 0; i < X; i++) {
    tmp2[i] = a[b[i]];
  }
  for (int i = 0; i < X; i++) {
    b[i] = tmp2[i];
  }
  for (int i = 0; i < X; i++) {
    a[i] = tmp[i];
  }
}

__noinline__ extern "C" __device__ void process_array_with_aligned_128_byte_buffer(
    int* a, int* b, int* c, int* d, int X, int a0, int a1, int a2, int a3, int a4, int a5, int a6,
    int a7, int a8, int a9, int a10, int a11, int a12, int a13, int a14, int a15, int a16, int a17,
    int a18, int a19, int a20, int a21, int a22, int a23, int a24, int a25, int a26, int a27,
    int a28, int a29, int a30, int a31, int a32, int a33, int a34, int a35, int a36, int a37,
    int a38, int a39, int a40) {
  alignas(128) volatile int tmp[X];
  uintptr_t addr = reinterpret_cast<std::uintptr_t>(tmp);
  int sum = 0;
  sum += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9;
  sum += a10 + a11 + a12 + a13 + a14 + a15 + a16 + a17 + a18 + a19;
  sum += a20 + a21 + a22 + a23 + a24 + a25 + a26 + a27 + a28 + a29;
  sum += a30 + a31 + a32 + a33 + a34 + a35 + a36 + a37 + a38 + a39 + a40;
  for (int i = 0; i < X; i++) {
    tmp[i] = a[c[i]] + sum;
  }
  for (int i = 0; i < X; i++) {
    c[i] = tmp[i];
    d[i] += addr % 128;
  }
}

__noinline__ extern "C" __device__ void process_nested_dynamic_allocas(int* a, int* b, int* c,
                                                                       int* d, int X, int N) {
  alignas(64) volatile int tmp[X];
  uintptr_t addr = reinterpret_cast<std::uintptr_t>(tmp);
  process_array_with_aligned_128_byte_buffer(
      a, b, c, d, X, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11],
      a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24],
      a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37],
      a[38], a[39], a[40]);
  for (int i = 0; i < X; i++) {
    tmp[i] = a[b[i]];
  }
  for (int i = 0; i < X; i++) {
    b[i] = tmp[i];
    d[i] += addr % 32;
  }
}

extern "C" __global__ void test_function_calls_parameter_passing_over_aligned(int* a, int* b,
                                                                              int* c, int* d,
                                                                              int N) {
  int X = threadIdx.x + 10;
  alignas(32) volatile int tmp[X];
  uintptr_t addr = reinterpret_cast<std::uintptr_t>(tmp);
  for (int i = 0; i < X; i++) {
    tmp[i] = a[b[i]];
  }
  process_nested_dynamic_allocas(a, b, c, d, X, N);
  for (int i = 0; i < X; i++) {
    a[i] = tmp[i];
    d[i] += addr % 32;
  }
}

extern "C" __global__ void validate_function_calls_parameter_passing_over_aligned(int* a, int* b,
                                                                                  int* c) {
  int X = threadIdx.x + 10;
  alignas(32) volatile int tmp[10];
  for (int i = 0; i < X; i++) {
    tmp[i] = a[b[i]];
  }
  alignas(64) volatile int tmp2[10];
  alignas(128) volatile int tmp3[10];
  int sum = 0;
  for (int i = 0; i <= 40; i++) sum += a[i];
  for (int i = 0; i < X; i++) {
    tmp3[i] = a[c[i]] + sum;
  }
  for (int i = 0; i < X; i++) {
    c[i] = tmp3[i];
  }
  for (int i = 0; i < X; i++) {
    tmp2[i] = a[b[i]];
  }
  for (int i = 0; i < X; i++) {
    b[i] = tmp2[i];
  }
  for (int i = 0; i < X; i++) {
    a[i] = tmp[i];
  }
}

extern "C" __global__ void test_function_calls_parameter_passing_sandwiched(int* a, int* b, int* c,
                                                                            int* d, int N) {
  int X = threadIdx.x + 10;
  process_array_with_aligned_128_byte_buffer(
      a, b, c, d, X, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11],
      a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24],
      a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37],
      a[38], a[39], a[40]);
  alignas(32) volatile int tmp[X];
  uintptr_t addr = reinterpret_cast<std::uintptr_t>(tmp);
  for (int i = 0; i < X; i++) {
    tmp[i] = a[b[i]];
  }
  process_array_with_aligned_32_byte_buffer(a, b, c, d, X);
  for (int i = 0; i < X; i++) {
    a[i] = tmp[i];
    d[i] += addr % 32;
  }
}

extern "C" __global__ void validate_function_calls_parameter_passing_sandwiched(int* a, int* b,
                                                                                int* c) {
  int X = threadIdx.x + 10;
  alignas(128) volatile int tmp[10];
  int sum = 0;
  for (int i = 0; i <= 40; i++) sum += a[i];
  for (int i = 0; i < X; i++) {
    tmp[i] = a[c[i]] + sum;
  }
  for (int i = 0; i < X; i++) {
    c[i] = tmp[i];
  }
  alignas(32) volatile int tmp2[10];
  for (int i = 0; i < X; i++) {
    tmp2[i] = a[b[i]];
  }
  alignas(32) volatile int tmp3[10];
  for (int i = 0; i < (X + 0); i++) {
    tmp3[i] = a[c[i]];
  }
  for (int i = 0; i < (X + 0); i++) {
    c[i] = tmp3[i];
  }
  for (int i = 0; i < X; i++) {
    a[i] = tmp2[i];
  }
}

static void runDeviceDynamicAllocasTest() {
  size_t size = 50 * sizeof(int);
  std::unique_ptr<int> num_elements = std::make_unique<int>();
  // allocate mem for the result on host side
  size_t no_array_elements = 50;
  std::unique_ptr<int[]> h1 = std::make_unique<int[]>(no_array_elements);
  std::unique_ptr<int[]> h2 = std::make_unique<int[]>(no_array_elements);
  std::unique_ptr<int[]> h3 = std::make_unique<int[]>(no_array_elements);
  std::unique_ptr<int[]> r1 = std::make_unique<int[]>(no_array_elements);
  std::unique_ptr<int[]> r2 = std::make_unique<int[]>(no_array_elements);
  std::unique_ptr<int[]> r3 = std::make_unique<int[]>(no_array_elements);
  std::unique_ptr<int[]> r4 = std::make_unique<int[]>(no_array_elements);
  std::unique_ptr<int[]> r5 = std::make_unique<int[]>(no_array_elements);
  std::unique_ptr<int[]> r6 = std::make_unique<int[]>(no_array_elements);
  std::unique_ptr<int[]> r7 = std::make_unique<int[]>(no_array_elements);
  std::unique_ptr<int[]> align = std::make_unique<int[]>(no_array_elements);
  // initialize the memory
  for (int i = 0; i < 50; i++) {
    *(h1.get() + i) = i;
    *(h2.get() + i) = 49 - i;
    *(h3.get() + i) = 7;
    *(align.get() + i) = 0;
  }
  *num_elements = 10;
  // allocate device memory for result
  int* d1 = nullptr;
  int* d2 = nullptr;
  int* d3 = nullptr;
  int* d4 = nullptr;
  int* size_kernel;
  HIP_CHECK(hipMalloc(&d1, size));
  HIP_CHECK(hipMalloc(&d2, size));
  HIP_CHECK(hipMalloc(&d3, size));
  HIP_CHECK(hipMalloc(&d4, size));
  HIP_CHECK(hipMalloc(&size_kernel, sizeof(int)));
  size_t stackSize = 1;
  HIP_CHECK(hipDeviceSetLimit(hipLimitStackSize, stackSize));
  // multiple device functions nested
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d3, h3.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d4, align.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(size_kernel, num_elements.get(), sizeof(int), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_multiple_device_functions_nested, dim3(1), dim3(1), 0, 0, d1, d2, d3, d4,
                     *size_kernel);
  HIP_CHECK(hipMemcpy(r1.get(), d1, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r2.get(), d2, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r3.get(), d3, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r4.get(), d4, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d3, h3.get(), size, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(validate_multiple_device_functions_nested, dim3(1), dim3(1), 0, 0, d1, d2, d3);
  HIP_CHECK(hipMemcpy(r5.get(), d1, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r6.get(), d2, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r7.get(), d3, size, hipMemcpyDeviceToHost));
  REQUIRE(verifyResult(r1.get(), r5.get()));
  REQUIRE(verifyResult(r2.get(), r6.get()));
  REQUIRE(verifyResult(r3.get(), r7.get()));
  REQUIRE(verifyResult(r4.get(), align.get()));
  // multiple device functions
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d3, h3.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(size_kernel, num_elements.get(), sizeof(int), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_multiple_device_functions, dim3(1), dim3(1), 0, 0, d1, d2, d3, d4,
                     *size_kernel);
  HIP_CHECK(hipMemcpy(r1.get(), d1, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r2.get(), d2, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r3.get(), d3, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r4.get(), d4, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d3, h3.get(), size, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(validate_multiple_device_functions, dim3(1), dim3(1), 0, 0, d1, d2, d3);
  HIP_CHECK(hipMemcpy(r5.get(), d1, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r6.get(), d2, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r7.get(), d3, size, hipMemcpyDeviceToHost));
  REQUIRE(verifyResult(r1.get(), r5.get()));
  REQUIRE(verifyResult(r2.get(), r6.get()));
  REQUIRE(verifyResult(r3.get(), r7.get()));
  REQUIRE(verifyResult(r4.get(), align.get()));
  // function calls with parameter passing
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d3, h3.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(size_kernel, num_elements.get(), sizeof(int), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_function_calls_parameter_passing, dim3(1), dim3(1), 0, 0, d1, d2, d3, d4,
                     *size_kernel);
  HIP_CHECK(hipMemcpy(r1.get(), d1, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r2.get(), d2, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r3.get(), d3, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r4.get(), d4, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d3, h3.get(), size, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(validate_function_calls_parameter_passing, dim3(1), dim3(1), 0, 0, d1, d2, d3);
  HIP_CHECK(hipMemcpy(r5.get(), d1, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r6.get(), d2, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r7.get(), d3, size, hipMemcpyDeviceToHost));
  REQUIRE(verifyResult(r1.get(), r5.get()));
  REQUIRE(verifyResult(r2.get(), r6.get()));
  REQUIRE(verifyResult(r3.get(), r7.get()));
  REQUIRE(verifyResult(r4.get(), align.get()));
  // function calls with parameter passing over aligned
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d3, h3.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(size_kernel, num_elements.get(), sizeof(int), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_function_calls_parameter_passing_over_aligned, dim3(1), dim3(1), 0, 0, d1,
                     d2, d3, d4, *size_kernel);
  HIP_CHECK(hipMemcpy(r1.get(), d1, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r2.get(), d2, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r3.get(), d3, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r4.get(), d4, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d3, h3.get(), size, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(validate_function_calls_parameter_passing_over_aligned, dim3(1), dim3(1), 0, 0,
                     d1, d2, d3);
  HIP_CHECK(hipMemcpy(r5.get(), d1, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r6.get(), d2, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r7.get(), d3, size, hipMemcpyDeviceToHost));
  REQUIRE(verifyResult(r1.get(), r5.get()));
  REQUIRE(verifyResult(r2.get(), r6.get()));
  REQUIRE(verifyResult(r3.get(), r7.get()));
  REQUIRE(verifyResult(r4.get(), align.get()));
  // function calls with parameter passing sandwiched
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d3, h3.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(size_kernel, num_elements.get(), sizeof(int), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(test_function_calls_parameter_passing_sandwiched, dim3(1), dim3(1), 0, 0, d1,
                     d2, d3, d4, *size_kernel);
  HIP_CHECK(hipMemcpy(r1.get(), d1, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r2.get(), d2, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r3.get(), d3, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r4.get(), d4, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(d1, h1.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d2, h2.get(), size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d3, h3.get(), size, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(validate_function_calls_parameter_passing_sandwiched, dim3(1), dim3(1), 0, 0,
                     d1, d2, d3);
  HIP_CHECK(hipMemcpy(r5.get(), d1, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r6.get(), d2, size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(r7.get(), d3, size, hipMemcpyDeviceToHost));
  REQUIRE(verifyResult(r1.get(), r5.get()));
  REQUIRE(verifyResult(r2.get(), r6.get()));
  REQUIRE(verifyResult(r3.get(), r7.get()));
  REQUIRE(verifyResult(r4.get(), align.get()));
  HIP_CHECK(hipFree(d1));
  HIP_CHECK(hipFree(d2));
  HIP_CHECK(hipFree(d3));
  HIP_CHECK(hipFree(size_kernel));
}

/*
These testcases perform the following scenarios of dynamic allocas
  // 1. dyn alloca in kernel - uniform value
  // 2. dyn alloca in kernel - uniform value - over aligned
  // 3. dyn alloca in kernel - divergent value
  // 4. dyn alloca in kernel - divergent value - over aligned
  // 5. multiple dyn alloca in kernel
  // 6. multiple dyn alloca in kernel - over aligned
  // 7. multiple device function calls
  // 8. multiple device function calls nested
  // 9. function calls with parameter passing
  // 10. function calls with parameter passing - over aligned
  // 11. function calls with parameter passing - sandwiched
*/
TEST_CASE("Dynamic_Alloca") {
  SECTION("Kernel dynamic allocas") { runKernelDynamicAllocasTest(); }
  SECTION("Function calls with dynamic allocas") { runDeviceDynamicAllocasTest(); }
}
