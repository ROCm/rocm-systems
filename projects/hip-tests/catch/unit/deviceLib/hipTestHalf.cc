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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip/hip_fp16.h>
#include <hip_test_common.hh>

#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"

__device__ void test_convert() {
  __half x;
  float y = static_cast<float>(x);
}

__global__ void __halfMath(bool* result, __half a) {
  result[0] = __heq(__hadd(a, __half{1}), __half{2});
  result[0] = __heq(__hadd_sat(a, __half{1}), __half{1}) && result[0];
  result[0] = __heq(__hfma(a, __half{2}, __half{3}), __half{5}) && result[0];
  result[0] = __heq(__hfma_sat(a, __half{2}, __half{3}), __half{1}) && result[0];
  result[0] = __heq(__hsub(a, __half{1}), __half{0}) && result[0];
  result[0] = __heq(__hsub_sat(a, __half{2}), __half{0}) && result[0];
  result[0] = __heq(__hmul(a, __half{2}), __half{2}) && result[0];
  result[0] = __heq(__hmul_sat(a, __half{2}), __half{1}) && result[0];
  result[0] = __heq(__hdiv(a, __half{2}), __half{0.5}) && result[0];
}

__device__ bool to_bool(const __half2& x) {
  auto r = static_cast<const __half2_raw&>(x);

  return r.data.x != 0 && r.data.y != 0;
}

__global__ void __half2Math(bool* result, __half2 a) {
  result[0] = to_bool(__heq2(__hadd2(a, __half2{1, 1}), __half2{2, 2}));
  result[0] = to_bool(__heq2(__hadd2_sat(a, __half2{1, 1}), __half2{1, 1})) && result[0];
  result[0] = to_bool(__heq2(__hfma2(a, __half2{2, 2}, __half2{3, 3}), __half2{5, 5})) && result[0];
  result[0] =
      to_bool(__heq2(__hfma2_sat(a, __half2{2, 2}, __half2{3, 3}), __half2{1, 1})) && result[0];
  result[0] = to_bool(__heq2(__hsub2(a, __half2{1, 1}), __half2{0, 0})) && result[0];
  result[0] = to_bool(__heq2(__hsub2_sat(a, __half2{2, 2}), __half2{0, 0})) && result[0];
  result[0] = to_bool(__heq2(__hmul2(a, __half2{2, 2}), __half2{2, 2})) && result[0];
  result[0] = to_bool(__heq2(__hmul2_sat(a, __half2{2, 2}), __half2{1, 1})) && result[0];
  result[0] = to_bool(__heq2(__h2div(a, __half2{2, 2}), __half2{0.5, 0.5})) && result[0];
}

__global__ void kernel_hisnan(__half* input, int* output) {
  int tx = threadIdx.x;
  output[tx] = __hisnan(input[tx]);
}

__global__ void kernel_hisinf(__half* input, int* output) {
  int tx = threadIdx.x;
  output[tx] = __hisinf(input[tx]);
}

__global__ void testHalfAbs(float* p) {
  auto a = __float2half(*p);
  a = __habs(a);
  *p = __half2float(a);
}

__global__ void testHalf2Abs(float2* p) {
  auto a = __float22half2_rn(*p);
  a = __habs2(a);
  *p = __half22float2(a);
}

__half host_ushort_as_half(uint32_t s) {
  union {
    __half h;
    uint32_t s;
  } converter;
  converter.s = s;
  return converter.h;
}

void check_hisnan(int NUM_INPUTS, __half* inputCPU, __half* inputGPU) {
  // allocate memory
  auto memsize = NUM_INPUTS * sizeof(int);
  int* outputGPU = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&outputGPU), memsize));

  // launch the kernel
  hipLaunchKernelGGL(kernel_hisnan, dim3(1), dim3(NUM_INPUTS), 0, 0, inputGPU, outputGPU);

  // copy output from device
  int* outputCPU = reinterpret_cast<int*>(malloc(memsize));
  HIP_CHECK(hipMemcpy(outputCPU, outputGPU, memsize, hipMemcpyDeviceToHost));

  // check output
  for (int i = 0; i < NUM_INPUTS; i++) {
    if ((2 <= i) && (i <= 5)) {  // inputs are nan, output should be true
      REQUIRE(outputCPU[i] == true);
    } else {  // inputs are NOT nan, output should be false
      REQUIRE(outputCPU[i] == false);
    }
  }

  // free memory
  free(outputCPU);
  HIP_CHECK(hipFree(outputGPU));
}


void check_hisinf(int NUM_INPUTS, __half* inputCPU, __half* inputGPU) {
  // allocate memory
  auto memsize = NUM_INPUTS * sizeof(int);
  int* outputGPU = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&outputGPU), memsize));

  // launch the kernel
  hipLaunchKernelGGL(kernel_hisinf, dim3(1), dim3(NUM_INPUTS), 0, 0, inputGPU, outputGPU);

  // copy output from device
  int* outputCPU = reinterpret_cast<int*>(malloc(memsize));
  HIP_CHECK(hipMemcpy(outputCPU, outputGPU, memsize, hipMemcpyDeviceToHost));

  // check output
  for (int i = 0; i < NUM_INPUTS; i++) {
    if ((0 <= i) && (i <= 1)) {  // inputs are inf, output should be true
      REQUIRE(outputCPU[i] == true);
    } else {  // inputs are NOT inf, output should be false
      REQUIRE(outputCPU[i] == false);
    }
  }
  // free memory
  free(outputCPU);
  HIP_CHECK(hipFree(outputGPU));
}


void checkFunctional() {
  // allocate memory
  const int NUM_INPUTS = 16;
  auto memsize = NUM_INPUTS * sizeof(__half);
  __half* inputCPU = reinterpret_cast<__half*>(malloc(memsize));

  // populate inputs
  inputCPU[0] = host_ushort_as_half(0x7c00);   //  inf
  inputCPU[1] = host_ushort_as_half(0xfc00);   // -inf
  inputCPU[2] = host_ushort_as_half(0x7c01);   //  nan
  inputCPU[3] = host_ushort_as_half(0x7e00);   //  nan
  inputCPU[4] = host_ushort_as_half(0xfc01);   //  nan
  inputCPU[5] = host_ushort_as_half(0xfe00);   //  nan
  inputCPU[6] = host_ushort_as_half(0x0000);   //  0
  inputCPU[7] = host_ushort_as_half(0x8000);   // -0
  inputCPU[8] = host_ushort_as_half(0x7bff);   // max +ve normal
  inputCPU[9] = host_ushort_as_half(0xfbff);   // max -ve normal
  inputCPU[10] = host_ushort_as_half(0x0400);  // min +ve normal
  inputCPU[11] = host_ushort_as_half(0x8400);  // min -ve normal
  inputCPU[12] = host_ushort_as_half(0x03ff);  // max +ve sub-normal
  inputCPU[13] = host_ushort_as_half(0x83ff);  // max -ve sub-normal
  inputCPU[14] = host_ushort_as_half(0x0001);  // min +ve sub-normal
  inputCPU[15] = host_ushort_as_half(0x8001);  // min -ve sub-normal

  // copy inputs to the GPU
  __half* inputGPU = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&inputGPU), memsize));
  HIP_CHECK(hipMemcpy(inputGPU, inputCPU, memsize, hipMemcpyHostToDevice));

  // run checks
  check_hisnan(NUM_INPUTS, inputCPU, inputGPU);
  check_hisinf(NUM_INPUTS, inputCPU, inputGPU);

  // free memory
  HIP_CHECK(hipFree(inputGPU));
  free(inputCPU);
}

void checkHalfAbs() {
  SECTION("Half Abs") {
    float* p;
    HIP_CHECK(hipMalloc(&p, sizeof(float)));
    float pp = -2.1f;
    HIP_CHECK(hipMemcpy(p, &pp, sizeof(float), hipMemcpyDefault));
    hipLaunchKernelGGL(testHalfAbs, 1, 1, 0, 0, p);
    HIP_CHECK(hipMemcpy(&pp, p, sizeof(float), hipMemcpyDefault));
    HIP_CHECK(hipFree(p));
    REQUIRE(pp >= 0.0f);
  }
  SECTION("Half2 Abs") {
    float2* p;
    HIP_CHECK(hipMalloc(&p, sizeof(float2)));
    float2 pp;
    pp.x = -2.1f;
    pp.y = -1.1f;
    HIP_CHECK(hipMemcpy(p, &pp, sizeof(float2), hipMemcpyDefault));
    hipLaunchKernelGGL(testHalf2Abs, 1, 1, 0, 0, p);
    HIP_CHECK(hipMemcpy(&pp, p, sizeof(float2), hipMemcpyDefault));
    HIP_CHECK(hipFree(p));
    bool result = true;
    if (pp.x < 0.0f || pp.y < 0.0f) {
      result = false;
    }
    REQUIRE(result == true);
  }
}

TEST_CASE("Unit_hipTestHalf") {
  bool* result{nullptr};
  HIP_CHECK(hipHostMalloc(&result, sizeof(result)));

  SECTION("Test half math") {
    result[0] = false;
    hipLaunchKernelGGL(__halfMath, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, result, __half{1});
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(result[0] == true);
  }
  SECTION("Test half math") {
    result[0] = false;
    hipLaunchKernelGGL(__half2Math, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, result, __half2{1, 1});
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(result[0] == true);
  }
  SECTION("Functional checks") {
    checkFunctional();
    checkHalfAbs();
  }
  HIP_CHECK(hipHostFree(result));
}
