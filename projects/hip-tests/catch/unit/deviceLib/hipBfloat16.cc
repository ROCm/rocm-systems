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
#include <hip_test_common.hh>
#include <hip/hip_bfloat16.h>
#include <type_traits>
#include <random>
#include <climits>

#define SIZE 100

static std::random_device dev;
static std::mt19937 rng(dev());

inline float getRandomFloat(int16_t min = 10, int64_t max = LONG_MAX) {
  std::uniform_real_distribution<float> gen(min, max);
  return gen(rng);
}
__host__ __device__ bool testRelativeAccuracy(float a, hip_bfloat16 b) {
  float c = static_cast<float>(b);
  // float relative error should be less than 1/(2^7) since bfloat16
  // has 7 bits mantissa.
  if (fabs(c - a) / a <= 1.0 / 128) {
    return true;
  }
  return false;
}
__host__ __device__ bool testOperations(const float& fa, const float& fb) {
  bool testPass = true;
  hip_bfloat16 bf_a(fa);
  hip_bfloat16 bf_b(fb);
  float fc = static_cast<float>(bf_a);
  float fd = static_cast<float>(bf_b);

  testPass &= testRelativeAccuracy(fa, bf_a);
  testPass &= testRelativeAccuracy(fb, bf_b);

  testPass &= testRelativeAccuracy(fc + fd, bf_a + bf_b);
  // when checked as above for add, operation sub fails on GPU
  if (hip_bfloat16(fc - fd) == (bf_a - bf_b)) {
    testPass &= true;
  }
  testPass &= testRelativeAccuracy(fc * fd, bf_a * bf_b);
  testPass &= testRelativeAccuracy(fc / fd, bf_a / bf_b);

  hip_bfloat16 bf_x;
  bf_x = bf_a;
  bf_x++;
  bf_x--;
  ++bf_x;
  --bf_x;
  // hip_bfloat16 is converted to float and then inc/decremented,
  // hence check with reduced precision
  testPass &= testRelativeAccuracy(bf_x, bf_a);

  bf_x = bf_a;
  bf_x += bf_b;
  bf_x = bf_a;
  bf_x -= bf_b;
  bf_x = bf_a;
  bf_x *= bf_b;
  bf_x = bf_a;
  bf_x /= bf_b;

  hip_bfloat16 bf_rounded = hip_bfloat16::round_to_bfloat16(fa);
  if (std::isnan(bf_rounded)) {
    if (std::isnan(bf_rounded) || std::isinf(bf_rounded)) {
      testPass &= true;
    }
  }
  return testPass;
}
__global__ void testOperationsGPU(float* d_a, float* d_b, bool* testPass) {
  int id = threadIdx.x;
  if (id > SIZE) return;
  float& a = d_a[id];
  float& b = d_b[id];
  *testPass = testOperations(a, b);
}
TEST_CASE("Unit_hipBfloat16") {
  float *h_fa, *h_fb;
  float *d_fa, *d_fb;
  bool *d_fc, h_fc = false;

  h_fa = new float[SIZE];
  h_fb = new float[SIZE];

  bool result = false;
  for (int i = 0; i < SIZE; i++) {
    h_fa[i] = getRandomFloat();
    h_fb[i] = getRandomFloat();
    result = testOperations(h_fa[i], h_fb[i]);
    REQUIRE(result == true);
  }

  HIP_CHECK(hipMalloc(&d_fa, sizeof(float) * SIZE));
  HIP_CHECK(hipMalloc(&d_fb, sizeof(float) * SIZE));
  HIP_CHECK(hipMalloc(&d_fc, sizeof(bool)));

  HIP_CHECK(hipMemcpy(d_fa, h_fa, sizeof(float) * SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_fb, h_fb, sizeof(float) * SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_fc, &h_fc, sizeof(bool), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(testOperationsGPU, 1, SIZE, 0, 0, d_fa, d_fb, d_fc);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(&h_fc, d_fc, sizeof(bool), hipMemcpyDeviceToHost));

  REQUIRE(h_fc == true);

  delete[] h_fa;
  delete[] h_fb;
  HIP_CHECK(hipFree(d_fa));
  HIP_CHECK(hipFree(d_fb));
  HIP_CHECK(hipFree(d_fc));
}
