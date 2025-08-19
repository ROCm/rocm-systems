/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include <algorithm>

#include <hip_test_common.hh>

#include <hip/hip_fp4.h>

template <typename Lambda, typename... Type>
static __global__ void lambda_kernel_launch(Lambda l, Type... args) {
  l(args...);
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to convert given double type data to FP4 type with E2M1
 * format.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/fp4_ocp.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_all_fp4_from_double") {
  SECTION("sanityx1") {
    std::vector<double> inputs{-1.0, 0.0, 1.0};
    for (const auto input : inputs) {
      __hip_fp4_e2m1 fp4(input);
      double ret = fp4;
      INFO("Original: " << input << " Return: " << ret);
      REQUIRE(ret == input);
    }
  }
  SECTION("sanityx2") {
    std::vector<double2> inputs{{-1.0, 1.0}, {-2.0, 2.0}};
    for (const auto input : inputs) {
      __hip_fp4x2_e2m1 fp4x2(input);
      double2 ret = fp4x2;
      INFO("Original: " << input.x << " Return: " << ret.x);
      INFO("Original: " << input.y << " Return: " << ret.y);

      REQUIRE(ret.x == input.x);
      REQUIRE(ret.y == input.y);
    }
  }
  SECTION("sanityx4") {
    std::vector<double4> inputs{
        {-1.0, 0.5, 1.5, 1.0}, {-2.0, 0.5, 1.5, 2.0}, {-3.0, 0.5, 1.5, 3.0}};
    for (const auto& input : inputs) {
      __hip_fp4x4_e2m1 fp4x4(input);
      double4 ret = fp4x4;
      INFO("Original: " << input.x << ", " << input.y << ", " << input.z << ", " << input.w
                        << " Return: " << ret.x << ", " << ret.y << ret.z << ", " << ret.w);
      REQUIRE(ret.x == input.x);
      REQUIRE(ret.y == input.y);
      REQUIRE(ret.z == input.z);
      REQUIRE(ret.w == input.w);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to convert given double data to FP4 type with E2M1
 * format in the device.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/fp4_ocp.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_all_fp4_from_double_device") {
  SECTION("sanityx1") {
    auto fp4x1_l = [] __device__(double* inputs, float* outputs, size_t size) {
      int i = threadIdx.x;
      if (i < size) {
        __hip_fp4_e2m1 fp4(inputs[i]);
        outputs[i] = fp4;
      }
    };

    std::vector<double> inputs{-1.0, 0.0, 1.0};
    double* d_in;
    float* d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(double) * inputs.size()));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float) * inputs.size()));

    HIP_CHECK(
        hipMemcpy(d_in, inputs.data(), sizeof(double) * inputs.size(), hipMemcpyHostToDevice));
    lambda_kernel_launch<<<1, 32>>>(fp4x1_l, d_in, d_out, inputs.size());
    std::vector<float> outputs(inputs.size(), 0.0f);
    HIP_CHECK(
        hipMemcpy(outputs.data(), d_out, sizeof(float) * inputs.size(), hipMemcpyDeviceToHost));

    for (size_t i = 0; i < inputs.size(); i++) {
      INFO("Original: " << inputs[i] << " Output: " << outputs[i]);
      REQUIRE(inputs[i] == outputs[i]);
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("sanityx2") {
    auto fp4x2_l = [] __device__(double2 * inputs, float2 * outputs, size_t size) {
      int i = threadIdx.x;
      if (i < size) {
        __hip_fp4x2_e2m1 fp4(inputs[i]);
        outputs[i] = fp4;
      }
    };

    std::vector<double2> inputs{{-1.0, 0.0}, {0.0, 1.0}, {1.0, -1.0}, {1.0, 0.0}, {0.0, -1.0}};
    double2* d_in;
    float2* d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(double2) * inputs.size()));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float2) * inputs.size()));

    HIP_CHECK(
        hipMemcpy(d_in, inputs.data(), sizeof(double2) * inputs.size(), hipMemcpyHostToDevice));
    lambda_kernel_launch<<<1, 32>>>(fp4x2_l, d_in, d_out, inputs.size());
    std::vector<float2> outputs(inputs.size());
    HIP_CHECK(
        hipMemcpy(outputs.data(), d_out, sizeof(float2) * inputs.size(), hipMemcpyDeviceToHost));

    for (size_t i = 0; i < inputs.size(); i++) {
      INFO("Original: " << inputs[i].x << ", " << inputs[i].y << " Output: " << outputs[i].x << ", "
                        << outputs[i].y);
      REQUIRE(inputs[i].x == outputs[i].x);
      REQUIRE(inputs[i].y == outputs[i].y);
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("sanityx4") {
    auto fp4x4_l = [] __device__(double4 * inputs, float4 * outputs, size_t size) {
      int i = threadIdx.x;
      if (i < size) {
        __hip_fp4x4_e2m1 fp4(inputs[i]);
        outputs[i] = fp4;
      }
    };

    std::vector<double4> inputs{
        {-1.0, 0.0, 1.0, 0.5}, {0.0, 1.0, -0.5, -1.0}, {1.0, 0.0, 1.0, -1.0}};
    double4* d_in;
    float4* d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(double4) * inputs.size()));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float4) * inputs.size()));

    HIP_CHECK(
        hipMemcpy(d_in, inputs.data(), sizeof(double4) * inputs.size(), hipMemcpyHostToDevice));
    lambda_kernel_launch<<<1, 32>>>(fp4x4_l, d_in, d_out, inputs.size());
    std::vector<float4> outputs(inputs.size());
    HIP_CHECK(
        hipMemcpy(outputs.data(), d_out, sizeof(float4) * inputs.size(), hipMemcpyDeviceToHost));

    for (size_t i = 0; i < inputs.size(); i++) {
      INFO("Original: " << inputs[i].x << ", " << inputs[i].y << ", " << inputs[i].z << ", "
                        << inputs[i].w << " Output: " << outputs[i].x << ", " << outputs[i].y
                        << ", " << outputs[i].z << ", " << outputs[i].w);
      REQUIRE(inputs[i].x == outputs[i].x);
      REQUIRE(inputs[i].y == outputs[i].y);
      REQUIRE(inputs[i].z == outputs[i].z);
      REQUIRE(inputs[i].w == outputs[i].w);
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to convert given signed interger data to FP4 type with E2M1
 * format.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/fp4_ocp.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEMPLATE_TEST_CASE("Unit_all_fp4_from_interger_data", "", int, long int, long long int, short int) {
  SECTION("Fp4 with e2m1") {
    std::vector<TestType> input{-1, 0, 1};
    for (const auto val : input) {
      __hip_fp4_e2m1 fp4(val);
      float ret = fp4;
      INFO("In: " << val);
      INFO("Out: " << ret);
      REQUIRE(ret == val);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to convert given unsigned integer data to FP4 type with E2M1
 * format.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/fp4_ocp.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEMPLATE_TEST_CASE("Unit_all_fp4_from__unsigned_integer_data", "", unsigned int, unsigned long int,
                   unsigned long long int, unsigned short int) {
  SECTION("Fp4 with e2m1") {
    std::vector<TestType> input{1, 2, 3};
    for (const auto val : input) {
      __hip_fp4_e2m1 fp4(val);
      float ret = fp4;
      INFO("In: " << val);
      INFO("Out: " << ret);
      REQUIRE(ret == val);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to convert given signed interger data to FP4 type in device with E2M1
 * format.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/fp4_ocp.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */

TEMPLATE_TEST_CASE("Unit_all_fp4_from_integer_data_device", "", int, long int, long long int,
                   short int) {
  std::vector<float> all_fp4{-6.0f, -4.0f, -3.0f, -2.0f, -1.5f, -1.0f, -0.5f, 0.0f,
                             0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f};
  auto fp4x1_l = [] __device__(TestType * inputs, float* outputs, size_t size) {
    int i = threadIdx.x;
    if (i < size) {
      __hip_fp4_e2m1 fp4(inputs[i]);
      outputs[i] = fp4;
    }
  };

  std::vector<TestType> inputs;
  inputs.reserve(30);
  for (int i = 0; i <= 6; i += 1) {
    inputs.push_back(i);
  }
  TestType* d_in;
  float* d_out;
  HIP_CHECK(hipMalloc(&d_in, sizeof(TestType) * inputs.size()));
  HIP_CHECK(hipMalloc(&d_out, sizeof(float) * inputs.size()));

  HIP_CHECK(
      hipMemcpy(d_in, inputs.data(), sizeof(TestType) * inputs.size(), hipMemcpyHostToDevice));
  lambda_kernel_launch<<<1, 32>>>(fp4x1_l, d_in, d_out, inputs.size());

  std::vector<float> outputs(inputs.size(), 0.0f);
  HIP_CHECK(hipMemcpy(outputs.data(), d_out, sizeof(float) * inputs.size(), hipMemcpyDeviceToHost));

  for (size_t i = 0; i < inputs.size(); i++) {
    auto lbound = std::lower_bound(all_fp4.begin(), all_fp4.end(), outputs[i]);
    INFO("Original: " << inputs[i] << " Output: " << *lbound);
    REQUIRE(*lbound == outputs[i]);
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to convert given unsigned interger data to FP4 type in device with E2M1
 * format.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/fp4_ocp.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEMPLATE_TEST_CASE("Unit_all_fp4_from__unsigned_integer_data_device", "", unsigned int,
                   unsigned long int, unsigned long long int, unsigned short int) {
  std::vector<float> all_fp4{-6.0f, -4.0f, -3.0f, -2.0f, -1.5f, -1.0f, -0.5f, 0.0f,
                             0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f};
  auto fp4x1_l = [] __device__(TestType * inputs, float* outputs, size_t size) {
    int i = threadIdx.x;
    if (i < size) {
      __hip_fp4_e2m1 fp4(inputs[i]);
      outputs[i] = fp4;
    }
  };

  std::vector<TestType> inputs;
  inputs.reserve(30);
  for (int i = -6; i <= 6; i += 1) {
    inputs.push_back(i);
  }
  TestType* d_in;
  float* d_out;
  HIP_CHECK(hipMalloc(&d_in, sizeof(TestType) * inputs.size()));
  HIP_CHECK(hipMalloc(&d_out, sizeof(float) * inputs.size()));

  HIP_CHECK(
      hipMemcpy(d_in, inputs.data(), sizeof(TestType) * inputs.size(), hipMemcpyHostToDevice));
  lambda_kernel_launch<<<1, 32>>>(fp4x1_l, d_in, d_out, inputs.size());

  std::vector<float> outputs(inputs.size(), 0.0f);
  HIP_CHECK(hipMemcpy(outputs.data(), d_out, sizeof(float) * inputs.size(), hipMemcpyDeviceToHost));

  for (size_t i = 0; i < inputs.size(); i++) {
    auto lbound = std::lower_bound(all_fp4.begin(), all_fp4.end(), outputs[i]);
    INFO("Original: " << inputs[i] << " Output: " << *lbound);
    REQUIRE(*lbound == outputs[i]);
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to convert given double type data to FP4 type with E2M3 and
 * E3M2 formats.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/fp4_ocp.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_ocp_fp4_from_double_full_range_host") {
  std::vector<double> in;
  in.reserve(30);
  for (double i = -6.0; i <= 6.0; i += 0.5) {
    in.push_back(i);
  }

  std::vector<float> expected{-6.0f, -6.0f, -4.0f, -4.0f, -4.0f, -4.0f, -3.0f, -2.0f, -2.0f,
                              -1.5f, -1.0f, -0.5f, 0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  2.0f,
                              3.0f,  4.0f,  4.0f,  4.0f,  4.0f,  6.0f,  6.0f};

  for (size_t i = 0; i < in.size(); i++) {
    __hip_fp4_e2m1 fp4(in[i]);
    float fp32 = fp4;
    INFO("Original: " << in[i] << " Output: " << fp32 << " Expected: " << expected[i]);
    REQUIRE(expected[i] == fp32);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to convert given double type data to FP4 type in device with
 * E2M3 and E3M2 formats.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/fp4_ocp.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */

TEST_CASE("Unit_ocp_fp4_from_double_full_range_device") {
  std::vector<float> all_fp4{-6.0f, -4.0f, -3.0f, -2.0f, -1.5f, -1.0f, -0.5f, 0.0f,
                             0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f};
  auto fp4x1_l = [] __device__(double* inputs, float* outputs, size_t size) {
    int i = threadIdx.x;
    if (i < size) {
      __hip_fp4_e2m1 fp4(inputs[i]);
      outputs[i] = fp4;
    }
  };

  std::vector<double> inputs;
  inputs.reserve(30);
  for (double i = -6.0; i <= 6.0; i += 0.5) {
    inputs.push_back(i);
  }
  double* d_in;
  float* d_out;
  HIP_CHECK(hipMalloc(&d_in, sizeof(double) * inputs.size()));
  HIP_CHECK(hipMalloc(&d_out, sizeof(float) * inputs.size()));

  HIP_CHECK(hipMemcpy(d_in, inputs.data(), sizeof(double) * inputs.size(), hipMemcpyHostToDevice));
  lambda_kernel_launch<<<1, 32>>>(fp4x1_l, d_in, d_out, inputs.size());

  std::vector<float> outputs(inputs.size(), 0.0f);
  HIP_CHECK(hipMemcpy(outputs.data(), d_out, sizeof(float) * inputs.size(), hipMemcpyDeviceToHost));

  for (size_t i = 0; i < inputs.size(); i++) {
    auto lbound = std::lower_bound(all_fp4.begin(), all_fp4.end(), outputs[i]);
    INFO("Original: " << inputs[i] << " Output: " << *lbound);
    REQUIRE(*lbound == outputs[i]);
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
}
