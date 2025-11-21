/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip_test_common.hh>
#include <hip/hip_fp8.h>

#include <type_traits>
#include <vector>
#include <bitset>
#ifdef __gfx942__
#define __gfx94plus_local__
#endif

/*
 * This catch test is meant for FP8 FNUZ conversion checking
 * tests only supported on gfx942 archs.
 */

static_assert(sizeof(unsigned int) == sizeof(float));

std::string get_arch_type() {
  hipDeviceProp_t prop;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  std::string gfxName(prop.gcnArchName);
  return gfxName;
}

#define ARCH_TYPE_GFX940(name) (name.find("gfx942") != std::string::npos)

#define FP8_FNUZ_SKIP_TEST                                                                         \
  std::string gfxName = get_arch_type();                                                           \
  if (!(ARCH_TYPE_GFX940(gfxName))) {                                                              \
    HipTest::HIP_SKIP_TEST("This test can only be run on gfx942 arch");                            \
    return;                                                                                        \
  }

#define __FP8_DEVICE__ __device__ static inline

template <typename T> __FP8_DEVICE__ void e4m3_fnuz_device(T* val) {
#if defined(__gfx94plus_local__) && __HIP_DEVICE_COMPILE__
  __hip_fp8_e4m3_fnuz tmp(*val);
  *val = tmp;
#else
  *val = 0;
#endif
}

template <typename T> __FP8_DEVICE__ void e5m2_fnuz_device(T* val) {
#if defined(__gfx94plus_local__) && __HIP_DEVICE_COMPILE__
  __hip_fp8_e5m2_fnuz tmp(*val);
  *val = tmp;
#else
  *val = 0;
#endif
}

template <typename T, bool is_e4m3_fnuz>
__global__ void cvt_float_fp8_float_fnuz(T* in, size_t len) {
  int i = threadIdx.x;
  if (i < len) {
    T val = in[i];
    if constexpr (is_e4m3_fnuz) {
      e4m3_fnuz_device<T>(&val);
      in[i] = val;
    } else {
      e5m2_fnuz_device<T>(&val);
      in[i] = val;
    }
  }
}

template <typename T, bool is_e4m3_fnuz>
std::vector<T> cpu_cvt_float_fp8_float_fnuz(const std::vector<T>& nums) {
  std::vector<T> ret;
  ret.reserve(nums.size());
  for (const auto& num : nums) {
    T out = 0.0;
    if constexpr (is_e4m3_fnuz) {
      __hip_fp8_e4m3_fnuz tmp(num);
      out = tmp;
    } else {
      __hip_fp8_e5m2_fnuz tmp(num);
      out = tmp;
    }
    ret.push_back(out);
  }
  return ret;
}

// floats to fp8
TEMPLATE_TEST_CASE("Unit_fp8_fnuz_compare_host_device", "", float, double) {
  FP8_FNUZ_SKIP_TEST
  std::vector<TestType> numbers = {0.0f, 1.0f, 1.1f, 2.0f,  2.1f,  3.0f,  3.2f,
                                   3.3f, 4.0f, 4.5f, 10.0f, 11.0f, 12.2f, 14.1f};
  TestType* d_numbers;
  HIP_CHECK(hipMalloc(&d_numbers, sizeof(TestType) * numbers.size()));
  HIP_CHECK(hipMemcpy(d_numbers, numbers.data(), sizeof(TestType) * numbers.size(),
                      hipMemcpyHostToDevice));

  std::vector<TestType> result(numbers.size(), 0.0f);
  std::vector<TestType> cpu_result;

  SECTION("e4m3_fnuz") {
    cpu_result = cpu_cvt_float_fp8_float_fnuz<TestType, true>(numbers);
    auto kernel = cvt_float_fp8_float_fnuz<TestType, true>;
    kernel<<<1, numbers.size()>>>(d_numbers, numbers.size());
    HIP_CHECK(hipMemcpy(result.data(), d_numbers, sizeof(TestType) * numbers.size(),
                        hipMemcpyDeviceToHost));
  }

  SECTION("e5m2_fnuz") {
    cpu_result = cpu_cvt_float_fp8_float_fnuz<TestType, false>(numbers);
    auto kernel = cvt_float_fp8_float_fnuz<TestType, false>;
    kernel<<<1, numbers.size()>>>(d_numbers, numbers.size());
    HIP_CHECK(hipMemcpy(result.data(), d_numbers, sizeof(TestType) * numbers.size(),
                        hipMemcpyDeviceToHost));
  }

  REQUIRE(cpu_result.size() == result.size());
  for (size_t i = 0; i < result.size(); i++) {
    INFO("Checking: " << numbers[i] << " cpu convert: " << cpu_result[i]
                      << " - gpu_result: " << result[i]);
    CHECK(cpu_result[i] == result[i]);
  }
}

__FP8_DEVICE__ void e4m3_fp8x2_fnuz_device(float2* val) {
#if defined(__gfx94plus_local__) && __HIP_DEVICE_COMPILE__
  __hip_fp8x2_e4m3_fnuz tmp(*val);
  *val = tmp;
#else
  *val = float2(0.0, 0.0);
#endif
}

__FP8_DEVICE__ void e5m2_fp8x2_fnuz_device(float2* val) {
#if defined(__gfx94plus_local__) && __HIP_DEVICE_COMPILE__
  __hip_fp8x2_e5m2_fnuz tmp(*val);
  *val = tmp;
#else
  *val = float2(0.0, 0.0);
#endif
}

template <bool is_e4m3_fnuz> __global__ void cvt_float2_fp8x2_float2_fnuz(float2* in, size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    float2 val = in[i];
    if constexpr (is_e4m3_fnuz) {
      e4m3_fp8x2_fnuz_device(&val);
      in[i] = val;
    } else {
      e5m2_fp8x2_fnuz_device(&val);
      in[i] = val;
    }
  }
}

template <bool is_e4m3_fnuz>
std::vector<float2> cpu_cvt_float2_fp8x2_float2_fnuz(const std::vector<float2>& nums) {
  std::vector<float2> ret;
  ret.reserve(nums.size());
  for (const auto& num : nums) {
    float2 out = {0.0f, 0.0f};
    if constexpr (is_e4m3_fnuz) {
      __hip_fp8x2_e4m3_fnuz tmp(num);
      out = tmp;
    } else {
      __hip_fp8x2_e5m2_fnuz tmp(num);
      out = tmp;
    }
    ret.push_back(out);
  }
  return ret;
}

TEST_CASE("Unit_fp8x2_fnuz_compare_host_device") {
  FP8_FNUZ_SKIP_TEST
  std::vector<float> numbers_input = {0.0f, 1.0f, 1.1f, 2.0f,  2.1f,  3.0f,  3.2f,
                                      3.3f, 4.0f, 4.5f, 10.0f, 11.0f, 12.2f, 14.1f};

  std::vector<float2> numbers;
  numbers.reserve(numbers_input.size());
  for (size_t i = 0, end = numbers_input.size() - 1; i < numbers_input.size(); i++, end--) {
    float2 ret(numbers_input[i], numbers_input[end]);
    numbers.push_back(ret);
  }

  float2* d_numbers;
  HIP_CHECK(hipMalloc(&d_numbers, sizeof(float2) * numbers.size()));
  HIP_CHECK(
      hipMemcpy(d_numbers, numbers.data(), sizeof(float2) * numbers.size(), hipMemcpyHostToDevice));

  std::vector<float2> result(numbers.size(), float2{0.0f, 0.0f});
  std::vector<float2> cpu_result;

  SECTION("e4m3_fnuz") {
    cpu_result = cpu_cvt_float2_fp8x2_float2_fnuz<true>(numbers);
    auto kernel = cvt_float2_fp8x2_float2_fnuz<true>;
    kernel<<<1, numbers.size()>>>(d_numbers, numbers.size());
    HIP_CHECK(hipMemcpy(result.data(), d_numbers, sizeof(float2) * numbers.size(),
                        hipMemcpyDeviceToHost));
  }

  SECTION("e5m2_fnuz") {
    cpu_result = cpu_cvt_float2_fp8x2_float2_fnuz<false>(numbers);
    auto kernel = cvt_float2_fp8x2_float2_fnuz<false>;
    kernel<<<1, numbers.size()>>>(d_numbers, numbers.size());
    HIP_CHECK(hipMemcpy(result.data(), d_numbers, sizeof(float2) * numbers.size(),
                        hipMemcpyDeviceToHost));
  }

  REQUIRE(cpu_result.size() == result.size());
  for (size_t i = 0; i < result.size(); i++) {
    CHECK(cpu_result[i] == result[i]);
  }
}

TEST_CASE("Unit_fp8x2_fnuz_split_compare") {
  FP8_FNUZ_SKIP_TEST
  std::vector<float> numbers_input = {0.0f, 1.0f, 1.1f, 2.0f,  2.1f,  3.0f,  3.2f,
                                      3.3f, 4.0f, 4.5f, 10.0f, 11.0f, 12.2f, 14.1f};

  std::vector<float2> numbers;
  numbers.reserve(numbers_input.size());
  for (size_t i = 0, end = numbers_input.size() - 1; i < numbers_input.size(); i++, end--) {
    float2 ret(numbers_input[i], numbers_input[end]);
    numbers.push_back(ret);
  }

  float2* d_numbers;
  HIP_CHECK(hipMalloc(&d_numbers, sizeof(float2) * numbers.size()));
  HIP_CHECK(
      hipMemcpy(d_numbers, numbers.data(), sizeof(float2) * numbers.size(), hipMemcpyHostToDevice));

  std::vector<float2> result(numbers.size(), float2{0.0f, 0.0f});
  std::vector<float2> cpu_result;
  cpu_result.reserve(result.size());

  SECTION("e4m3_fnuz") {
    for (const auto& num : numbers) {
      __hip_fp8_e4m3_fnuz t_a(num.x);
      __hip_fp8_e4m3_fnuz t_b(num.y);
      float a = t_a, b = t_b;
      cpu_result.push_back(float2(a, b));
    }
    auto kernel = cvt_float2_fp8x2_float2_fnuz<true>;
    kernel<<<1, numbers.size()>>>(d_numbers, numbers.size());
    HIP_CHECK(hipMemcpy(result.data(), d_numbers, sizeof(float2) * numbers.size(),
                        hipMemcpyDeviceToHost));
  }

  SECTION("e5m2_fnuz") {
    for (const auto& num : numbers) {
      __hip_fp8_e5m2_fnuz t_a(num.x);
      __hip_fp8_e5m2_fnuz t_b(num.y);
      float a = t_a, b = t_b;
      cpu_result.push_back(float2(a, b));
    }
    auto kernel = cvt_float2_fp8x2_float2_fnuz<false>;
    kernel<<<1, numbers.size()>>>(d_numbers, numbers.size());
    HIP_CHECK(hipMemcpy(result.data(), d_numbers, sizeof(float2) * numbers.size(),
                        hipMemcpyDeviceToHost));
  }

  HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(cpu_result.size() == result.size());
  for (size_t i = 0; i < result.size(); i++) {
    INFO("cpu x: " << cpu_result[i].x << " y: " << cpu_result[i].y << " gpu x: " << result[i].x
                   << " y: " << result[i].y);
    CHECK(cpu_result[i] == result[i]);
  }
}

__FP8_DEVICE__ void e4m3_fp8x4_fnuz_device(float4* val) {
#if defined(__gfx94plus_local__) && __HIP_DEVICE_COMPILE__
  __hip_fp8x4_e4m3_fnuz tmp(*val);
  *val = tmp;
#else
  *val = float4(0.0, 0.0, 0.0, 0.0);
#endif
}

__FP8_DEVICE__ void e5m2_fp8x4_fnuz_device(float4* val) {
#if defined(__gfx94plus_local__) && __HIP_DEVICE_COMPILE__
  __hip_fp8x4_e5m2_fnuz tmp(*val);
  *val = tmp;
#else
  *val = float4(0.0, 0.0, 0.0, 0.0);
#endif
}

template <bool is_e4m3_fnuz> __global__ void cvt_float4_fp8x4_float4_fnuz(float4* in, size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    float4 val = in[i];
    if constexpr (is_e4m3_fnuz) {
      e4m3_fp8x4_fnuz_device(&val);
      in[i] = val;
    } else {
      e5m2_fp8x4_fnuz_device(&val);
      in[i] = val;
    }
  }
}

TEST_CASE("Unit_fp8x4_fnuz_split_compare") {
  FP8_FNUZ_SKIP_TEST
  std::vector<float> numbers_input = {0.0f, 1.0f, 1.1f, 2.0f,  2.1f,  3.0f,  3.2f,
                                      3.3f, 4.0f, 4.5f, 10.0f, 11.0f, 12.2f, 14.1f};
  std::vector<float> numbers_input2 = {1.3f, 1.6f, 1.8f, 2.5f,  2.9f,  3.8f,  3.9f,
                                       5.5f, 7.1f, 8.5f, 11.2f, 13.5f, 16.1f, 19.4f};

  std::vector<float4> numbers;
  numbers.reserve(numbers_input.size());
  for (size_t i = 0, end = numbers_input.size() - 1; i < numbers_input.size(); i++, end--) {
    float4 ret(numbers_input[i], numbers_input[end], numbers_input2[i], numbers_input2[end]);
    numbers.push_back(ret);
  }

  float4* d_numbers;
  HIP_CHECK(hipMalloc(&d_numbers, sizeof(float4) * numbers.size()));
  HIP_CHECK(
      hipMemcpy(d_numbers, numbers.data(), sizeof(float4) * numbers.size(), hipMemcpyHostToDevice));

  std::vector<float4> result(numbers.size(), float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<float4> cpu_result;
  cpu_result.reserve(result.size());

  SECTION("e4m3_fnuz") {
    for (const auto& num : numbers) {
      __hip_fp8_e4m3_fnuz t_a(num.x);
      __hip_fp8_e4m3_fnuz t_b(num.y);
      __hip_fp8_e4m3_fnuz t_c(num.z);
      __hip_fp8_e4m3_fnuz t_d(num.w);
      float a = t_a, b = t_b, c = t_c, d = t_d;
      cpu_result.push_back(float4(a, b, c, d));
    }
    auto kernel = cvt_float4_fp8x4_float4_fnuz<true>;
    kernel<<<1, numbers.size()>>>(d_numbers, numbers.size());
    HIP_CHECK(hipMemcpy(result.data(), d_numbers, sizeof(float4) * numbers.size(),
                        hipMemcpyDeviceToHost));
  }

  SECTION("e5m2_fnuz") {
    for (const auto& num : numbers) {
      __hip_fp8_e5m2_fnuz t_a(num.x);
      __hip_fp8_e5m2_fnuz t_b(num.y);
      __hip_fp8_e5m2_fnuz t_c(num.z);
      __hip_fp8_e5m2_fnuz t_d(num.w);
      float a = t_a, b = t_b, c = t_c, d = t_d;
      cpu_result.push_back(float4(a, b, c, d));
    }
    auto kernel = cvt_float4_fp8x4_float4_fnuz<false>;
    kernel<<<1, numbers.size()>>>(d_numbers, numbers.size());
    HIP_CHECK(hipMemcpy(result.data(), d_numbers, sizeof(float4) * numbers.size(),
                        hipMemcpyDeviceToHost));
  }

  HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(cpu_result.size() == result.size());
  for (size_t i = 0; i < result.size(); i++) {
    INFO("original: x: " << numbers[i].x << " y: " << numbers[i].y << " z: " << numbers[i].z
                         << " w: " << numbers[i].w);
    INFO("cpu x: " << cpu_result[i].x << " y: " << cpu_result[i].y << " z: " << cpu_result[i].z
                   << " w: " << cpu_result[i].w);
    INFO("gpu x: " << result[i].x << " y: " << result[i].y << " z: " << result[i].z
                   << " w: " << result[i].w);
    CHECK(cpu_result[i] == result[i]);
  }
}
__FP8_DEVICE__ bool e4m3_bool_fnuz_device(float val) {
  bool x = false;
  float y = val;
#if defined(__gfx94plus_local__) && __HIP_DEVICE_COMPILE__
  __hip_fp8_e4m3_fnuz tmp(y);
  x = tmp;
#else
  x = (y == 0);
#endif
  return x;
}

__FP8_DEVICE__ bool e5m2_bool_fnuz_device(float val) {
  bool x = false;
  float y = val;
#if defined(__gfx94plus_local__) && __HIP_DEVICE_COMPILE__
  __hip_fp8_e5m2_fnuz tmp(y);
  x = tmp;
#else
  x = (y == 0);
#endif
  return x;
}

template <bool is_e4m3_fnuz> __global__ void fp8_2_bool_fnuz(float* f, bool* ret, size_t size) {
  int i = threadIdx.x;
  bool r = false;
  if (i < size) {
    if constexpr (is_e4m3_fnuz) {
      r = e4m3_bool_fnuz_device(f[i]);
    } else {
      r = e5m2_bool_fnuz_device(f[i]);
    }
    ret[i] = r;
  }
}

TEST_CASE("Unit_fp8_fnuz_bool_device") {
  FP8_FNUZ_SKIP_TEST
  // clang-format off
  std::vector<float> fvals{-10.0f, -1.0f, -0.0f,  0.0f, 1.0f, 10.0f};
  std::vector<bool> tvals   {true,  true, false, false, true,  true};
  // clang-format on

  bool result[] = {false, false, false,
                   false, false, false};  // cant use std::vector coz data() = delete

  SECTION("e4m3_fnuz-gpu") {
    float* d_in{nullptr};
    bool* d_res{nullptr};
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * tvals.size()));
    HIP_CHECK(hipMalloc(&d_res, sizeof(bool) * tvals.size()));

    auto kernel = fp8_2_bool_fnuz<true>;
    HIP_CHECK(hipMemcpy(d_in, fvals.data(), sizeof(float) * fvals.size(), hipMemcpyHostToDevice));
    kernel<<<1, tvals.size()>>>(d_in, d_res, tvals.size());

    HIP_CHECK(hipMemcpy(result, d_res, sizeof(bool) * tvals.size(), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_res));
  }

  SECTION("e5m2_fnuz-gpu") {
    float* d_in{nullptr};
    bool* d_res{nullptr};
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * tvals.size()));
    HIP_CHECK(hipMalloc(&d_res, sizeof(bool) * tvals.size()));

    HIP_CHECK(hipMemcpy(d_in, fvals.data(), sizeof(float) * fvals.size(), hipMemcpyHostToDevice));
    auto kernel = fp8_2_bool_fnuz<false>;
    kernel<<<1, tvals.size()>>>(d_in, d_res, tvals.size());

    HIP_CHECK(hipMemcpy(result, d_res, sizeof(bool) * tvals.size(), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_res));
  }

  for (size_t i = 0; i < tvals.size(); i++) {
    INFO("Check for: " << fvals[i] << " expected: " << tvals[i] << " result: " << result[i]);
    REQUIRE(result[i] == tvals[i]);
  }
}

std::vector<__hip_fp8_storage_t> get_all_fp8_fnuz_nums() {
  std::vector<__hip_fp8_storage_t> ret;
  constexpr unsigned short max_fp8_num = 0b1111'1111;
  ret.reserve(max_fp8_num + 1 /* 0 */);

  for (unsigned short i = 0; i <= max_fp8_num; i++) {
    if ((i | 0x80) != 0x80) {
      ret.push_back(static_cast<__hip_fp8_storage_t>(i));
    }
  }
  return ret;
}

__FP8_DEVICE__ __hip_fp8_storage_t e4m3_fnuz_fp8_device(float val) {
  __hip_fp8_storage_t x = 0;
  float y = val;
#if defined(__gfx94plus_local__) && __HIP_DEVICE_COMPILE__
  __hip_fp8_e4m3_fnuz tmp(y);
  x = tmp.__x;
#else
  x = (y == 0) ? 0x0 : 0x80;
#endif
  return x;
}

__FP8_DEVICE__ __hip_fp8_storage_t e5m2_fnuz_fp8_device(float val) {
  __hip_fp8_storage_t x = 0;
  float y = val;
#if defined(__gfx94plus_local__) && __HIP_DEVICE_COMPILE__
  __hip_fp8_e5m2_fnuz tmp(y);
  x = tmp.__x;
#else
  x = (y == 0) ? 0x0 : 0x80;
#endif
  return x;
}

template <bool is_e4m3_fnuz>
__global__ void Type_to_fp8_fnuz(float* f, __hip_fp8_storage_t* res, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    if constexpr (is_e4m3_fnuz) {
      res[i] = e4m3_fnuz_fp8_device(f[i]);
    } else {
      res[i] = e5m2_fnuz_fp8_device(f[i]);
    }
  }
}

TEST_CASE("Unit_all_fp8_fnuz_cvt") {
  FP8_FNUZ_SKIP_TEST
  bool is_e4m3_fnuz = GENERATE(true, false);
  std::vector<float> f_vals;
  std::vector<__hip_fp8_storage_t> all_vals;

  SECTION("all representable number") {
    all_vals = get_all_fp8_fnuz_nums();
    f_vals.reserve(all_vals.size());

    for (const auto& fp8 : all_vals) {
      float f = 0.0f;
      if (is_e4m3_fnuz) {
        __hip_fp8_e4m3_fnuz tmp;
        tmp.__x = fp8;
        f = tmp;
      } else {
        __hip_fp8_e5m2_fnuz tmp;
        tmp.__x = fp8;
        f = tmp;
      }
      f_vals.push_back(f);
    }
  }

  SECTION("Range stepped numbers") {
    constexpr float lhs = -200.0f;
    constexpr float rhs = 200.0f;
    constexpr float step = 0.1234f;

    f_vals.reserve(4000);
    all_vals.reserve(4000);

    for (float fval = lhs; fval <= rhs; fval += step) {
      if (is_e4m3_fnuz) {
        __hip_fp8_e4m3_fnuz tmp = fval;
        all_vals.push_back(tmp.__x);
      } else {
        __hip_fp8_e5m2_fnuz tmp = fval;
        all_vals.push_back(tmp.__x);
      }
      f_vals.push_back(fval);
    }
  }

  float* d_f_vals;
  __hip_fp8_storage_t* d_res;

  HIP_CHECK(hipMalloc(&d_f_vals, sizeof(float) * f_vals.size()));
  HIP_CHECK(hipMalloc(&d_res, sizeof(__hip_fp8_storage_t) * f_vals.size()));

  HIP_CHECK(
      hipMemcpy(d_f_vals, f_vals.data(), sizeof(float) * f_vals.size(), hipMemcpyHostToDevice));

  auto fp8_kernel = is_e4m3_fnuz ? Type_to_fp8_fnuz<true> : Type_to_fp8_fnuz<false>;
  fp8_kernel<<<(f_vals.size() / 256) + 1, 256>>>(d_f_vals, d_res, f_vals.size());

  std::vector<__hip_fp8_storage_t> final_res(f_vals.size(), static_cast<__hip_fp8_storage_t>(0));

  HIP_CHECK(hipMemcpy(final_res.data(), d_res, sizeof(__hip_fp8_storage_t) * final_res.size(),
                      hipMemcpyDeviceToHost));

  for (size_t i = 0; i < final_res.size(); i++) {
    INFO("Checking: " << f_vals[i] << " for: " << (is_e4m3_fnuz ? "e4m3_fnuz" : "e5m2_fnuz")
                      << " original: " << (int)all_vals[i]
                      << " convert back: " << (int)final_res[i]);
    float gpu_cvt_res = 0.0f, cpu_cvt_res = 0.0f;
    if (is_e4m3_fnuz) {
      __hip_fp8_e4m3_fnuz gtmp;
      gtmp.__x = final_res[i];
      gpu_cvt_res = gtmp;
      __hip_fp8_e4m3_fnuz ctmp;
      ctmp.__x = all_vals[i];
      cpu_cvt_res = ctmp;
    } else {
      __hip_fp8_e5m2_fnuz gtmp;
      gtmp.__x = final_res[i];
      gpu_cvt_res = gtmp;
      __hip_fp8_e5m2_fnuz ctmp;
      ctmp.__x = all_vals[i];
      cpu_cvt_res = ctmp;
    }

    INFO("cpu cvt val: " << cpu_cvt_res << " gpu cvt val: " << gpu_cvt_res);
    REQUIRE(cpu_cvt_res == gpu_cvt_res);
  }

  HIP_CHECK(hipFree(d_f_vals));
  HIP_CHECK(hipFree(d_res));
}

template <typename T> __FP8_DEVICE__ void e4m3_fnuz_fp8_cvt(T val, float* cvt1, float* cvt2) {
  T y = val;
#if defined(__gfx94plus_local__) && __HIP_DEVICE_COMPILE__
  __hip_fp8_e4m3_fnuz tmp(y);
  *cvt1 = tmp;

  __hip_fp8_e4m3_fnuz tmp1;
  tmp1.__x = std::is_same<T, float>::value
                 ? __hip_cvt_float_to_fp8(val, __HIP_SATFINITE, __HIP_E4M3_FNUZ)
                 : __hip_cvt_double_to_fp8(val, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
  ;
  *cvt2 = tmp1;
#else
  *cvt1 = (y == 0) ? 0 : y;
  *cvt2 = (y == 0) ? 0 : y;
#endif
}

template <typename T> __FP8_DEVICE__ void e5m2_fnuz_fp8_cvt(T val, float* cvt1, float* cvt2) {
  T y = val;
#if defined(__gfx94plus_local__) && __HIP_DEVICE_COMPILE__
  __hip_fp8_e5m2_fnuz tmp(y);
  *cvt1 = tmp;

  __hip_fp8_e5m2_fnuz tmp1;
  tmp1.__x = std::is_same<T, float>::value
                 ? __hip_cvt_float_to_fp8(val, __HIP_SATFINITE, __HIP_E5M2_FNUZ)
                 : __hip_cvt_double_to_fp8(val, __HIP_SATFINITE, __HIP_E5M2_FNUZ);
  ;
  *cvt2 = tmp1;
#else
  *cvt1 = (y == 0) ? 0 : y;
  *cvt2 = (y == 0) ? 0 : y;
#endif
}


template <typename T, bool is_e4m3_fnuz>
__global__ void Type_to_fp8_fnuz_cvt(T* f, float* cvt1, float* cvt2, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    if constexpr (is_e4m3_fnuz) {
      e4m3_fnuz_fp8_cvt(f[i], &cvt1[i], &cvt2[i]);
    } else {
      e5m2_fnuz_fp8_cvt(f[i], &cvt1[i], &cvt2[i]);
    }
  }
}

TEMPLATE_TEST_CASE("Unit_fp8_fnuz_correctness_device", "", float, double) {
  FP8_FNUZ_SKIP_TEST

  SECTION("e4m3_fnuz") {
    /* These are basically all the fp8 - e4m3_fnuz type numbers.
     * They can be generated by iterating over 0'0000'000 and converting them to fp32 number
     * skipping the nan/inf */
    std::vector<TestType> e4m3_fnuz_nums = {0,           0.000976562, 0.00195312,
                                            0.00292969,  0.00390625,  0.00488281,
                                            0.00585938,  0.00683594,  0.0078125,
                                            0.00878906,  0.00976562,  0.0107422,
                                            0.0117188,   0.0126953,   0.0136719,
                                            0.0146484,   0.015625,    0.0175781,
                                            0.0195312,   0.0214844,   0.0234375,
                                            0.0253906,   0.0273438,   0.0292969,
                                            0.03125,     0.0351562,   0.0390625,
                                            0.0429688,   0.046875,    0.0507812,
                                            0.0546875,   0.0585938,   0.0625,
                                            0.0703125,   0.078125,    0.0859375,
                                            0.09375,     0.101562,    0.109375,
                                            0.117188,    0.125,       0.140625,
                                            0.15625,     0.171875,    0.1875,
                                            0.203125,    0.21875,     0.234375,
                                            0.25,        0.28125,     0.3125,
                                            0.34375,     0.375,       0.40625,
                                            0.4375,      0.46875,     0.5,
                                            0.5625,      0.625,       0.6875,
                                            0.75,        0.8125,      0.875,
                                            0.9375,      1,           1.125,
                                            1.25,        1.375,       1.5,
                                            1.625,       1.75,        1.875,
                                            2,           2.25,        2.5,
                                            2.75,        3,           3.25,
                                            3.5,         3.75,        4,
                                            4.5,         5,           5.5,
                                            6,           6.5,         7,
                                            7.5,         8,           9,
                                            10,          11,          12,
                                            13,          14,          15,
                                            16,          18,          20,
                                            22,          24,          26,
                                            28,          30,          32,
                                            36,          40,          44,
                                            48,          52,          56,
                                            60,          64,          72,
                                            80,          88,          96,
                                            104,         112,         120,
                                            128,         144,         160,
                                            176,         192,         208,
                                            224,         240,         -0.000976562,
                                            -0.00195312, -0.00292969, -0.00390625,
                                            -0.00488281, -0.00585938, -0.00683594,
                                            -0.0078125,  -0.00878906, -0.00976562,
                                            -0.0107422,  -0.0117188,  -0.0126953,
                                            -0.0136719,  -0.0146484,  -0.015625,
                                            -0.0175781,  -0.0195312,  -0.0214844,
                                            -0.0234375,  -0.0253906,  -0.0273438,
                                            -0.0292969,  -0.03125,    -0.0351562,
                                            -0.0390625,  -0.0429688,  -0.046875,
                                            -0.0507812,  -0.0546875,  -0.0585938,
                                            -0.0625,     -0.0703125,  -0.078125,
                                            -0.0859375,  -0.09375,    -0.101562,
                                            -0.109375,   -0.117188,   -0.125,
                                            -0.140625,   -0.15625,    -0.171875,
                                            -0.1875,     -0.203125,   -0.21875,
                                            -0.234375,   -0.25,       -0.28125,
                                            -0.3125,     -0.34375,    -0.375,
                                            -0.40625,    -0.4375,     -0.46875,
                                            -0.5,        -0.5625,     -0.625,
                                            -0.6875,     -0.75,       -0.8125,
                                            -0.875,      -0.9375,     -1,
                                            -1.125,      -1.25,       -1.375,
                                            -1.5,        -1.625,      -1.75,
                                            -1.875,      -2,          -2.25,
                                            -2.5,        -2.75,       -3,
                                            -3.25,       -3.5,        -3.75,
                                            -4,          -4.5,        -5,
                                            -5.5,        -6,          -6.5,
                                            -7,          -7.5,        -8,
                                            -9,          -10,         -11,
                                            -12,         -13,         -14,
                                            -15,         -16,         -18,
                                            -20,         -22,         -24,
                                            -26,         -28,         -30,
                                            -32,         -36,         -40,
                                            -44,         -48,         -52,
                                            -56,         -60,         -64,
                                            -72,         -80,         -88,
                                            -96,         -104,        -112,
                                            -120,        -128,        -144,
                                            -160,        -176,        -192,
                                            -208,        -224,        -240};

    size_t totalnums = e4m3_fnuz_nums.size();
    TestType* fnums;
    HIP_CHECK(hipMalloc((void**)&fnums, totalnums * sizeof(TestType)));
    float* cvt1_dev;
    HIP_CHECK(hipMalloc((void**)&cvt1_dev, totalnums * sizeof(TestType)));
    float* cvt2_dev;
    HIP_CHECK(hipMalloc((void**)&cvt2_dev, totalnums * sizeof(TestType)));

    HIP_CHECK(hipMemcpy(fnums, e4m3_fnuz_nums.data(), totalnums * sizeof(TestType),
                        hipMemcpyHostToDevice));

    auto fp8_kernel = Type_to_fp8_fnuz_cvt<TestType, true>;
    fp8_kernel<<<totalnums / 256 + 1, 256>>>(fnums, cvt1_dev, cvt2_dev, totalnums);

    float* cvt1_host = (float*)malloc(sizeof(float) * totalnums);
    float* cvt2_host = (float*)malloc(sizeof(float) * totalnums);

    HIP_CHECK(hipMemcpy(cvt1_host, cvt1_dev, totalnums * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(cvt2_host, cvt2_dev, totalnums * sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipDeviceSynchronize());

    for (size_t idx = 0; idx < totalnums; idx++) {
      TestType orig = e4m3_fnuz_nums[idx];
      float cvt1 = cvt1_host[idx];
      float cvt2 = cvt2_host[idx];

      INFO("Original: " << std::bitset<32>(*reinterpret_cast<const unsigned int*>(&orig)));
      INFO("Cvt back: " << std::bitset<32>(*reinterpret_cast<const unsigned int*>(&cvt1)));
      REQUIRE(cvt1 == Approx(orig));
      REQUIRE(cvt2 == cvt1);
    }

    HIP_CHECK(hipFree(fnums));
    HIP_CHECK(hipFree(cvt1_dev));
    HIP_CHECK(hipFree(cvt2_dev));
    free(cvt1_host);
    free(cvt2_host);
  }

  SECTION("e5m2_fnuz") {
    /* These are basically all the fp8 - e5m2_fnuz type numbers.
     * They can be generated by iterating over 0'00000'00 converting them to fp32 number skipping
     * the nan/inf */
    std::vector<TestType> e5m2_fnuz_nums = {0,
                                            7.62939e-06,
                                            1.52588e-05,
                                            2.28882e-05,
                                            3.05176e-05,
                                            3.8147e-05,
                                            4.57764e-05,
                                            5.34058e-05,
                                            6.10352e-05,
                                            7.62939e-05,
                                            9.15527e-05,
                                            0.000106812,
                                            0.00012207,
                                            0.000152588,
                                            0.000183105,
                                            0.000213623,
                                            0.000244141,
                                            0.000305176,
                                            0.000366211,
                                            0.000427246,
                                            0.000488281,
                                            0.000610352,
                                            0.000732422,
                                            0.000854492,
                                            0.000976562,
                                            0.0012207,
                                            0.00146484,
                                            0.00170898,
                                            0.00195312,
                                            0.00244141,
                                            0.00292969,
                                            0.00341797,
                                            0.00390625,
                                            0.00488281,
                                            0.00585938,
                                            0.00683594,
                                            0.0078125,
                                            0.00976562,
                                            0.0117188,
                                            0.0136719,
                                            0.015625,
                                            0.0195312,
                                            0.0234375,
                                            0.0273438,
                                            0.03125,
                                            0.0390625,
                                            0.046875,
                                            0.0546875,
                                            0.0625,
                                            0.078125,
                                            0.09375,
                                            0.109375,
                                            0.125,
                                            0.15625,
                                            0.1875,
                                            0.21875,
                                            0.25,
                                            0.3125,
                                            0.375,
                                            0.4375,
                                            0.5,
                                            0.625,
                                            0.75,
                                            0.875,
                                            1,
                                            1.25,
                                            1.5,
                                            1.75,
                                            2,
                                            2.5,
                                            3,
                                            3.5,
                                            4,
                                            5,
                                            6,
                                            7,
                                            8,
                                            10,
                                            12,
                                            14,
                                            16,
                                            20,
                                            24,
                                            28,
                                            32,
                                            40,
                                            48,
                                            56,
                                            64,
                                            80,
                                            96,
                                            112,
                                            128,
                                            160,
                                            192,
                                            224,
                                            256,
                                            320,
                                            384,
                                            448,
                                            512,
                                            640,
                                            768,
                                            896,
                                            1024,
                                            1280,
                                            1536,
                                            1792,
                                            2048,
                                            2560,
                                            3072,
                                            3584,
                                            4096,
                                            5120,
                                            6144,
                                            7168,
                                            8192,
                                            10240,
                                            12288,
                                            14336,
                                            16384,
                                            20480,
                                            24576,
                                            28672,
                                            32768,
                                            40960,
                                            49152,
                                            57344,
                                            -7.62939e-06,
                                            -1.52588e-05,
                                            -2.28882e-05,
                                            -3.05176e-05,
                                            -3.8147e-05,
                                            -4.57764e-05,
                                            -5.34058e-05,
                                            -6.10352e-05,
                                            -7.62939e-05,
                                            -9.15527e-05,
                                            -0.000106812,
                                            -0.00012207,
                                            -0.000152588,
                                            -0.000183105,
                                            -0.000213623,
                                            -0.000244141,
                                            -0.000305176,
                                            -0.000366211,
                                            -0.000427246,
                                            -0.000488281,
                                            -0.000610352,
                                            -0.000732422,
                                            -0.000854492,
                                            -0.000976562,
                                            -0.0012207,
                                            -0.00146484,
                                            -0.00170898,
                                            -0.00195312,
                                            -0.00244141,
                                            -0.00292969,
                                            -0.00341797,
                                            -0.00390625,
                                            -0.00488281,
                                            -0.00585938,
                                            -0.00683594,
                                            -0.0078125,
                                            -0.00976562,
                                            -0.0117188,
                                            -0.0136719,
                                            -0.015625,
                                            -0.0195312,
                                            -0.0234375,
                                            -0.0273438,
                                            -0.03125,
                                            -0.0390625,
                                            -0.046875,
                                            -0.0546875,
                                            -0.0625,
                                            -0.078125,
                                            -0.09375,
                                            -0.109375,
                                            -0.125,
                                            -0.15625,
                                            -0.1875,
                                            -0.21875,
                                            -0.25,
                                            -0.3125,
                                            -0.375,
                                            -0.4375,
                                            -0.5,
                                            -0.625,
                                            -0.75,
                                            -0.875,
                                            -1,
                                            -1.25,
                                            -1.5,
                                            -1.75,
                                            -2,
                                            -2.5,
                                            -3,
                                            -3.5,
                                            -4,
                                            -5,
                                            -6,
                                            -7,
                                            -8,
                                            -10,
                                            -12,
                                            -14,
                                            -16,
                                            -20,
                                            -24,
                                            -28,
                                            -32,
                                            -40,
                                            -48,
                                            -56,
                                            -64,
                                            -80,
                                            -96,
                                            -112,
                                            -128,
                                            -160,
                                            -192,
                                            -224,
                                            -256,
                                            -320,
                                            -384,
                                            -448,
                                            -512,
                                            -640,
                                            -768,
                                            -896,
                                            -1024,
                                            -1280,
                                            -1536,
                                            -1792,
                                            -2048,
                                            -2560,
                                            -3072,
                                            -3584,
                                            -4096,
                                            -5120,
                                            -6144,
                                            -7168,
                                            -8192,
                                            -10240,
                                            -12288,
                                            -14336,
                                            -16384,
                                            -20480,
                                            -24576,
                                            -28672,
                                            -32768,
                                            -40960,
                                            -49152,
                                            -57344};

    size_t totalnums = e5m2_fnuz_nums.size();
    TestType* fnums;
    HIP_CHECK(hipMalloc((void**)&fnums, totalnums * sizeof(TestType)));
    float* cvt1_dev;
    HIP_CHECK(hipMalloc((void**)&cvt1_dev, totalnums * sizeof(TestType)));
    float* cvt2_dev;
    HIP_CHECK(hipMalloc((void**)&cvt2_dev, totalnums * sizeof(TestType)));

    HIP_CHECK(hipMemcpy(fnums, e5m2_fnuz_nums.data(), totalnums * sizeof(TestType),
                        hipMemcpyHostToDevice));

    auto fp8_kernel = Type_to_fp8_fnuz_cvt<TestType, false>;
    fp8_kernel<<<totalnums / 256 + 1, 256>>>(fnums, cvt1_dev, cvt2_dev, totalnums);

    float* cvt1_host = (float*)malloc(sizeof(float) * totalnums);
    float* cvt2_host = (float*)malloc(sizeof(float) * totalnums);

    HIP_CHECK(hipMemcpy(cvt1_host, cvt1_dev, totalnums * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(cvt2_host, cvt2_dev, totalnums * sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipDeviceSynchronize());

    for (size_t idx = 0; idx < totalnums; idx++) {
      TestType orig = e5m2_fnuz_nums[idx];
      float cvt1 = cvt1_host[idx];
      float cvt2 = cvt2_host[idx];

      INFO("Original: " << std::bitset<32>(*reinterpret_cast<const unsigned int*>(&orig)));
      INFO("Cvt back: " << std::bitset<32>(*reinterpret_cast<const unsigned int*>(&cvt1)));
      REQUIRE(cvt1 == Approx(orig));
      REQUIRE(cvt2 == cvt1);
    }

    HIP_CHECK(hipFree(fnums));
    HIP_CHECK(hipFree(cvt1_dev));
    HIP_CHECK(hipFree(cvt2_dev));
    free(cvt1_host);
    free(cvt2_host);
  }
}
