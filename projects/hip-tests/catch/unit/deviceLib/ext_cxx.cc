/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip/hip_ext_ocp.h>
#include <hip/hip_fp16.h>
#include <hip_test_common.hh>

#include <iostream>
#include <vector>


// CXX

__global__ void cxx_fp8_e4m3_device_cvt(float* in, float* out, size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    __hipext_ocp_fp8_e4m3 tmp(in[i]);
    out[i] = tmp;
  }
}

__global__ void cxx_fp8_sr_e4m3_device_cvt(float* in, float* out, unsigned int seed, size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    __hipext_ocp_fp8_e4m3 tmp(in[i], seed);
    out[i] = tmp;
  }
}

__global__ void cxx_fp8_e5m2_device_cvt(float* in, float* out, size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    __hipext_ocp_fp8_e5m2 tmp(in[i]);
    out[i] = tmp;
  }
}

__global__ void cxx_fp8_sr_e5m2_device_cvt(float* in, float* out, unsigned int seed, size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    __hipext_ocp_fp8_e5m2 tmp(in[i], seed);
    out[i] = tmp;
  }
}

__global__ void cxx_fp8x2_e4m3_device_cvt(__amd_floatx2_storage_t* in, __amd_floatx2_storage_t* out,
                                          size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    __hipext_ocp_fp8x2_e4m3 tmp(in[i]);
    out[i] = tmp;
  }
}

__global__ void cxx_fp8x2_e5m2_device_cvt(__amd_floatx2_storage_t* in, __amd_floatx2_storage_t* out,
                                          size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    __hipext_ocp_fp8x2_e5m2 tmp(in[i]);
    out[i] = tmp;
  }
}

TEST_CASE("Unit_ocp_cxx_fp8_host_conv") {
  SECTION("e4m3") {
    constexpr size_t size = 449 * 2 + 1;
    std::vector<float> in;
    in.reserve(size);
    for (int i = -449; i <= 449; i++) {
      in.push_back(static_cast<float>(i));
    }
    REQUIRE(in.size() == size);
    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float) * size));
    HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(float) * size, hipMemcpyHostToDevice));
    cxx_fp8_e4m3_device_cvt<<<1, size>>>(d_in, d_out, size);
    // CPU calc
    std::vector<float> cpu_res(size, 0.0f);
    for (size_t i = 0; i < size; i++) {
      __hipext_ocp_fp8_e4m3 tmp(in[i]);
      cpu_res[i] = tmp;
    }
    std::vector<float> gpu_res(size, 0.0f);
    HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << " in: " << in[i] << " cpu: " << cpu_res[i] << " gpu: " << gpu_res[i]);
      REQUIRE(cpu_res[i] == gpu_res[i]);
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("e4m3-sr") {
    constexpr size_t size = 449 * 2 + 1;
    constexpr unsigned int seed = 10;
    std::vector<float> in;
    in.reserve(size);
    for (int i = -449; i <= 449; i++) {
      in.push_back(static_cast<float>(i));
    }
    REQUIRE(in.size() == size);
    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float) * size));
    HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(float) * size, hipMemcpyHostToDevice));
    cxx_fp8_sr_e4m3_device_cvt<<<1, size>>>(d_in, d_out, seed, size);
    // CPU calc
    std::vector<float> cpu_res(size, 0.0f);
    for (size_t i = 0; i < size; i++) {
      __hipext_ocp_fp8_e4m3 tmp(in[i], seed);
      cpu_res[i] = tmp;
    }
    std::vector<float> gpu_res(size, 0.0f);
    HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << " in: " << in[i] << " cpu: " << cpu_res[i] << " gpu: " << gpu_res[i]);
      REQUIRE(cpu_res[i] == gpu_res[i]);
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("e5m2") {
    constexpr size_t size = 511 * 2 + 1;
    std::vector<float> in;
    in.reserve(size);
    for (int i = -511; i <= 511; i++) {
      in.push_back(static_cast<float>(i));
    }
    REQUIRE(in.size() == size);
    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float) * size));
    HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(float) * size, hipMemcpyHostToDevice));
    cxx_fp8_e5m2_device_cvt<<<1, size>>>(d_in, d_out, size);
    // CPU calc
    std::vector<float> cpu_res(size, 0.0f);
    for (size_t i = 0; i < size; i++) {
      __hipext_ocp_fp8_e5m2 tmp(in[i]);
      cpu_res[i] = tmp;
    }
    std::vector<float> gpu_res(size, 0.0f);
    HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << " in: " << in[i] << " cpu: " << cpu_res[i] << " gpu: " << gpu_res[i]);
      CHECK(cpu_res[i] == gpu_res[i]);
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("e5m2-sr") {
    constexpr size_t size = 511 * 2 + 1;
    constexpr unsigned int seed = 10;
    std::vector<float> in;
    in.reserve(size);
    for (int i = -511; i <= 511; i++) {
      in.push_back(static_cast<float>(i));
    }
    REQUIRE(in.size() == size);
    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float) * size));
    HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(float) * size, hipMemcpyHostToDevice));
    cxx_fp8_sr_e5m2_device_cvt<<<1, size>>>(d_in, d_out, seed, size);
    // CPU calc
    std::vector<float> cpu_res(size, 0.0f);
    for (size_t i = 0; i < size; i++) {
      __hipext_ocp_fp8_e5m2 tmp(in[i], seed);
      cpu_res[i] = tmp;
    }
    std::vector<float> gpu_res(size, 0.0f);
    HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << " in: " << in[i] << " cpu: " << cpu_res[i] << " gpu: " << gpu_res[i]);
      CHECK(cpu_res[i] == gpu_res[i]);
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("e4m3x2") {
    constexpr size_t size = 448 * 2 + 1;
    std::vector<__amd_floatx2_storage_t> in;
    in.reserve(size);
    for (int i = -448, j = 448; i <= 448; i++, j--) {
      __amd_floatx2_storage_t tmp{static_cast<float>(i), static_cast<float>(j)};
      in.push_back(tmp);
    }
    REQUIRE(in.size() == size);
    __amd_floatx2_storage_t *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(
        hipMemcpy(d_in, in.data(), sizeof(__amd_floatx2_storage_t) * size, hipMemcpyHostToDevice));
    cxx_fp8x2_e4m3_device_cvt<<<1, size>>>(d_in, d_out, size);
    // CPU calc
    std::vector<__amd_floatx2_storage_t> cpu_res(size, 0.0f);
    for (size_t i = 0; i < size; i++) {
      __hipext_ocp_fp8x2_e4m3 tmp(in[i]);
      cpu_res[i] = tmp;
    }
    std::vector<__amd_floatx2_storage_t> gpu_res(size, 0.0f);
    HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_floatx2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << "\n\tin:  a: " << in[i][0] << " b: " << in[i][1]
                     << "\n\tcpu: a: " << cpu_res[i][0] << " b: " << cpu_res[i][1]
                     << "\n\tgpu: a: " << gpu_res[i][0] << " b: " << gpu_res[i][0]);
      REQUIRE(cpu_res[i][0] == gpu_res[i][0]);
      REQUIRE(cpu_res[i][1] == gpu_res[i][1]);
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("e5m2x2") {
    constexpr size_t size = 511 * 2 + 1;
    std::vector<__amd_floatx2_storage_t> in;
    in.reserve(size);
    for (int i = -511, j = 511; i <= 511; i++, j--) {
      __amd_floatx2_storage_t tmp{static_cast<float>(i), static_cast<float>(j)};
      in.push_back(tmp);
    }
    REQUIRE(in.size() == size);
    __amd_floatx2_storage_t *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx2_storage_t) * size));
    HIP_CHECK(
        hipMemcpy(d_in, in.data(), sizeof(__amd_floatx2_storage_t) * size, hipMemcpyHostToDevice));
    cxx_fp8x2_e5m2_device_cvt<<<1, size>>>(d_in, d_out, size);
    // CPU calc
    std::vector<__amd_floatx2_storage_t> cpu_res(size, 0.0f);
    for (size_t i = 0; i < size; i++) {
      __hipext_ocp_fp8x2_e5m2 tmp(in[i]);
      cpu_res[i] = tmp;
    }
    std::vector<__amd_floatx2_storage_t> gpu_res(size, 0.0f);
    HIP_CHECK(hipMemcpy(gpu_res.data(), d_out, sizeof(__amd_floatx2_storage_t) * size,
                        hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      INFO("Index: " << i << "\n\tin:  a: " << in[i][0] << " b: " << in[i][1]
                     << "\n\tcpu: a: " << cpu_res[i][0] << " b: " << cpu_res[i][1]
                     << "\n\tgpu: a: " << gpu_res[i][0] << " b: " << gpu_res[i][0]);
      REQUIRE(cpu_res[i][0] == gpu_res[i][0]);
      REQUIRE(cpu_res[i][1] == gpu_res[i][1]);
    }
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }
}

__global__ void fp8x2_e4m3_cxx_convert_fp32(__amd_floatx2_storage_t* in,
                                            __amd_floatx2_storage_t* out, size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    __hipext_ocp_fp8x2_e4m3 tmp(in[i]);
    out[i] = tmp;
  }
}

__global__ void fp8x2_e4m3_cxx_convert_fp16(__amd_fp16x2_storage_t* in, __amd_fp16x2_storage_t* out,
                                            size_t size, __amd_fp8x2_storage_t* t_out = nullptr) {
  int i = threadIdx.x;
  if (i < size) {
    __hipext_ocp_fp8x2_e4m3 tmp(in[i], 0);
    if (t_out != nullptr) {
      t_out[i] = tmp.__x;
    }
    out[i] = tmp.get_scaled_fp16x2(0);
  }
}

__global__ void fp8x2_e4m3_cxx_convert_bf16(__amd_bf16x2_storage_t* in, __amd_bf16x2_storage_t* out,
                                            size_t size, __amd_fp8x2_storage_t* t_out = nullptr) {
  int i = threadIdx.x;
  if (i < size) {
    __hipext_ocp_fp8x2_e4m3 tmp(in[i], 0);
    if (t_out != nullptr) {
      t_out[i] = tmp.__x;
    }
    out[i] = tmp.get_scaled_bf16x2(0);
  }
}

TEST_CASE("Unit_ocp_fp8x2_host_device") {
  SECTION("floatx2 to e4m3") {
    __amd_floatx2_storage_t in{-10.0f, 10.0f}, *d_in, *d_out, out{0.0f, 0.0f};
    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx2_storage_t)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx2_storage_t)));
    HIP_CHECK(hipMemcpy(d_in, &in, sizeof(__amd_floatx2_storage_t), hipMemcpyHostToDevice));
    fp8x2_e4m3_cxx_convert_fp32<<<1, 32>>>(d_in, d_out, 1);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_floatx2_storage_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    __hipext_ocp_fp8x2_e4m3 tmp(in);
    __amd_floatx2_storage_t cpu_out = tmp;
    INFO("    In : " << in[0] << ", " << in[1]);
    INFO("GPU Out: " << out[0] << ", " << out[1]);
    INFO("CPU Out: " << cpu_out[0] << ", " << cpu_out[1]);
    REQUIRE(out[0] == cpu_out[0]);
    REQUIRE(out[1] == cpu_out[1]);
  }

  SECTION("fp16x2 to e4m3") {
    __amd_fp16x2_storage_t in{-10.0f, 10.0f}, *d_in, *d_out, out{0.0f, 0.0f};
    __amd_fp8x2_storage_t *t_storage, gt_storage;
    HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_fp16x2_storage_t)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_fp16x2_storage_t)));
    HIP_CHECK(hipMalloc(&t_storage, sizeof(__amd_fp8x2_storage_t)));
    HIP_CHECK(hipMemcpy(d_in, &in, sizeof(__amd_fp16x2_storage_t), hipMemcpyHostToDevice));
    fp8x2_e4m3_cxx_convert_fp16<<<1, 32>>>(d_in, d_out, 1, t_storage);
    HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_fp16x2_storage_t), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(&gt_storage, t_storage, sizeof(__amd_fp8x2_storage_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    __hipext_ocp_fp8x2_e4m3 tmp(in, 0);
    __amd_fp16x2_storage_t cpu_out = tmp.get_scaled_fp16x2(0);
    INFO("    In : " << float(in[0]) << ", " << float(in[1]));
    INFO("GPU Out: " << float(out[0]) << ", " << float(out[1]));
    INFO("CPU Out: " << float(cpu_out[0]) << ", " << float(cpu_out[1]));
    INFO("gpu tmp: " << std::hex << unsigned(gt_storage) << ", cpu: " << unsigned(tmp.__x));
    REQUIRE(out[0] == cpu_out[0]);
    REQUIRE(out[1] == cpu_out[1]);
  }

  // SECTION("bf16x2 to e4m3") {
  //   __amd_bf16x2_storage_t in{-10.0f, 10.0f}, *d_in, *d_out, out{0.0f, 0.0f};
  //   HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_bf16x2_storage_t)));
  //   HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_bf16x2_storage_t)));
  //   HIP_CHECK(hipMemcpy(d_in, &in, sizeof(__amd_bf16x2_storage_t),
  //                       hipMemcpyHostToDevice));
  //   fp8x2_e4m3_cxx_convert_bf16<<<1, 32>>>(d_in, d_out, 1);
  //   HIP_CHECK(hipMemcpy(&out, d_out, sizeof(__amd_bf16x2_storage_t),
  //                       hipMemcpyDeviceToHost));
  //   HIP_CHECK(hipFree(d_in));
  //   HIP_CHECK(hipFree(d_out));
  //   __hipext_ocp_fp8x2_e4m3 tmp(in, 0);
  //   __amd_bf16x2_storage_t cpu_out = tmp.get_scaled_bf16x2(0);
  //   REQUIRE(out[0] == cpu_out[0]);
  //   REQUIRE(out[1] == cpu_out[1]);
  // }
}


namespace cxx_ocp {
__host__ __device__ static float fp8_e4m3_to_float(const float in) {
  return __hipext_ocp_fp8_e4m3{in};
}

__host__ __device__ static float fp8_e4m3_to_float_scale(const float in,
                                                         const __amd_scale_t scale) {
  return __hipext_ocp_fp8_e4m3(in, /* seed */ 0, scale).get_scaled_float(scale);
}

__host__ __device__ static __amd_fp16_storage_t fp8_e4m3_to_fp16(const __amd_fp16_storage_t in) {
  return __hipext_ocp_fp8_e4m3(in, /* seed */ 0, 0).get_scaled_fp16(0);
}

__host__ __device__ static __amd_fp16_storage_t fp8_e4m3_to_fp16_scale(
    const __amd_fp16_storage_t in, const __amd_scale_t scale) {
  return __hipext_ocp_fp8_e4m3(in, /* seed */ 0, scale).get_scaled_fp16(scale);
}

__host__ __device__ static __amd_bf16_storage_t fp8_e4m3_to_bf16(const __amd_bf16_storage_t in) {
  return __hipext_ocp_fp8_e4m3(in, /* seed */ 0, 0).get_scaled_bf16(0);
}

__host__ __device__ static __amd_bf16_storage_t fp8_e4m3_to_bf16_scale(
    const __amd_bf16_storage_t in, const __amd_scale_t scale) {
  return __hipext_ocp_fp8_e4m3(in, /* seed */ 0, scale).get_scaled_bf16(scale);
}

__global__ void kernel_fp8_e4m3_to_float(float* in, float* out, size_t size = 1) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = fp8_e4m3_to_float(in[i]);
  }
}

__global__ void kernel_fp8_e4m3_to_float_scale(float* in, float* out, const __amd_scale_t scale,
                                               const size_t size = 1) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = fp8_e4m3_to_float_scale(in[i], scale);
  }
}

__global__ void kernel_fp8_e4m3_to_fp16(__amd_fp16_storage_t* in, __amd_fp16_storage_t* out,
                                        size_t size = 1) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = fp8_e4m3_to_fp16(in[i]);
  }
}

__global__ void kernel_fp8_e4m3_to_fp16_scale(__amd_fp16_storage_t* in, __amd_fp16_storage_t* out,
                                              const __amd_scale_t scale, const size_t size = 1) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = fp8_e4m3_to_fp16_scale(in[i], scale);
  }
}

__global__ void kernel_fp8_e4m3_to_bf16(__amd_bf16_storage_t* in, __amd_bf16_storage_t* out,
                                        size_t size = 1) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = fp8_e4m3_to_bf16(in[i]);
  }
}

__global__ void kernel_fp8_e4m3_to_bf16_scale(__amd_bf16_storage_t* in, __amd_bf16_storage_t* out,
                                              const __amd_scale_t scale, const size_t size = 1) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = fp8_e4m3_to_bf16_scale(in[i], scale);
  }
}

__host__ __device__ static float fp8_e5m2_to_float(const float in) {
  return __hipext_ocp_fp8_e5m2{in};
}

__host__ __device__ static float fp8_e5m2_to_float_scale(const float in,
                                                         const __amd_scale_t scale) {
  return __hipext_ocp_fp8_e5m2(in, /* seed */ 0, scale).get_scaled_float(scale);
}

__host__ __device__ static __amd_fp16_storage_t fp8_e5m2_to_fp16(const __amd_fp16_storage_t in) {
  return __hipext_ocp_fp8_e5m2(in, /* seed */ 0, 0).get_scaled_fp16(0);
}

__host__ __device__ static __amd_fp16_storage_t fp8_e5m2_to_fp16_scale(
    const __amd_fp16_storage_t in, const __amd_scale_t scale) {
  return __hipext_ocp_fp8_e5m2(in, /* seed */ 0, scale).get_scaled_fp16(scale);
}

__global__ void kernel_fp8_e5m2_to_float(float* in, float* out, size_t size = 1) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = fp8_e5m2_to_float(in[i]);
  }
}

__global__ void kernel_fp8_e5m2_to_float_scale(float* in, float* out, const __amd_scale_t scale,
                                               const size_t size = 1) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = fp8_e5m2_to_float_scale(in[i], scale);
  }
}

__global__ void kernel_fp8_e5m2_to_fp16(__amd_fp16_storage_t* in, __amd_fp16_storage_t* out,
                                        size_t size = 1) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = fp8_e5m2_to_fp16(in[i]);
  }
}

__global__ void kernel_fp8_e5m2_to_fp16_scale(__amd_fp16_storage_t* in, __amd_fp16_storage_t* out,
                                              const __amd_scale_t scale, const size_t size = 1) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = fp8_e5m2_to_fp16_scale(in[i], scale);
  }
}
}  // namespace cxx_ocp

TEST_CASE("Unit_ocp_cxx_fp8") {
  using namespace cxx_ocp;
  const float in = 20.0f;
  float *d_in, *d_out;
  __amd_fp16_storage_t fp16_in = in, *fp16_d_in, *fp16_d_out;
  HIP_CHECK(hipMalloc(&d_in, sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, sizeof(float)));
  HIP_CHECK(hipMalloc(&fp16_d_in, sizeof(__amd_fp16_storage_t)));
  HIP_CHECK(hipMalloc(&fp16_d_out, sizeof(__amd_fp16_storage_t)));
  HIP_CHECK(hipMemcpy(d_in, &in, sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(fp16_d_in, &fp16_in, sizeof(__amd_fp16_storage_t), hipMemcpyHostToDevice));
  SECTION("E4M3 ") {
    SECTION("CPU") {
      float out = fp8_e4m3_to_float(in);
      INFO("CPU In: " << in << " Out: " << out);
      REQUIRE(in == out);
    }

    SECTION("GPU") {
      kernel_fp8_e4m3_to_float<<<1, 32>>>(d_in, d_out);
      float out;
      HIP_CHECK(hipMemcpy(&out, d_out, sizeof(float), hipMemcpyDeviceToHost));
      INFO("GPU In: " << in << " Out: " << out);
      REQUIRE(in == out);
    }

    SECTION("CPU Scale") {
      std::vector<__amd_scale_t> scales = {0, 1, 2, 3};
      for (const auto scale : scales) {
        float out = fp8_e4m3_to_float_scale(in, scale);
        INFO("Scale: " << int(scale));
        INFO("In: " << in << " Out: " << out);
        REQUIRE(in == out);
      }
    }

    SECTION("GPU Scale") {
      std::vector<__amd_scale_t> scales = {0, 1, 2, 3};
      for (const auto scale : scales) {
        kernel_fp8_e4m3_to_float_scale<<<1, 32>>>(d_in, d_out, scale);
        float out;
        HIP_CHECK(hipMemcpy(&out, d_out, sizeof(float), hipMemcpyDeviceToHost));
        INFO("Scale: " << int(scale));
        INFO("GPU In: " << in << " Out: " << out);
        CHECK(in == out);
      }
    }

    SECTION("CPU fp16") {
      __amd_fp16_storage_t out = fp8_e4m3_to_fp16(fp16_in);
      INFO("GPU In: " << in << " Out: " << float(out));
      REQUIRE(in == out);
    }

    SECTION("GPU fp16") {
      kernel_fp8_e4m3_to_fp16<<<1, 32>>>(fp16_d_in, fp16_d_out);
      __amd_fp16_storage_t out;
      HIP_CHECK(hipMemcpy(&out, fp16_d_out, sizeof(__amd_fp16_storage_t), hipMemcpyDeviceToHost));
      INFO("GPU In: " << in << " Out: " << float(out));
      REQUIRE(in == out);
    }

    SECTION("CPU fp16 Scale") {
      std::vector<__amd_scale_t> scales = {0, 1, 2, 3};
      for (const auto scale : scales) {
        __amd_fp16_storage_t out = fp8_e4m3_to_fp16_scale(in, scale);
        INFO("Scale: " << int(scale));
        INFO("In: " << in << " Out: " << float(out));
        REQUIRE(fp16_in == out);
      }
    }

    SECTION("GPU fp16 Scale") {
      std::vector<__amd_scale_t> scales = {0, 1, 2, 3};
      for (const auto scale : scales) {
        kernel_fp8_e4m3_to_fp16_scale<<<1, 32>>>(fp16_d_in, fp16_d_out, scale);
        __amd_fp16_storage_t out;
        HIP_CHECK(hipMemcpy(&out, fp16_d_out, sizeof(__amd_fp16_storage_t), hipMemcpyDeviceToHost));
        INFO("Scale: " << int(scale));
        INFO("GPU In: " << in << " Out: " << float(out));
        CHECK(in == out);
      }
    }
  }

  SECTION("E5M2 ") {
    SECTION("CPU") {
      float out = fp8_e5m2_to_float(in);
      INFO("CPU In: " << in << " Out: " << out);
      REQUIRE(in == out);
    }

    SECTION("GPU") {
      kernel_fp8_e5m2_to_float<<<1, 32>>>(d_in, d_out);
      float out;
      HIP_CHECK(hipMemcpy(&out, d_out, sizeof(float), hipMemcpyDeviceToHost));
      INFO("GPU In: " << in << " Out: " << out);
      REQUIRE(in == out);
    }

    SECTION("CPU Scale") {
      std::vector<__amd_scale_t> scales = {0, 1, 2, 3};
      for (const auto scale : scales) {
        float in = 20.0f;
        float out = fp8_e5m2_to_float_scale(in, scale);
        INFO("Scale: " << int(scale));
        INFO("In: " << in << " Out: " << out);
        REQUIRE(in == out);
      }
    }

    SECTION("GPU Scale") {
      std::vector<__amd_scale_t> scales = {0, 1, 2, 3};
      for (const auto scale : scales) {
        kernel_fp8_e5m2_to_float_scale<<<1, 32>>>(d_in, d_out, scale);
        float out;
        HIP_CHECK(hipMemcpy(&out, d_out, sizeof(float), hipMemcpyDeviceToHost));
        INFO("Scale: " << int(scale));
        INFO("GPU In: " << in << " Out: " << out);
        CHECK(in == out);
      }
    }

    SECTION("CPU fp16") {
      __amd_fp16_storage_t out = fp8_e5m2_to_fp16(fp16_in);
      INFO("GPU In: " << in << " Out: " << float(out));
      REQUIRE(in == out);
    }

    SECTION("GPU fp16") {
      kernel_fp8_e5m2_to_fp16<<<1, 32>>>(fp16_d_in, fp16_d_out);
      __amd_fp16_storage_t out;
      HIP_CHECK(hipMemcpy(&out, fp16_d_out, sizeof(__amd_fp16_storage_t), hipMemcpyDeviceToHost));
      INFO("GPU In: " << in << " Out: " << float(out));
      REQUIRE(in == out);
    }

    SECTION("CPU fp16 Scale") {
      std::vector<__amd_scale_t> scales = {0, 1, 2, 3};
      for (const auto scale : scales) {
        __amd_fp16_storage_t out = fp8_e5m2_to_fp16_scale(in, scale);
        INFO("Scale: " << int(scale));
        INFO("In: " << in << " Out: " << float(out));
        REQUIRE(in == out);
      }
    }

    SECTION("GPU fp16 Scale") {
      std::vector<__amd_scale_t> scales = {0, 1, 2, 3};
      for (const auto scale : scales) {
        kernel_fp8_e5m2_to_fp16_scale<<<1, 32>>>(fp16_d_in, fp16_d_out, scale);
        __amd_fp16_storage_t out;
        HIP_CHECK(hipMemcpy(&out, fp16_d_out, sizeof(__amd_fp16_storage_t), hipMemcpyDeviceToHost));
        INFO("Scale: " << int(scale));
        INFO("GPU In: " << in << " Out: " << float(out));
        CHECK(in == out);
      }
    }
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(fp16_d_in));
  HIP_CHECK(hipFree(fp16_d_out));
}

namespace cxx_ocp {
__host__ __device__ static __amd_floatx2_storage_t fp8x2_e4m3_to_float(
    const __amd_floatx2_storage_t in) {
  return __hipext_ocp_fp8x2_e4m3{in};
}

__host__ __device__ static __amd_floatx2_storage_t fp8x2_e4m3_to_float_scale(
    const __amd_floatx2_storage_t in, const __amd_scale_t scale) {
  return __hipext_ocp_fp8x2_e4m3(in, scale).get_scaled_floatx2(scale);
}

__host__ __device__ static __amd_fp16x2_storage_t fp8x2_e4m3_to_fp16_scale(
    const __amd_fp16x2_storage_t in, const __amd_scale_t scale) {
  return __hipext_ocp_fp8x2_e4m3(in, scale).get_scaled_fp16x2(scale);
}

__global__ void kernel_fp8x2_e4m3_to_float(__amd_floatx2_storage_t* in,
                                           __amd_floatx2_storage_t* out, const size_t size = 1) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = fp8x2_e4m3_to_float(in[i]);
  }
}

__global__ void kernel_fp8x2_e4m3_to_float_scale(__amd_floatx2_storage_t* in,
                                                 __amd_floatx2_storage_t* out,
                                                 const __amd_scale_t scale, const size_t size = 1) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = fp8x2_e4m3_to_float_scale(in[i], scale);
  }
}

__global__ void kernel_fp8x2_e4m3_to_fp16_scale(__amd_fp16x2_storage_t* in,
                                                __amd_fp16x2_storage_t* out,
                                                const __amd_scale_t scale, const size_t size = 1) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = fp8x2_e4m3_to_fp16_scale(in[i], scale);
  }
}

__host__ __device__ static __amd_floatx2_storage_t fp8x2_e5m2_to_float(
    const __amd_floatx2_storage_t in) {
  return __hipext_ocp_fp8x2_e5m2{in};
}

__host__ __device__ static __amd_floatx2_storage_t fp8x2_e5m2_to_float_scale(
    const __amd_floatx2_storage_t in, const __amd_scale_t scale) {
  return __hipext_ocp_fp8x2_e5m2(in, scale).get_scaled_floatx2(scale);
}

__host__ __device__ static __amd_fp16x2_storage_t fp8x2_e5m2_to_fp16_scale(
    const __amd_fp16x2_storage_t in, const __amd_scale_t scale) {
  return __hipext_ocp_fp8x2_e5m2(in, scale).get_scaled_fp16x2(scale);
}

__global__ void kernel_fp8x2_e5m2_to_float(__amd_floatx2_storage_t* in,
                                           __amd_floatx2_storage_t* out, size_t size = 1) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = fp8x2_e5m2_to_float(in[i]);
  }
}

__global__ void kernel_fp8x2_e5m2_to_float_scale(__amd_floatx2_storage_t* in,
                                                 __amd_floatx2_storage_t* out,
                                                 const __amd_scale_t scale, const size_t size = 1) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = fp8x2_e5m2_to_float_scale(in[i], scale);
  }
}

__global__ void kernel_fp8x2_e5m2_to_fp16_scale(__amd_fp16x2_storage_t* in,
                                                __amd_fp16x2_storage_t* out,
                                                const __amd_scale_t scale, const size_t size = 1) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = fp8x2_e5m2_to_fp16_scale(in[i], scale);
  }
}
}  // namespace cxx_ocp

TEST_CASE("Unit_ocp_cxx_fp8x2") {
  using namespace cxx_ocp;
  const __amd_floatx2_storage_t in = {-5.0f, 7.0f};
  __amd_floatx2_storage_t *d_in, *d_out;
  __amd_fp16x2_storage_t fp16_in{static_cast<_Float16>(in[0]), static_cast<_Float16>(in[1])},
      *fp16_d_in, *fp16_d_out;
  HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx2_storage_t)));
  HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx2_storage_t)));
  HIP_CHECK(hipMalloc(&fp16_d_in, sizeof(__amd_fp16x2_storage_t)));
  HIP_CHECK(hipMalloc(&fp16_d_out, sizeof(__amd_fp16x2_storage_t)));
  HIP_CHECK(hipMemcpy(d_in, &in, sizeof(__amd_floatx2_storage_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(fp16_d_in, &fp16_in, sizeof(__amd_fp16x2_storage_t), hipMemcpyHostToDevice));

  SECTION("E4M3") {
    SECTION("CPU") {
      auto ret = fp8x2_e4m3_to_float(in);
      INFO("In : " << in[0] << ", " << in[1]);
      INFO("Out: " << ret[0] << ", " << ret[1]);
      REQUIRE(in[0] == ret[0]);
      REQUIRE(in[1] == ret[1]);
    }

    SECTION("GPU") {
      kernel_fp8x2_e4m3_to_float<<<1, 32>>>(d_in, d_out);
      __amd_floatx2_storage_t ret;
      HIP_CHECK(hipMemcpy(&ret, d_out, sizeof(__amd_floatx2_storage_t), hipMemcpyDeviceToHost));
      INFO("In : " << in[0] << ", " << in[1]);
      INFO("Out: " << ret[0] << ", " << ret[1]);
      REQUIRE(in[0] == ret[0]);
      REQUIRE(in[1] == ret[1]);
    }

    SECTION("CPU Scaled") {
      std::vector<__amd_scale_t> scales = {0, 1, 2, 3};
      for (const auto scale : scales) {
        __amd_floatx2_storage_t ret = fp8x2_e4m3_to_float_scale(in, scale);
        INFO("In : " << in[0] << ", " << in[1]);
        INFO("Out: " << ret[0] << ", " << ret[1]);
        REQUIRE(in[0] == ret[0]);
        REQUIRE(in[1] == ret[1]);
      }
    }

    SECTION("GPU Scaled") {
      std::vector<__amd_scale_t> scales = {0, 1, 2, 3};
      for (const auto scale : scales) {
        __amd_floatx2_storage_t ret;
        kernel_fp8x2_e4m3_to_float_scale<<<1, 32>>>(d_in, d_out, scale);
        HIP_CHECK(hipMemcpy(&ret, d_out, sizeof(__amd_floatx2_storage_t), hipMemcpyDeviceToHost));
        INFO("In : " << in[0] << ", " << in[1]);
        INFO("Out: " << ret[0] << ", " << ret[1]);
        REQUIRE(in[0] == ret[0]);
        REQUIRE(in[1] == ret[1]);
      }
    }

    SECTION("CPU fp16 scalex") {
      std::vector<__amd_scale_t> scales = {0, 1, 2, 3};
      for (const auto scale : scales) {
        auto ret = fp8x2_e4m3_to_fp16_scale(fp16_in, scale);
        INFO("In : " << in[0] << ", " << in[1]);
        INFO("Out: " << float(ret[0]) << ", " << float(ret[1]));
        REQUIRE(fp16_in[0] == ret[0]);
        REQUIRE(fp16_in[1] == ret[1]);
      }
    }

    SECTION("GPU fp16 scale") {
      __amd_fp16x2_storage_t ret;
      std::vector<__amd_scale_t> scales = {0, 1, 2, 3};
      for (const auto scale : scales) {
        kernel_fp8x2_e4m3_to_fp16_scale<<<1, 32>>>(fp16_d_in, fp16_d_out, scale);
        HIP_CHECK(
            hipMemcpy(&ret, fp16_d_out, sizeof(__amd_fp16x2_storage_t), hipMemcpyDeviceToHost));
        INFO("In : " << in[0] << ", " << in[1]);
        INFO("Out: " << float(ret[0]) << ", " << float(ret[1]));
        REQUIRE(fp16_in[0] == ret[0]);
        REQUIRE(fp16_in[1] == ret[1]);
      }
    }
  }

  SECTION("E5M2") {
    SECTION("CPU") {
      auto ret = fp8x2_e5m2_to_float(in);
      INFO("In : " << in[0] << ", " << in[1]);
      INFO("Out: " << ret[0] << ", " << ret[1]);
      REQUIRE(in[0] == ret[0]);
      REQUIRE(in[1] == ret[1]);
    }

    SECTION("GPU") {
      kernel_fp8x2_e4m3_to_float<<<1, 32>>>(d_in, d_out);
      __amd_floatx2_storage_t ret;
      HIP_CHECK(hipMemcpy(&ret, d_out, sizeof(__amd_floatx2_storage_t), hipMemcpyDeviceToHost));
      INFO("In : " << in[0] << ", " << in[1]);
      INFO("Out: " << ret[0] << ", " << ret[1]);
      REQUIRE(in[0] == ret[0]);
      REQUIRE(in[1] == ret[1]);
    }

    SECTION("CPU Scaled") {
      std::vector<__amd_scale_t> scales = {0, 1, 2, 3};
      for (const auto scale : scales) {
        __amd_floatx2_storage_t ret = fp8x2_e5m2_to_float_scale(in, scale);
        INFO("In : " << in[0] << ", " << in[1]);
        INFO("Out: " << ret[0] << ", " << ret[1]);
        REQUIRE(in[0] == ret[0]);
        REQUIRE(in[1] == ret[1]);
      }
    }

    SECTION("GPU Scaled") {
      std::vector<__amd_scale_t> scales = {0, 1, 2, 3};
      for (const auto scale : scales) {
        __amd_floatx2_storage_t ret;
        kernel_fp8x2_e5m2_to_float_scale<<<1, 32>>>(d_in, d_out, scale);
        HIP_CHECK(hipMemcpy(&ret, d_out, sizeof(__amd_floatx2_storage_t), hipMemcpyDeviceToHost));
        INFO("In : " << in[0] << ", " << in[1]);
        INFO("Out: " << ret[0] << ", " << ret[1]);
        REQUIRE(in[0] == ret[0]);
        REQUIRE(in[1] == ret[1]);
      }
    }

    SECTION("CPU fp16 scale") {
      std::vector<__amd_scale_t> scales = {0, 1, 2, 3};
      for (const auto scale : scales) {
        auto ret = fp8x2_e5m2_to_fp16_scale(fp16_in, scale);
        INFO("In : " << in[0] << ", " << in[1]);
        INFO("Out: " << float(ret[0]) << ", " << float(ret[1]));
        REQUIRE(fp16_in[0] == ret[0]);
        REQUIRE(fp16_in[1] == ret[1]);
      }
    }

    SECTION("GPU fp16 scale") {
      __amd_fp16x2_storage_t ret;
      std::vector<__amd_scale_t> scales = {0, 1, 2, 3};
      for (const auto scale : scales) {
        kernel_fp8x2_e5m2_to_fp16_scale<<<1, 32>>>(fp16_d_in, fp16_d_out, scale);
        HIP_CHECK(
            hipMemcpy(&ret, fp16_d_out, sizeof(__amd_fp16x2_storage_t), hipMemcpyDeviceToHost));
        INFO("In : " << in[0] << ", " << in[1]);
        INFO("Out: " << float(ret[0]) << ", " << float(ret[1]));
        REQUIRE(fp16_in[0] == ret[0]);
        REQUIRE(fp16_in[1] == ret[1]);
      }
    }
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(fp16_d_in));
  HIP_CHECK(hipFree(fp16_d_out));
}

namespace cxx_ocp {
#if __AVX512F__
__host__ __device__ __amd_floatx32_storage_t
fp6x32_e3m2_to_float_scale(const __amd_floatx32_storage_t in, __amd_scale_t scale) {
  __hipext_ocp_fp6x32_e3m2 tmp(in, 0, scale);
  return tmp.get_scaled_floatx32(scale);
}

__global__ void kernel_fp6x32_e3m2_to_float_scale(__amd_floatx32_storage_t* in,
                                                  __amd_floatx32_storage_t* out,
                                                  const __amd_scale_t scale,
                                                  const size_t size = 1) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = fp6x32_e3m2_to_float_scale(in[i], scale);
  }
}

__host__ __device__ __amd_floatx32_storage_t
fp6x32_e2m3_to_float_scale(const __amd_floatx32_storage_t in, __amd_scale_t scale) {
  __hipext_ocp_fp6x32_e2m3 tmp(in, 0, scale);
  return tmp.get_scaled_floatx32(scale);
}

__global__ void kernel_fp6x32_e2m3_to_float_scale(__amd_floatx32_storage_t* in,
                                                  __amd_floatx32_storage_t* out,
                                                  const __amd_scale_t scale,
                                                  const size_t size = 1) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = fp6x32_e2m3_to_float_scale(in[i], scale);
  }
}
#endif
}  // namespace cxx_ocp

#if __AVX512F__
TEST_CASE("Unit_ocp_cxx_fp6x32") {
  using namespace cxx_ocp;
  __amd_floatx32_storage_t in;
  float iter = -8.0f;
  for (int i = 0; i < 32; i++) {
    in[i] = iter;
    iter++;
    if (iter > 8.0f) {
      iter = -8.0f;
    }
  }
  __amd_floatx32_storage_t *d_in, *d_out;
  HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx32_storage_t)));
  HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx32_storage_t)));
  HIP_CHECK(hipMemcpy(d_in, &in, sizeof(__amd_floatx32_storage_t), hipMemcpyHostToDevice));

  SECTION("E3M2") {
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      auto cpu_res = fp6x32_e3m2_to_float_scale(in, scale);
      kernel_fp6x32_e3m2_to_float_scale<<<1, 32>>>(d_in, d_out, scale);
      __amd_floatx32_storage_t gpu_res;
      HIP_CHECK(
          hipMemcpy(&gpu_res, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
      for (size_t i = 0; i < 32; i++) {
        INFO("Index: " << i << " cpu: " << cpu_res[i] << " gpu:" << gpu_res[i]);
        REQUIRE(cpu_res[i] == gpu_res[i]);
      }
    }
  }

  SECTION("E2M3") {
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      auto cpu_res = fp6x32_e2m3_to_float_scale(in, scale);
      kernel_fp6x32_e2m3_to_float_scale<<<1, 32>>>(d_in, d_out, scale);
      __amd_floatx32_storage_t gpu_res;
      HIP_CHECK(
          hipMemcpy(&gpu_res, d_out, sizeof(__amd_floatx32_storage_t), hipMemcpyDeviceToHost));
      for (size_t i = 0; i < 32; i++) {
        INFO("Index: " << i << " cpu: " << cpu_res[i] << " gpu:" << gpu_res[i]);
        REQUIRE(cpu_res[i] == gpu_res[i]);
      }
    }
  }
}
#endif

namespace cxx_ocp {
__host__ __device__ __amd_floatx2_storage_t
fp4x2_e2m1_to_float_scale(const __amd_floatx2_storage_t in, __amd_scale_t scale) {
  __hipext_ocp_fp4x2_e2m1 tmp(in, 0, scale);
  return tmp.get_scaled_floatx2(scale);
}

__global__ void kernel_fp4x2_e2m1_to_float_scale(__amd_floatx2_storage_t* in,
                                                 __amd_floatx2_storage_t* out,
                                                 const __amd_scale_t scale, const size_t size = 1) {
  int i = threadIdx.x;
  if (i < size) {
    out[i] = fp4x2_e2m1_to_float_scale(in[i], scale);
  }
}
}  // namespace cxx_ocp

TEST_CASE("Unit_ocp_cxx_fp4x2") {
  using namespace cxx_ocp;
  const __amd_floatx2_storage_t in = {-2.0f, 3.0f};
  __amd_floatx2_storage_t *d_in, *d_out;
  HIP_CHECK(hipMalloc(&d_in, sizeof(__amd_floatx2_storage_t)));
  HIP_CHECK(hipMalloc(&d_out, sizeof(__amd_floatx2_storage_t)));
  HIP_CHECK(hipMemcpy(d_in, &in, sizeof(__amd_floatx2_storage_t), hipMemcpyHostToDevice));

  SECTION("CPU-GPU compare") {
    std::vector<__amd_scale_t> scales{0, 1, 2};
    for (const auto scale : scales) {
      auto cpu_res = fp4x2_e2m1_to_float_scale(in, scale);
      __amd_floatx2_storage_t gpu_res;
      kernel_fp4x2_e2m1_to_float_scale<<<1, 32>>>(d_in, d_out, scale);
      HIP_CHECK(hipMemcpy(&gpu_res, d_out, sizeof(__amd_floatx2_storage_t), hipMemcpyDeviceToHost));
      INFO("CPU: " << cpu_res[0] << ", " << cpu_res[1]);
      INFO("GPU: " << gpu_res[0] << ", " << gpu_res[1]);
      REQUIRE(cpu_res[0] == gpu_res[0]);
      REQUIRE(cpu_res[1] == gpu_res[1]);
    }
  }
}
