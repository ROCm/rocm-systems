/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

/*
Negative kernels used for the float type casting negative Test Cases that are using RTC.
*/

static constexpr auto kFloat2Int{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void float2int_rd_kernel_v1(int* result, float* x) { *result = __float2int_rd(x); }
  __global__ void float2int_rd_kernel_v2(int* result, Dummy x) { *result = __float2int_rd(x); }
  __global__ void float2int_rd_kernel_v3(Dummy* result, float x) { *result = __float2int_rd(x); }
  __global__ void float2int_rn_kernel_v1(int* result, float* x) { *result = __float2int_rn(x); }
  __global__ void float2int_rn_kernel_v2(int* result, Dummy x) { *result = __float2int_rn(x); }
  __global__ void float2int_rn_kernel_v3(Dummy* result, float x) { *result = __float2int_rn(x); }
  __global__ void float2int_ru_kernel_v1(int* result, float* x) { *result = __float2int_ru(x); }
  __global__ void float2int_ru_kernel_v2(int* result, Dummy x) { *result = __float2int_ru(x); }
  __global__ void float2int_ru_kernel_v3(Dummy* result, float x) { *result = __float2int_ru(x); }
  __global__ void float2int_rz_kernel_v1(int* result, float* x) { *result = __float2int_rz(x); }
  __global__ void float2int_rz_kernel_v2(int* result, Dummy x) { *result = __float2int_rz(x); }
  __global__ void float2int_rz_kernel_v3(Dummy* result, float x) { *result = __float2int_rz(x); }
)"};

static constexpr auto kFloat2Uint{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void float2uint_rd_kernel_v1(unsigned int* result, float* x) { *result = __float2uint_rd(x); }
  __global__ void float2uint_rd_kernel_v2(unsigned int* result, Dummy x) { *result = __float2uint_rd(x); }
  __global__ void float2uint_rd_kernel_v3(Dummy* result, float x) { *result = __float2uint_rd(x); }
  __global__ void float2uint_rn_kernel_v1(unsigned int* result, float* x) { *result = __float2uint_rn(x); }
  __global__ void float2uint_rn_kernel_v2(unsigned int* result, Dummy x) { *result = __float2uint_rn(x); }
  __global__ void float2uint_rn_kernel_v3(Dummy* result, float x) { *result = __float2uint_rn(x); }
  __global__ void float2uint_ru_kernel_v1(unsigned int* result, float* x) { *result = __float2uint_ru(x); }
  __global__ void float2uint_ru_kernel_v2(unsigned int* result, Dummy x) { *result = __float2uint_ru(x); }
  __global__ void float2uint_ru_kernel_v3(Dummy* result, float x) { *result = __float2uint_ru(x); }
  __global__ void float2uint_rz_kernel_v1(unsigned int* result, float* x) { *result = __float2uint_rz(x); }
  __global__ void float2uint_rz_kernel_v2(unsigned int* result, Dummy x) { *result = __float2uint_rz(x); }
  __global__ void float2uint_rz_kernel_v3(Dummy* result, float x) { *result = __float2uint_rz(x); }
)"};

static constexpr auto kFloat2LL{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void float2ll_rd_kernel_v1(long long int* result, float* x) { *result = __float2ll_rd(x); }
  __global__ void float2ll_rd_kernel_v2(long long int* result, Dummy x) { *result = __float2ll_rd(x); }
  __global__ void float2ll_rd_kernel_v3(Dummy* result, float x) { *result = __float2ll_rd(x); }
  __global__ void float2ll_rn_kernel_v1(long long int* result, float* x) { *result = __float2ll_rn(x); }
  __global__ void float2ll_rn_kernel_v2(long long int* result, Dummy x) { *result = __float2ll_rn(x); }
  __global__ void float2ll_rn_kernel_v3(Dummy* result, float x) { *result = __float2ll_rn(x); }
  __global__ void float2ll_ru_kernel_v1(long long int* result, float* x) { *result = __float2ll_ru(x); }
  __global__ void float2ll_ru_kernel_v2(long long int* result, Dummy x) { *result = __float2ll_ru(x); }
  __global__ void float2ll_ru_kernel_v3(Dummy* result, float x) { *result = __float2ll_ru(x); }
  __global__ void float2ll_rz_kernel_v1(long long int* result, float* x) { *result = __float2ll_rz(x); }
  __global__ void float2ll_rz_kernel_v2(long long int* result, Dummy x) { *result = __float2ll_rz(x); }
  __global__ void float2ll_rz_kernel_v3(Dummy* result, float x) { *result = __float2ll_rz(x); }
)"};

static constexpr auto kFloat2ULL{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void float2ull_rd_kernel_v1(unsigned long long int* result, float* x) { *result = __float2ull_rd(x); }
  __global__ void float2ull_rd_kernel_v2(unsigned long long int* result, Dummy x) { *result = __float2ull_rd(x); }
  __global__ void float2ull_rd_kernel_v3(Dummy* result, float x) { *result = __float2ull_rd(x); }
  __global__ void float2ull_rn_kernel_v1(unsigned long long int* result, float* x) { *result = __float2ull_rn(x); }
  __global__ void float2ull_rn_kernel_v2(unsigned long long int* result, Dummy x) { *result = __float2ull_rn(x); }
  __global__ void float2ull_rn_kernel_v3(Dummy* result, float x) { *result = __float2ull_rn(x); }
  __global__ void float2ull_ru_kernel_v1(unsigned long long int* result, float* x) { *result = __float2ull_ru(x); }
  __global__ void float2ull_ru_kernel_v2(unsigned long long int* result, Dummy x) { *result = __float2ull_ru(x); }
  __global__ void float2ull_ru_kernel_v3(Dummy* result, float x) { *result = __float2ull_ru(x); }
  __global__ void float2ull_rz_kernel_v1(unsigned long long int* result, float* x) { *result = __float2ull_rz(x); }
  __global__ void float2ull_rz_kernel_v2(unsigned long long int* result, Dummy x) { *result = __float2ull_rz(x); }
  __global__ void float2ull_rz_kernel_v3(Dummy* result, float x) { *result = __float2ull_rz(x); }
)"};

static constexpr auto kFloatAsInt{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void float_as_int_kernel_v1(int* result, float* x) { *result = __float_as_int(x); }
  __global__ void float_as_int_kernel_v2(int* result, Dummy x) { *result = __float_as_int(x); }
  __global__ void float_as_int_kernel_v3(Dummy* result, float x) { *result = __float_as_int(x); }
)"};

static constexpr auto kFloatAsUint{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void float_as_uint_kernel_v1(unsigned int* result, float* x) { *result = __float_as_uint(x); }
  __global__ void float_as_uint_kernel_v2(unsigned int* result, Dummy x) { *result = __float_as_uint(x); }
  __global__ void float_as_uint_kernel_v3(Dummy* result, float x) { *result = __float_as_uint(x); }
)"};
