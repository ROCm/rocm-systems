/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

/*
Negative kernels used for the double type casting negative Test Cases that are using RTC.
*/

static constexpr auto kDouble2Int{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void double2int_rd_kernel_v1(int* result, double* x) { *result = __double2int_rd(x); }
  __global__ void double2int_rd_kernel_v2(int* result, Dummy x) { *result = __double2int_rd(x); }
  __global__ void double2int_rd_kernel_v3(Dummy* result, double x) { *result = __double2int_rd(x); }
  __global__ void double2int_rn_kernel_v1(int* result, double* x) { *result = __double2int_rn(x); }
  __global__ void double2int_rn_kernel_v2(int* result, Dummy x) { *result = __double2int_rn(x); }
  __global__ void double2int_rn_kernel_v3(Dummy* result, double x) { *result = __double2int_rn(x); }
  __global__ void double2int_ru_kernel_v1(int* result, double* x) { *result = __double2int_ru(x); }
  __global__ void double2int_ru_kernel_v2(int* result, Dummy x) { *result = __double2int_ru(x); }
  __global__ void double2int_ru_kernel_v3(Dummy* result, double x) { *result = __double2int_ru(x); }
  __global__ void double2int_rz_kernel_v1(int* result, double* x) { *result = __double2int_rz(x); }
  __global__ void double2int_rz_kernel_v2(int* result, Dummy x) { *result = __double2int_rz(x); }
  __global__ void double2int_rz_kernel_v3(Dummy* result, double x) { *result = __double2int_rz(x); }
)"};

static constexpr auto kDouble2Uint{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void double2uint_rd_kernel_v1(unsigned int* result, double* x) { *result = __double2uint_rd(x); }
  __global__ void double2uint_rd_kernel_v2(unsigned int* result, Dummy x) { *result = __double2uint_rd(x); }
  __global__ void double2uint_rd_kernel_v3(Dummy* result, double x) { *result = __double2uint_rd(x); }
  __global__ void double2uint_rn_kernel_v1(unsigned int* result, double* x) { *result = __double2uint_rn(x); }
  __global__ void double2uint_rn_kernel_v2(unsigned int* result, Dummy x) { *result = __double2uint_rn(x); }
  __global__ void double2uint_rn_kernel_v3(Dummy* result, double x) { *result = __double2uint_rn(x); }
  __global__ void double2uint_ru_kernel_v1(unsigned int* result, double* x) { *result = __double2uint_ru(x); }
  __global__ void double2uint_ru_kernel_v2(unsigned int* result, Dummy x) { *result = __double2uint_ru(x); }
  __global__ void double2uint_ru_kernel_v3(Dummy* result, double x) { *result = __double2uint_ru(x); }
  __global__ void double2uint_rz_kernel_v1(unsigned int* result, double* x) { *result = __double2uint_rz(x); }
  __global__ void double2uint_rz_kernel_v2(unsigned int* result, Dummy x) { *result = __double2uint_rz(x); }
  __global__ void double2uint_rz_kernel_v3(Dummy* result, double x) { *result = __double2uint_rz(x); }
)"};

static constexpr auto kDouble2LL{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void double2ll_rd_kernel_v1(long long int* result, double* x) { *result = __double2ll_rd(x); }
  __global__ void double2ll_rd_kernel_v2(long long int* result, Dummy x) { *result = __double2ll_rd(x); }
  __global__ void double2ll_rd_kernel_v3(Dummy* result, double x) { *result = __double2ll_rd(x); }
  __global__ void double2ll_rn_kernel_v1(long long int* result, double* x) { *result = __double2ll_rn(x); }
  __global__ void double2ll_rn_kernel_v2(long long int* result, Dummy x) { *result = __double2ll_rn(x); }
  __global__ void double2ll_rn_kernel_v3(Dummy* result, double x) { *result = __double2ll_rn(x); }
  __global__ void double2ll_ru_kernel_v1(long long int* result, double* x) { *result = __double2ll_ru(x); }
  __global__ void double2ll_ru_kernel_v2(long long int* result, Dummy x) { *result = __double2ll_ru(x); }
  __global__ void double2ll_ru_kernel_v3(Dummy* result, double x) { *result = __double2ll_ru(x); }
  __global__ void double2ll_rz_kernel_v1(long long int* result, double* x) { *result = __double2ll_rz(x); }
  __global__ void double2ll_rz_kernel_v2(long long int* result, Dummy x) { *result = __double2ll_rz(x); }
  __global__ void double2ll_rz_kernel_v3(Dummy* result, double x) { *result = __double2ll_rz(x); }
)"};

static constexpr auto kDouble2ULL{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void double2ull_rd_kernel_v1(unsigned long long int* result, double* x) { *result = __double2ull_rd(x); }
  __global__ void double2ull_rd_kernel_v2(unsigned long long int* result, Dummy x) { *result = __double2ull_rd(x); }
  __global__ void double2ull_rd_kernel_v3(Dummy* result, double x) { *result = __double2ull_rd(x); }
  __global__ void double2ull_rn_kernel_v1(unsigned long long int* result, double* x) { *result = __double2ull_rn(x); }
  __global__ void double2ull_rn_kernel_v2(unsigned long long int* result, Dummy x) { *result = __double2ull_rn(x); }
  __global__ void double2ull_rn_kernel_v3(Dummy* result, double x) { *result = __double2ull_rn(x); }
  __global__ void double2ull_ru_kernel_v1(unsigned long long int* result, double* x) { *result = __double2ull_ru(x); }
  __global__ void double2ull_ru_kernel_v2(unsigned long long int* result, Dummy x) { *result = __double2ull_ru(x); }
  __global__ void double2ull_ru_kernel_v3(Dummy* result, double x) { *result = __double2ull_ru(x); }
  __global__ void double2ull_rz_kernel_v1(unsigned long long int* result, double* x) { *result = __double2ull_rz(x); }
  __global__ void double2ull_rz_kernel_v2(unsigned long long int* result, Dummy x) { *result = __double2ull_rz(x); }
  __global__ void double2ull_rz_kernel_v3(Dummy* result, double x) { *result = __double2ull_rz(x); }
)"};

static constexpr auto kDouble2Float{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void double2float_rd_kernel_v1(float* result, double* x) { *result = __double2float_rd(x); }
  __global__ void double2float_rd_kernel_v2(float* result, Dummy x) { *result = __double2float_rd(x); }
  __global__ void double2float_rd_kernel_v3(Dummy* result, double x) { *result = __double2float_rd(x); }
  __global__ void double2float_rn_kernel_v1(float* result, double* x) { *result = __double2float_rn(x); }
  __global__ void double2float_rn_kernel_v2(float* result, Dummy x) { *result = __double2float_rn(x); }
  __global__ void double2float_rn_kernel_v3(Dummy* result, double x) { *result = __double2float_rn(x); }
  __global__ void double2float_ru_kernel_v1(float* result, double* x) { *result = __double2float_ru(x); }
  __global__ void double2float_ru_kernel_v2(float* result, Dummy x) { *result = __double2float_ru(x); }
  __global__ void double2float_ru_kernel_v3(Dummy* result, double x) { *result = __double2float_ru(x); }
  __global__ void double2float_rz_kernel_v1(float* result, double* x) { *result = __double2float_rz(x); }
  __global__ void double2float_rz_kernel_v2(float* result, Dummy x) { *result = __double2float_rz(x); }
  __global__ void double2float_rz_kernel_v3(Dummy* result, double x) { *result = __double2float_rz(x); }
)"};

static constexpr auto kDouble2Hiint{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void double2hiint_kernel_v1(int* result, double* x) { *result = __double2hiint(x); }
  __global__ void double2hiint_kernel_v2(int* result, Dummy x) { *result = __double2hiint(x); }
  __global__ void double2hiint_kernel_v3(Dummy* result, double x) { *result = __double2hiint(x); }
)"};

static constexpr auto kDouble2Loint{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void double2loint_kernel_v1(int* result, double* x) { *result = __double2loint(x); }
  __global__ void double2loint_kernel_v2(int* result, Dummy x) { *result = __double2loint(x); }
  __global__ void double2loint_kernel_v3(Dummy* result, double x) { *result = __double2loint(x); }
)"};

static constexpr auto kDoubleAsLonglong{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void double_as_longlong_kernel_v1(long long int* result, double* x) { *result = __double_as_longlong(x); }
  __global__ void double_as_longlong_kernel_v2(long long int* result, Dummy x) { *result = __double_as_longlong(x); }
  __global__ void double_as_longlong_kernel_v3(Dummy* result, double x) { *result = __double_as_longlong(x); }
)"};
