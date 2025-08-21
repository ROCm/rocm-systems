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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*
RtcKernels.h contains the string's with the which includes the kernel code.
They are utilized by the compiler option functions, defined in RtcFunctions.cpp
*/

#ifndef CATCH_UNIT_RTC_HEADERS_RTCKERNELS_H_
#define CATCH_UNIT_RTC_HEADERS_RTCKERNELS_H_
#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <math.h>

static constexpr auto max_thread_string{
    R"(
extern "C"
__global__ void max_thread(int* a) {
  int BD = blockDim.x;
  *a = BD;
}
)"};

static constexpr auto denormals_string{
    R"(
extern "C"
__global__ void denormals(double* base, double* power, double* result) {
  float denorm = powf(*base, *power);
  if (*result == 0 || *result ==1 )
    *result = (denorm==0) ? 0 : 1;
  else
    *result = powf(*base, *power);
}
)"};

static constexpr auto warning_string{
    R"(
extern "C"
__global__ void warning() {
  #warning "Just printing a WARNING message onto the terminal";
}
)"};

static constexpr auto fp32_div_sqrt_string{
    R"(
extern "C"
__global__ void fp32_div_sqrt(float* result) {
  float input = 109.6209;
  *result = sqrt(input);
}
)"};

static constexpr auto error_string{
    R"(
extern "C"
__global__ void error() {
  unsigned int a = -1;
  unsigned int b = +1;
  signed int c = -1;
  signed int d = +1;
}
)"};

static constexpr auto macro_string{
    R"(
extern "C"
__global__ void macro(int *result) {
  *result = PI;
}
)"};

static constexpr auto undef_macro_string{
    R"(
extern "C"
__global__ void undef_macro() {
  int a = Z;
}
)"};

static constexpr auto header_dir_string{
    R"(
#include "RtcFact.h"
extern "C"
__global__ void header_dir(int* a, int* val) {
  *a = fact(*val);
}
)"};

static constexpr auto rdc_string{
    R"(
extern "C"
__global__ void rdc(float* a, float* b, float* c) {
  *c = *a * *b;
}
)"};

static constexpr auto ffp_contract_string{
    R"(
extern "C"
__global__ void ffp_contract(float* a, float* b, float* c) {
  *c = *a * *b + *c;
}
)"};

static constexpr auto slp_vectorize_string{
    R"(
extern "C"
__global__ void slp_vectorize(__half2 a, __half2 x, __half2 *y) {
  (*y).data.x = x.data.x + a.data.x;
  (*y).data.y = x.data.y + a.data.y;
}
)"};

static constexpr auto unsafe_atomic_string{
    R"(
extern "C"
__global__ void unsafe_atomic(float* a) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < 1000) {
    unsafeAtomicAdd(&a[id], 0.2f);
  }
}
)"};

static constexpr auto amdgpu_ieee_string{
    R"(
extern "C"
__global__ void amdgpu_ieee(float* a, float* b, float* c) {
  *c = sqrt(*a / *b);
  printf("sqrt(a * b) = %f\n", *c);
}
)"};

static constexpr auto associative_math_string{
    R"(
extern "C"
__global__ void associative_math(int* check) {
  double x = 0.1f;
  double y = 0.2f;
  double z = 0.3f;
  if((x*y)*z != x*(y*z))
    *check = 1;
  else *check = 0;
}
)"};

#endif  // CATCH_UNIT_RTC_HEADERS_RTCKERNELS_H_
