/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

/*
Negative kernels used for the math root negative Test Cases that are using RTC.
*/

static constexpr auto kSqrt{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void sqrt_kernel_v1(double* x) { double result = sqrt(x); }
  __global__ void sqrt_kernel_v2(Dummy x) { double result = sqrt(x); }
  __global__ void sqrtf_kernel_v1(float* x) { float result = sqrtf(x); }
  __global__ void sqrtf_kernel_v2(Dummy x) { float result = sqrtf(x); }
)"};

static constexpr auto kRsqrt{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void rsqrt_kernel_v1(double* x) { double result = rsqrt(x); }
  __global__ void rsqrt_kernel_v2(Dummy x) { double result = rsqrt(x); }
  __global__ void rsqrtf_kernel_v1(float* x) { float result = rsqrtf(x); }
  __global__ void rsqrtf_kernel_v2(Dummy x) { float result = rsqrtf(x); }
)"};

static constexpr auto kCbrt{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void cbrt_kernel_v1(double* x) { double result = cbrt(x); }
  __global__ void cbrt_kernel_v2(Dummy x) { double result = cbrt(x); }
  __global__ void cbrtf_kernel_v1(float* x) { float result = cbrtf(x); }
  __global__ void cbrtf_kernel_v2(Dummy x) { float result = cbrtf(x); }
)"};

static constexpr auto kRcbrt{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void rcbrt_kernel_v1(double* x) { double result = rcbrt(x); }
  __global__ void rcbrt_kernel_v2(Dummy x) { double result = rcbrt(x); }
  __global__ void rcbrtf_kernel_v1(float* x) { float result = rcbrtf(x); }
  __global__ void rcbrtf_kernel_v2(Dummy x) { float result = rcbrtf(x); }
)"};

static constexpr auto kHypot{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void hypot_kernel_v1(double* x, double y) { double result = hypot(x, y); }
  __global__ void hypot_kernel_v2(double x, double* y) { double result = hypot(x, y); }
  __global__ void hypot_kernel_v3(Dummy x, double y) { double result = hypot(x, y); }
  __global__ void hypot_kernel_v4(double x, Dummy y) { double result = hypot(x, y); }
  __global__ void hypotf_kernel_v1(float* x, float y) { float result = hypotf(x, y); }
  __global__ void hypotf_kernel_v2(float x, float* y) { float result = hypotf(x, y); }
  __global__ void hypotf_kernel_v3(Dummy x, float y) { float result = hypotf(x, y); }
  __global__ void hypotf_kernel_v4(float x, Dummy y) { float result = hypotf(x, y); }
)"};

static constexpr auto kRhypot{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void rhypot_kernel_v1(double* x, double y) { double result = rhypot(x, y); }
  __global__ void rhypot_kernel_v2(double x, double* y) { double result = rhypot(x, y); }
  __global__ void rhypot_kernel_v3(Dummy x, double y) { double result = rhypot(x, y); }
  __global__ void rhypot_kernel_v4(double x, Dummy y) { double result = rhypot(x, y); }
  __global__ void rhypotf_kernel_v1(float* x, float y) { float result = rhypotf(x, y); }
  __global__ void rhypotf_kernel_v2(float x, float* y) { float result = rhypotf(x, y); }
  __global__ void rhypotf_kernel_v3(Dummy x, float y) { float result = rhypotf(x, y); }
  __global__ void rhypotf_kernel_v4(float x, Dummy y) { float result = rhypotf(x, y); }
)"};

static constexpr auto kNorm3D{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void norm3d_kernel_v1(double* x, double y, double z) {
    double result = norm3d(x, y, z);
  }
  __global__ void norm3d_kernel_v2(double x, double* y, double z) {
    double result = norm3d(x, y, z);
  }
  __global__ void norm3d_kernel_v3(double x, double y, double* z) {
    double result = norm3d(x, y, z);
  }
  __global__ void norm3d_kernel_v4(Dummy x, double y, double z) {
    double result = norm3d(x, y, z);
  }
  __global__ void norm3d_kernel_v5(double x, Dummy y, double z) {
    double result = norm3d(x, y, z);
  }
  __global__ void norm3d_kernel_v6(double x, double y, Dummy z) {
    double result = norm3d(x, y, z);
  }
  __global__ void norm3df_kernel_v1(float* x, float y, float z) {
    float result = norm3df(x, y, z);
  }
  __global__ void norm3df_kernel_v2(float x, float* y, float z) {
    float result = norm3df(x, y, z);
  }
  __global__ void norm3df_kernel_v3(float x, float y, float* z) {
    float result = norm3df(x, y, z);
  }
  __global__ void norm3df_kernel_v4(Dummy x, float y, float z) {
    float result = norm3df(x, y, z);
  }
  __global__ void norm3df_kernel_v5(float x, Dummy y, float z) {
    float result = norm3df(x, y, z);
  }
  __global__ void norm3df_kernel_v6(float x, float y, Dummy z) {
    float result = norm3df(x, y, z);
  }
)"};

static constexpr auto kRnorm3D{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void rnorm3d_kernel_v1(double* x, double y, double z) {
    double result = rnorm3d(x, y, z);
  }
  __global__ void rnorm3d_kernel_v2(double x, double* y, double z) {
    double result = rnorm3d(x, y, z);
  }
  __global__ void rnorm3d_kernel_v3(double x, double y, double* z) {
    double result = rnorm3d(x, y, z);
  }
  __global__ void rnorm3d_kernel_v4(Dummy x, double y, double z) {
    double result = rnorm3d(x, y, z);
  }
  __global__ void rnorm3d_kernel_v5(double x, Dummy y, double z) {
    double result = rnorm3d(x, y, z);
  }
  __global__ void rnorm3d_kernel_v6(double x, double y, Dummy z) {
    double result = rnorm3d(x, y, z);
  }
  __global__ void rnorm3df_kernel_v1(float* x, float y, float z) {
    float result = rnorm3df(x, y, z);
  }
  __global__ void rnorm3df_kernel_v2(float x, float* y, float z) {
    float result = rnorm3df(x, y, z);
  }
  __global__ void rnorm3df_kernel_v3(float x, float y, float* z) {
    float result = rnorm3df(x, y, z);
  }
  __global__ void rnorm3df_kernel_v4(Dummy x, float y, float z) {
    float result = rnorm3df(x, y, z);
  }
  __global__ void rnorm3df_kernel_v5(float x, Dummy y, float z) {
    float result = rnorm3df(x, y, z);
  }
  __global__ void rnorm3df_kernel_v6(float x, float y, Dummy z) {
    float result = rnorm3df(x, y, z);
  }
)"};

static constexpr auto kNorm4D{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void norm4d_kernel_v1(double* x, double y, double z, double w) {
    double result = norm4d(x, y, z, w);
  }
  __global__ void norm4d_kernel_v2(double x, double* y, double z, double w) {
    double result = norm4d(x, y, z, w);
  }
  __global__ void norm4d_kernel_v3(double x, double y, double* z, double w) {
    double result = norm4d(x, y, z, w);
  }
  __global__ void norm4d_kernel_v4(double x, double y, double z, double* w) {
    double result = norm4d(x, y, z, w);
  }
  __global__ void norm4d_kernel_v5(Dummy x, double y, double z, double w) {
    double result = norm4d(x, y, z, w);
  }
  __global__ void norm4d_kernel_v6(double x, Dummy y, double z, double w) {
    double result = norm4d(x, y, z, w);
  }
  __global__ void norm4d_kernel_v7(double x, double y, Dummy z, double w) {
    double result = norm4d(x, y, z, w);
  }
  __global__ void norm4d_kernel_v8(double x, double y, double z, Dummy w) {
    double result = norm4d(x, y, z, w);
  }
  __global__ void norm4df_kernel_v1(float* x, float y, float z, float w) {
    float result = norm4df(x, y, z, w);
  }
  __global__ void norm4df_kernel_v2(float x, float* y, float z, float w) {
    float result = norm4df(x, y, z, w);
  }
  __global__ void norm4df_kernel_v3(float x, float y, float* z, float w) {
    float result = norm4df(x, y, z, w);
  }
  __global__ void norm4df_kernel_v4(float x, float y, float z, float* w) {
    float result = norm4df(x, y, z, w);
  }
  __global__ void norm4df_kernel_v5(Dummy x, float y, float z, float w) {
    float result = norm4df(x, y, z, w);
  }
  __global__ void norm4df_kernel_v6(float x, Dummy y, float z, float w) {
    float result = norm4df(x, y, z, w);
  }
  __global__ void norm4df_kernel_v7(float x, float y, Dummy z, float w) {
    float result = norm4df(x, y, z, w);
  }
  __global__ void norm4df_kernel_v8(float x, float y, float z, Dummy w) {
    float result = norm4df(x, y, z, w);
  }
)"};

static constexpr auto kRnorm4D{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void rnorm4d_kernel_v1(double* x, double y, double z, double w) {
    double result = rnorm4d(x, y, z, w);
  }
  __global__ void rnorm4d_kernel_v2(double x, double* y, double z, double w) {
    double result = rnorm4d(x, y, z, w);
  }
  __global__ void rnorm4d_kernel_v3(double x, double y, double* z, double w) {
    double result = rnorm4d(x, y, z, w);
  }
  __global__ void rnorm4d_kernel_v4(double x, double y, double z, double* w) {
    double result = rnorm4d(x, y, z, w);
  }
  __global__ void rnorm4d_kernel_v5(Dummy x, double y, double z, double w) {
    double result = rnorm4d(x, y, z, w);
  }
  __global__ void rnorm4d_kernel_v6(double x, Dummy y, double z, double w) {
    double result = rnorm4d(x, y, z, w);
  }
  __global__ void rnorm4d_kernel_v7(double x, double y, Dummy z, double w) {
    double result = rnorm4d(x, y, z, w);
  }
  __global__ void rnorm4d_kernel_v8(double x, double y, double z, Dummy w) {
    double result = rnorm4d(x, y, z, w);
  }
  __global__ void rnorm4df_kernel_v1(float* x, float y, float z, float w) {
    float result = rnorm4df(x, y, z, w);
  }
  __global__ void rnorm4df_kernel_v2(float x, float* y, float z, float w) {
    float result = rnorm4df(x, y, z, w);
  }
  __global__ void rnorm4df_kernel_v3(float x, float y, float* z, float w) {
    float result = rnorm4df(x, y, z, w);
  }
  __global__ void rnorm4df_kernel_v4(float x, float y, float z, float* w) {
    float result = rnorm4df(x, y, z, w);
  }
  __global__ void rnorm4df_kernel_v5(Dummy x, float y, float z, float w) {
    float result = rnorm4df(x, y, z, w);
  }
  __global__ void rnorm4df_kernel_v6(float x, Dummy y, float z, float w) {
    float result = rnorm4df(x, y, z, w);
  }
  __global__ void rnorm4df_kernel_v7(float x, float y, Dummy z, float w) {
    float result = rnorm4df(x, y, z, w);
  }
  __global__ void rnorm4df_kernel_v8(float x, float y, float z, Dummy w) {
    float result = rnorm4df(x, y, z, w);
  }
)"};

static constexpr auto kNorm{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void norm_kernel_v1(int* dim, const double* a) {
    double result = norm(dim, a);
  }
  __global__ void norm_kernel_v2(Dummy dim, const double* a) {
    double result = norm(dim, a);
  }
  __global__ void norm_kernel_v3(int dim, const int* a) {
    double result = norm(dim, a);
  }
  __global__ void norm_kernel_v4(int dim, const char* a) {
    double result = norm(dim, a);
  }
  __global__ void norm_kernel_v5(int dim, const short* a) {
    double result = norm(dim, a);
  }
  __global__ void norm_kernel_v6(int dim, const long* a) {
    double result = norm(dim, a);
  }
  __global__ void norm_kernel_v7(int dim, const long long* a) {
    double result = norm(dim, a);
  }
  __global__ void norm_kernel_v8(int dim, const float* a) {
    double result = norm(dim, a);
  }
  __global__ void norm_kernel_v9(int dim, const Dummy* a) {
    double result = norm(dim, a);
  }
  __global__ void normf_kernel_v1(int* dim, const float* a) {
    float result = normf(dim, a);
  }
  __global__ void normf_kernel_v2(Dummy dim, const float* a) {
    float result = normf(dim, a);
  }
  __global__ void normf_kernel_v3(int dim, const int* a) {
    float result = normf(dim, a);
  }
  __global__ void normf_kernel_v4(int dim, const char* a) {
    float result = normf(dim, a);
  }
  __global__ void normf_kernel_v5(int dim, const short* a) {
    float result = normf(dim, a);
  }
  __global__ void normf_kernel_v6(int dim, const long* a) {
    float result = normf(dim, a);
  }
  __global__ void normf_kernel_v7(int dim, const long long* a) {
    float result = normf(dim, a);
  }
  __global__ void normf_kernel_v8(int dim, const double* a) {
    float result = normf(dim, a);
  }
  __global__ void normf_kernel_v9(int dim, const Dummy* a) {
    double result = normf(dim, a);
  }
)"};

static constexpr auto kRnorm{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void rnorm_kernel_v1(int* dim, const double* a) {
    double result = rnorm(dim, a);
  }
  __global__ void rnorm_kernel_v2(Dummy dim, const double* a) {
    double result = rnorm(dim, a);
  }
  __global__ void rnorm_kernel_v3(int dim, const int* a) {
    double result = rnorm(dim, a);
  }
  __global__ void rnorm_kernel_v4(int dim, const char* a) {
    double result = rnorm(dim, a);
  }
  __global__ void rnorm_kernel_v5(int dim, const short* a) {
    double result = rnorm(dim, a);
  }
  __global__ void rnorm_kernel_v6(int dim, const long* a) {
    double result = rnorm(dim, a);
  }
  __global__ void rnorm_kernel_v7(int dim, const long long* a) {
    double result = rnorm(dim, a);
  }
  __global__ void rnorm_kernel_v8(int dim, const float* a) {
    double result = rnorm(dim, a);
  }
  __global__ void rnorm_kernel_v9(int dim, const Dummy* a) {
    double result = rnorm(dim, a);
  }
  __global__ void rnormf_kernel_v1(int* dim, const float* a) {
    float result = rnormf(dim, a);
  }
  __global__ void rnormf_kernel_v2(Dummy dim, const float* a) {
    float result = rnormf(dim, a);
  }
  __global__ void rnormf_kernel_v3(int dim, const int* a) {
    float result = rnormf(dim, a);
  }
  __global__ void rnormf_kernel_v4(int dim, const char* a) {
    float result = rnormf(dim, a);
  }
  __global__ void rnormf_kernel_v5(int dim, const short* a) {
    float result = rnormf(dim, a);
  }
  __global__ void rnormf_kernel_v6(int dim, const long* a) {
    float result = rnormf(dim, a);
  }
  __global__ void rnormf_kernel_v7(int dim, const long long* a) {
    float result = rnormf(dim, a);
  }
  __global__ void rnormf_kernel_v8(int dim, const double* a) {
    float result = rnormf(dim, a);
  }
  __global__ void rnormf_kernel_v9(int dim, const Dummy* a) {
    double result = rnormf(dim, a);
  }
)"};
