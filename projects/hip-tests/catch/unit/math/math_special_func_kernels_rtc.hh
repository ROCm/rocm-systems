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
Negative kernels used for the math special function negative Test Cases that are using RTC.
*/

static constexpr auto kErf{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void erf_kernel_v1(double* x) { double result = erf(x); }
  __global__ void erf_kernel_v2(Dummy x) { double result = erf(x); }
  __global__ void erff_kernel_v1(float* x) { float result = erff(x); }
  __global__ void erff_kernel_v2(Dummy x) { float result = erff(x); }
)"};

static constexpr auto kErfc{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void erfc_kernel_v1(double* x) { double result = erfc(x); }
  __global__ void erfc_kernel_v2(Dummy x) { double result = erfc(x); }
  __global__ void erfcf_kernel_v1(float* x) { float result = erfcf(x); }
  __global__ void erfcf_kernel_v2(Dummy x) { float result = erfcf(x); }
)"};

static constexpr auto kErfinv{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void erfinv_kernel_v1(double* x) { double result = erfinv(x); }
  __global__ void erfinv_kernel_v2(Dummy x) { double result = erfinv(x); }
  __global__ void erfinvf_kernel_v1(float* x) { float result = erfinvf(x); }
  __global__ void erfinvf_kernel_v2(Dummy x) { float result = erfinvf(x); }
)"};

static constexpr auto kErfcinv{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void erfcinv_kernel_v1(double* x) { double result = erfcinv(x); }
  __global__ void erfcinv_kernel_v2(Dummy x) { double result = erfcinv(x); }
  __global__ void erfcinvf_kernel_v1(float* x) { float result = erfcinvf(x); }
  __global__ void erfcinvf_kernel_v2(Dummy x) { float result = erfcinvf(x); }
)"};

static constexpr auto kErfcx{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void erfcx_kernel_v1(double* x) { double result = erfcx(x); }
  __global__ void erfcx_kernel_v2(Dummy x) { double result = erfcx(x); }
  __global__ void erfcxf_kernel_v1(float* x) { float result = erfcxf(x); }
  __global__ void erfcxf_kernel_v2(Dummy x) { float result = erfcxf(x); }
)"};

static constexpr auto kNormcdf{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void normcdf_kernel_v1(double* x) { double result = normcdf(x); }
  __global__ void normcdf_kernel_v2(Dummy x) { double result = normcdf(x); }
  __global__ void normcdff_kernel_v1(float* x) { float result = normcdff(x); }
  __global__ void normcdff_kernel_v2(Dummy x) { float result = normcdff(x); }
)"};

static constexpr auto kNormcdfinv{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void normcdfinv_kernel_v1(double* x) { double result = normcdfinv(x); }
  __global__ void normcdfinv_kernel_v2(Dummy x) { double result = normcdfinv(x); }
  __global__ void normcdfinvf_kernel_v1(float* x) { float result = normcdfinvf(x); }
  __global__ void normcdfinvf_kernel_v2(Dummy x) { float result = normcdfinvf(x); }
)"};

static constexpr auto kLgamma{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void lgamma_kernel_v1(double* x) { double result = lgamma(x); }
  __global__ void lgamma_kernel_v2(Dummy x) { double result = lgamma(x); }
  __global__ void lgammaf_kernel_v1(float* x) { float result = lgammaf(x); }
  __global__ void lgammaf_kernel_v2(Dummy x) { float result = lgammaf(x); }
)"};

static constexpr auto kTgamma{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void tgamma_kernel_v1(double* x) { double result = tgamma(x); }
  __global__ void tgamma_kernel_v2(Dummy x) { double result = tgamma(x); }
  __global__ void tgammaf_kernel_v1(float* x) { float result = tgammaf(x); }
  __global__ void tgammaf_kernel_v2(Dummy x) { float result = tgammaf(x); }
)"};

static constexpr auto kJ0{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void j0_kernel_v1(double* x) { double result = j0(x); }
  __global__ void j0_kernel_v2(Dummy x) { double result = j0(x); }
  __global__ void j0f_kernel_v1(float* x) { float result = j0f(x); }
  __global__ void j0f_kernel_v2(Dummy x) { float result = j0f(x); }
)"};

static constexpr auto kJ1{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void j1_kernel_v1(double* x) { double result = j1(x); }
  __global__ void j1_kernel_v2(Dummy x) { double result = j1(x); }
  __global__ void j1f_kernel_v1(float* x) { float result = j1f(x); }
  __global__ void j1f_kernel_v2(Dummy x) { float result = j1f(x); }
)"};

static constexpr auto kJn{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void jn_kernel_v1(int* x, double y) { double result = jn(x, y); }
  __global__ void jn_kernel_v2(int x, double* y) { double result = jn(x, y); }
  __global__ void jn_kernel_v3(Dummy x, double y) { double result = jn(x, y); }
  __global__ void jn_kernel_v4(int x, Dummy y) { double result = jn(x, y); }
  __global__ void jnf_kernel_v1(int* x, float y) { float result = jnf(x, y); }
  __global__ void jnf_kernel_v2(int x, float* y) { float result = jnf(x, y); }
  __global__ void jnf_kernel_v3(Dummy x, float y) { float result = jnf(x, y); }
  __global__ void jnf_kernel_v4(int x, Dummy y) { float result = jnf(x, y); }
)"};

static constexpr auto kY0{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void y0_kernel_v1(double* x) { double result = y0(x); }
  __global__ void y0_kernel_v2(Dummy x) { double result = y0(x); }
  __global__ void y0f_kernel_v1(float* x) { float result = y0f(x); }
  __global__ void y0f_kernel_v2(Dummy x) { float result = y0f(x); }
)"};

static constexpr auto kY1{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void y1_kernel_v1(double* x) { double result = y1(x); }
  __global__ void y1_kernel_v2(Dummy x) { double result = y1(x); }
  __global__ void y1f_kernel_v1(float* x) { float result = y1f(x); }
  __global__ void y1f_kernel_v2(Dummy x) { float result = y1f(x); }
)"};

static constexpr auto kYn{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void yn_kernel_v1(int* x, double y) { double result = yn(x, y); }
  __global__ void yn_kernel_v2(int x, double* y) { double result = yn(x, y); }
  __global__ void yn_kernel_v3(Dummy x, double y) { double result = yn(x, y); }
  __global__ void yn_kernel_v4(int x, Dummy y) { double result = yn(x, y); }
  __global__ void ynf_kernel_v1(int* x, float y) { float result = ynf(x, y); }
  __global__ void ynf_kernel_v2(int x, float* y) { float result = ynf(x, y); }
  __global__ void ynf_kernel_v3(Dummy x, float y) { float result = ynf(x, y); }
  __global__ void ynf_kernel_v4(int x, Dummy y) { float result = ynf(x, y); }
)"};

static constexpr auto kCylBesselI0{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void cyl_bessel_i0_kernel_v1(double* x) { double result = cyl_bessel_i0(x); }
  __global__ void cyl_bessel_i0_kernel_v2(Dummy x) { double result = cyl_bessel_i0(x); }
  __global__ void cyl_bessel_i0f_kernel_v1(float* x) { float result = cyl_bessel_i0f(x); }
  __global__ void cyl_bessel_i0f_kernel_v2(Dummy x) { float result = cyl_bessel_i0f(x); }
)"};

static constexpr auto kCylBesselI1{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void cyl_bessel_i1_kernel_v1(double* x) { double result = cyl_bessel_i1(x); }
  __global__ void cyl_bessel_i1_kernel_v2(Dummy x) { double result = cyl_bessel_i1(x); }
  __global__ void cyl_bessel_i1f_kernel_v1(float* x) { float result = cyl_bessel_i1f(x); }
  __global__ void cyl_bessel_i1f_kernel_v2(Dummy x) { float result = cyl_bessel_i1f(x); }
)"};
