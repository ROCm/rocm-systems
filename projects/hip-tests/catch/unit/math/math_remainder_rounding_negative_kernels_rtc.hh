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
Negative kernels used for the math remainder and rounding negative Test Cases that are using RTC.
*/

static constexpr auto kTrunc{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void trunc_kernel_v1(double* x) { auto result = trunc(x); }
  __global__ void trunc_kernel_v2(Dummy x) { auto result = trunc(x); }
  __global__ void truncf_kernel_v1(float* x) { auto result = truncf(x); }
  __global__ void truncf_kernel_v2(Dummy x) { auto result = truncf(x); }
)"};

static constexpr auto kRound{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void round_kernel_v1(double* x) { auto result = round(x); }
  __global__ void round_kernel_v2(Dummy x) { auto result = round(x); }
  __global__ void roundf_kernel_v1(float* x) { auto result = roundf(x); }
  __global__ void roundf_kernel_v2(Dummy x) { auto result = roundf(x); }
)"};

static constexpr auto kRint{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void rint_kernel_v1(double* x) { auto result = rint(x); }
  __global__ void rint_kernel_v2(Dummy x) { auto result = rint(x); }
  __global__ void rintf_kernel_v1(float* x) { auto result = rintf(x); }
  __global__ void rintf_kernel_v2(Dummy x) { auto result = rintf(x); }
)"};

static constexpr auto kNearbyint{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void nearbyint_kernel_v1(double* x) { auto result = nearbyint(x); }
  __global__ void nearbyint_kernel_v2(Dummy x) { auto result = nearbyint(x); }
  __global__ void nearbyintf_kernel_v1(float* x) { auto result = nearbyintf(x); }
  __global__ void nearbyintf_kernel_v2(Dummy x) { auto result = nearbyintf(x); }
)"};

static constexpr auto kCeil{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void ceil_kernel_v1(double* x) { auto result = ceil(x); }
  __global__ void ceil_kernel_v2(Dummy x) { auto result = ceil(x); }
  __global__ void ceilf_kernel_v1(float* x) { auto result = ceilf(x); }
  __global__ void ceilf_kernel_v2(Dummy x) { auto result = ceilf(x); }
)"};

static constexpr auto kFloor{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void floor_kernel_v1(double* x) { auto result = floor(x); }
  __global__ void floor_kernel_v2(Dummy x) { auto result = floor(x); }
  __global__ void floorf_kernel_v1(float* x) { auto result = floorf(x); }
  __global__ void floorf_kernel_v2(Dummy x) { auto result = floorf(x); }
)"};

static constexpr auto kLrint{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void lrint_kernel_v1(double* x) { auto result = lrint(x); }
  __global__ void lrint_kernel_v2(Dummy x) { auto result = lrint(x); }
  __global__ void lrintf_kernel_v1(float* x) { auto result = lrintf(x); }
  __global__ void lrintf_kernel_v2(Dummy x) { auto result = lrintf(x); }
)"};

static constexpr auto kLround{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void lround_kernel_v1(double* x) { auto result = lround(x); }
  __global__ void lround_kernel_v2(Dummy x) { auto result = lround(x); }
  __global__ void lroundf_kernel_v1(float* x) { auto result = lroundf(x); }
  __global__ void lroundf_kernel_v2(Dummy x) { auto result = lroundf(x); }
)"};

static constexpr auto kLlrint{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void llrint_kernel_v1(double* x) { auto result = llrint(x); }
  __global__ void llrint_kernel_v2(Dummy x) { auto result = llrint(x); }
  __global__ void llrintf_kernel_v1(float* x) { auto result = llrintf(x); }
  __global__ void llrintf_kernel_v2(Dummy x) { auto result = llrintf(x); }
)"};

static constexpr auto kLlround{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void llround_kernel_v1(double* x) { auto result = llround(x); }
  __global__ void llround_kernel_v2(Dummy x) { auto result = llround(x); }
  __global__ void llroundf_kernel_v1(float* x) { auto result = llroundf(x); }
  __global__ void llroundf_kernel_v2(Dummy x) { auto result = llroundf(x); }
)"};

static constexpr auto kFmod{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void fmod_kernel_v1(double* x, double y) { auto result = fmod(x, y); }
  __global__ void fmod_kernel_v2(double x, double* y) { auto result = fmod(x, y); }
  __global__ void fmod_kernel_v3(Dummy x, double y) { auto result = fmod(x, y); }
  __global__ void fmod_kernel_v4(double x, Dummy y) { auto result = fmod(x, y); }
  __global__ void fmodf_kernel_v1(float* x, float y) { auto result = fmodf(x, y); }
  __global__ void fmodf_kernel_v2(float x, float* y) { auto result = fmodf(x, y); }
  __global__ void fmodf_kernel_v3(Dummy x, float y) { auto result = fmodf(x, y); }
  __global__ void fmodf_kernel_v4(float x, Dummy y) { auto result = fmodf(x, y); }
)"};

static constexpr auto kRemainder{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void remainder_kernel_v1(double* x, double y) { auto result = remainder(x, y); }
  __global__ void remainder_kernel_v2(double x, double* y) { auto result = remainder(x, y); }
  __global__ void remainder_kernel_v3(Dummy x, double y) { auto result = remainder(x, y); }
  __global__ void remainder_kernel_v4(double x, Dummy y) { auto result = remainder(x, y); }
  __global__ void remainderf_kernel_v1(float* x, float y) { auto result = remainderf(x, y); }
  __global__ void remainderf_kernel_v2(float x, float* y) { auto result = remainderf(x, y); }
  __global__ void remainderf_kernel_v3(Dummy x, float y) { auto result = remainderf(x, y); }
  __global__ void remainderf_kernel_v4(float x, Dummy y) { auto result = remainderf(x, y); }
)"};

static constexpr auto kRemquo{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void remquo_kernel_v1(double* x, double y, int* quo) { auto result = remquo(x, y, quo); }
  __global__ void remquo_kernel_v2(Dummy x, double y, int* quo) { auto result = remquo(x, y, quo); }
  __global__ void remquo_kernel_v3(double x, double* y, int* quo) { auto result = remquo(x, y, quo); }
  __global__ void remquo_kernel_v4(double x, Dummy y, int* quo) { auto result = remquo(x, y, quo); }
  __global__ void remquo_kernel_v5(double x, double y, char* quo) { auto result = remquo(x, y, quo); }
  __global__ void remquo_kernel_v6(double x, double y, short* quo) {
    auto result = remquo(x, y, quo);
  }
  __global__ void remquo_kernel_v7(double x, double y, long* quo) { auto result = remquo(x, y, quo); }
  __global__ void remquo_kernel_v8(double x, double y, long long* quo) {
    auto result = remquo(x, y, quo);
  }
  __global__ void remquo_kernel_v9(double x, double y, float* quo) {
    auto result = remquo(x, y, quo);
  }
  __global__ void remquo_kernel_v10(double x, double y, double* quo) {
    auto result = remquo(x, y, quo);
  }
  __global__ void remquo_kernel_v11(double x, double y, Dummy* quo) {
    auto result = remquo(x, y, quo);
  }
  __global__ void remquo_kernel_v12(double x, double y, const int* quo) {
    auto result = remquo(x, y, quo);
  }
  __global__ void remquof_kernel_v1(float* x, float y, int* quo) { auto result = remquof(x, y, quo); }
  __global__ void remquof_kernel_v2(Dummy x, float y, int* quo) { auto result = remquof(x, y, quo); }
  __global__ void remquof_kernel_v3(float x, float* y, int* quo) { auto result = remquof(x, y, quo); }
  __global__ void remquof_kernel_v4(float x, Dummy y, int* quo) { auto result = remquof(x, y, quo); }
  __global__ void remquof_kernel_v5(float x, float y, char* quo) { auto result = remquof(x, y, quo); }
  __global__ void remquof_kernel_v6(float x, float y, short* quo) {
    auto result = remquof(x, y, quo);
  }
  __global__ void remquof_kernel_v7(float x, float y, long* quo) { auto result = remquof(x, y, quo); }
  __global__ void remquof_kernel_v8(float x, float y, long long* quo) {
    auto result = remquof(x, y, quo);
  }
  __global__ void remquof_kernel_v9(float x, float y, float* quo) {
    auto result = remquof(x, y, quo);
  }
  __global__ void remquof_kernel_v10(float x, float y, double* quo) {
    auto result = remquof(x, y, quo);
  }
  __global__ void remquof_kernel_v11(float x, float y, Dummy* quo) {
    auto result = remquof(x, y, quo);
  }
  __global__ void remquof_kernel_v12(float x, float y, const int* quo) {
    auto result = remquof(x, y, quo);
  }
)"};

static constexpr auto kModf{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void modf_kernel_v1(double* x, double* iptr) { auto result = modf(x, iptr); }
  __global__ void modf_kernel_v2(Dummy x, double* iptr) { auto result = modf(x, iptr); }
  __global__ void modf_kernel_v3(double x, int* iptr) { auto result = modf(x, iptr); }
  __global__ void modf_kernel_v4(double x, char* iptr) { auto result = modf(x, iptr); }
  __global__ void modf_kernel_v5(double x, short* iptr) { auto result = modf(x, iptr); }
  __global__ void modf_kernel_v6(double x, long* iptr) { auto result = modf(x, iptr); }
  __global__ void modf_kernel_v7(double x, long long* iptr) { auto result = modf(x, iptr); }
  __global__ void modf_kernel_v8(double x, float* iptr) { auto result = modf(x, iptr); }
  __global__ void modf_kernel_v9(double x, Dummy* iptr) { auto result = modf(x, iptr); }
  __global__ void modf_kernel_v10(double x, const double* iptr) { auto result = modf(x, iptr); }
  __global__ void modff_kernel_v1(float* x, float* iptr) { auto result = modff(x, iptr); }
  __global__ void modff_kernel_v2(Dummy x, float* iptr) { auto result = modff(x, iptr); }
  __global__ void modff_kernel_v3(float x, int* iptr) { auto result = modff(x, iptr); }
  __global__ void modff_kernel_v4(float x, char* iptr) { auto result = modff(x, iptr); }
  __global__ void modff_kernel_v5(float x, short* iptr) { auto result = modff(x, iptr); }
  __global__ void modff_kernel_v6(float x, long* iptr) { auto result = modff(x, iptr); }
  __global__ void modff_kernel_v7(float x, long long* iptr) { auto result = modff(x, iptr); }
  __global__ void modff_kernel_v8(float x, double* iptr) { auto result = modff(x, iptr); }
  __global__ void modff_kernel_v9(float x, Dummy* iptr) { auto result = modff(x, iptr); }
  __global__ void modff_kernel_v10(float x, const float* iptr) { auto result = modff(x, iptr); }
)"};

static constexpr auto kFdim{R"(
  class Dummy {
   public:
    __device__ Dummy() {}
    __device__ ~Dummy() {}
  };
  __global__ void fdim_kernel_v1(double* x, double y) { auto result = fdim(x, y); }
  __global__ void fdim_kernel_v2(double x, double* y) { auto result = fdim(x, y); }
  __global__ void fdim_kernel_v3(Dummy x, double y) { auto result = fdim(x, y); }
  __global__ void fdim_kernel_v4(double x, Dummy y) { auto result = fdim(x, y); }
  __global__ void fdimf_kernel_v1(float* x, float y) { auto result = fdimf(x, y); }
  __global__ void fdimf_kernel_v2(float x, float* y) { auto result = fdimf(x, y); }
  __global__ void fdimf_kernel_v3(Dummy x, float y) { auto result = fdimf(x, y); }
  __global__ void fdimf_kernel_v4(float x, Dummy y) { auto result = fdimf(x, y); }
)"};
