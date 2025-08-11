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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


static constexpr auto kFabs{R"(
class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};
__global__ void fabsf_kernel_v1(float* x) { float result = fabsf(x); }
__global__ void fabsf_kernel_v2(Dummy x) { float result = fabsf(x); }
__global__ void fabs_kernel_v1(double* x) { double result = fabs(x); }
__global__ void fabs_kernel_v2(Dummy x) { double result = fabs(x); }
)"};

static constexpr auto kCopySign{R"(
class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};
__global__ void copysignf_kernel_v1(float* x, float y) { float result = copysignf(x, y); }
__global__ void copysignf_kernel_v2(Dummy x, float y) { float result = copysignf(x, y); }
__global__ void copysignf_kernel_v3(float x, float* y) { float result = copysignf(x, y); }
__global__ void copysignf_kernel_v4(float x, Dummy y) { float result = copysignf(x, y); }
__global__ void copysign_kernel_v1(double* x, double y) { double result = copysign(x, y); }
__global__ void copysign_kernel_v2(Dummy x, double y) { double result = copysign(x, y); }
__global__ void copysign_kernel_v3(double x, double* y) { double result = copysign(x, y); }
__global__ void copysign_kernel_v4(double x, Dummy y) { double result = copysign(x, y); }
)"};

static constexpr auto kFmax{R"(
class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};
__global__ void fmaxf_kernel_v1(float* x, float y) { float result = fmaxf(x, y); }
__global__ void fmaxf_kernel_v2(Dummy x, float y) { float result = fmaxf(x, y); }
__global__ void fmaxf_kernel_v3(float x, float* y) { float result = fmaxf(x, y); }
__global__ void fmaxf_kernel_v4(float x, Dummy y) { float result = fmaxf(x, y); }
__global__ void fmax_kernel_v1(double* x, double y) { double result = fmax(x, y); }
__global__ void fmax_kernel_v2(Dummy x, double y) { double result = fmax(x, y); }
__global__ void fmax_kernel_v3(double x, double* y) { double result = fmax(x, y); }
__global__ void fmax_kernel_v4(double x, Dummy y) { double result = fmax(x, y); }
)"};

static constexpr auto kFmin{R"(
class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};
__global__ void fminf_kernel_v1(float* x, float y) { float result = fminf(x, y); }
__global__ void fminf_kernel_v2(Dummy x, float y) { float result = fminf(x, y); }
__global__ void fminf_kernel_v3(float x, float* y) { float result = fminf(x, y); }
__global__ void fminf_kernel_v4(float x, Dummy y) { float result = fminf(x, y); }
__global__ void fmin_kernel_v1(double* x, double y) { double result = fmin(x, y); }
__global__ void fmin_kernel_v2(Dummy x, double y) { double result = fmin(x, y); }
__global__ void fmin_kernel_v3(double x, double* y) { double result = fmin(x, y); }
__global__ void fmin_kernel_v4(double x, Dummy y) { double result = fmin(x, y); }
)"};

static constexpr auto kNextAfter{R"(
class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};
__global__ void nextafterf_kernel_v1(float* x, float y) { float result = nextafterf(x, y); }
__global__ void nextafterf_kernel_v2(Dummy x, float y) { float result = nextafterf(x, y); }
__global__ void nextafterf_kernel_v3(float x, float* y) { float result = nextafterf(x, y); }
__global__ void nextafterf_kernel_v4(float x, Dummy y) { float result = nextafterf(x, y); }
__global__ void nextafter_kernel_v1(double* x, double y) { double result = nextafter(x, y); }
__global__ void nextafter_kernel_v2(Dummy x, double y) { double result = nextafter(x, y); }
__global__ void nextafter_kernel_v3(double x, double* y) { double result = nextafter(x, y); }
__global__ void nextafter_kernel_v4(double x, Dummy y) { double result = nextafter(x, y); }
)"};

static constexpr auto kFma{R"(
class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};
__global__ void fmaf_kernel_v1(float* x, float y, float z) { float result = fmaf(x, y, z); }
__global__ void fmaf_kernel_v2(Dummy x, float y, float z) { float result = fmaf(x, y, z); }
__global__ void fmaf_kernel_v3(float x, float* y, float z) { float result = fmaf(x, y, z); }
__global__ void fmaf_kernel_v4(float x, Dummy y, float z) { float result = fmaf(x, y, z); }
__global__ void fmaf_kernel_v5(float x, float y, float* z) { float result = fmaf(x, y, z); }
__global__ void fmaf_kernel_v6(float x, float y, Dummy z) { float result = fmaf(x, y, z); }
__global__ void fma_kernel_v1(double* x, double y, double z) { double result = fmaf(x, y, z); }
__global__ void fma_kernel_v2(Dummy x, double y, double z) { double result = fmaf(x, y, z); }
__global__ void fma_kernel_v3(double x, double* y, double z) { double result = fmaf(x, y, z); }
__global__ void fma_kernel_v4(double x, Dummy y, double z) { double result = fmaf(x, y, z); }
__global__ void fma_kernel_v5(double x, double y, double* z) { double result = fmaf(x, y, z); }
__global__ void fma_kernel_v6(double x, double y, Dummy z) { double result = fmaf(x, y, z); }
)"};

static constexpr auto kFdividef{R"(
class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};
__global__ void fdividef_kernel_v1(float* x, float y) { float result = fdividef(x, y); }
__global__ void fdividef_kernel_v2(Dummy x, float y) { float result = fdivide(x); }
__global__ void fdividef_kernel_v3(float x, float* y) { float result = fdivide(x); }
__global__ void fdividef_kernel_v4(float x, Dummy y) { float result = fdivide(x); }
)"};

static constexpr auto kIsFinite{R"(
class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};
__global__ void isfinite_kernel_v1(float* x) { bool result = isfinite(x); }
__global__ void isfinite_kernel_v2(Dummy x) { bool result = isfinite(x); }
__global__ void isfinite_kernel_v3(double* x) { bool result = isfinite(x); }
__global__ void isfinite_kernel_v4(Dummy x) { bool result = isfinite(x); }
)"};

static constexpr auto kIsInf{R"(
class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};
__global__ void isinf_kernel_v1(float* x) { bool result = isinf(x); }
__global__ void isinf_kernel_v2(Dummy x) { bool result = isinf(x); }
__global__ void isinf_kernel_v3(double* x) { bool result = isinf(x); }
__global__ void isinf_kernel_v4(Dummy x) { bool result = isinf(x); }
)"};

static constexpr auto kIsNan{R"(
class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};
__global__ void isnan_kernel_v1(float* x) { bool result = isnan(x); }
__global__ void isnan_kernel_v2(Dummy x) { bool result = isnan(x); }
__global__ void isnan_kernel_v3(double* x) { bool result = isnan(x); }
__global__ void isnan_kernel_v4(Dummy x) { bool result = isnan(x); }
)"};

static constexpr auto kSignBit{R"(
class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};
__global__ void signbit_kernel_v1(float* x) { bool result = signbit(x); }
__global__ void signbit_kernel_v2(Dummy x) { bool result = signbit(x); }
__global__ void signbit_kernel_v3(double* x) { bool result = signbit(x); }
__global__ void signbit_kernel_v4(Dummy x) { bool result = signbit(x); }
)"};
