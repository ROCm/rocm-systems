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

#pragma once

#include <hip/hip_fp16.h>

#define FLOAT16_MAX 65504.0f

class Float16 {
 public:
  __host__ __device__ Float16() = default;
  __host__ __device__ Float16(__half x) : x_{x} {}
  __host__ __device__ Float16(__half2 x) : x_{__low2half(x)} {}
  __host__ __device__ Float16(float x) : x_{__float2half(x)} {}

  // __heq doesn't have a __host__ version
  __host__ __device__ bool operator==(Float16 other) const { return (static_cast<__half_raw>(x_).x == static_cast<__half_raw>(other.x_).x); }
  __host__ __device__ bool operator!=(Float16 other) const { return !(*this == other); }

  __host__ __device__ operator __half() const { return x_; }
  __host__ __device__ operator __half2() const { return __half2half2(x_); }
  __host__ __device__ operator float() const { return __half2float(x_); }

 private:
  __half x_;
};

namespace {

inline std::ostream& operator<<(std::ostream& o, Float16 x) {
  o << static_cast<float>(x);
  return o;
}

}  // namespace