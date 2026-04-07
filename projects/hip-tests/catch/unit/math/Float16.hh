/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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