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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>

static constexpr auto kBitwiseFloat{R"(
  __global__ void BitwiseDevice(float1* vector1_dev_ptr, float1* vector2_dev_ptr) {                    \
    float1 vector1_dev = *vector1_dev_ptr;                                                           \
    float1 vector2_dev = *vector2_dev_ptr;                                                           \
    vector1_dev = ~vector1_dev;                                                                    \
    vector1_dev %= vector2_dev;                                                                    \
    vector1_dev ^= vector2_dev;                                                                    \
    vector1_dev |= vector2_dev;                                                                    \
    vector1_dev &= vector2_dev;                                                                    \
    vector1_dev >>= vector2_dev;                                                                   \
    vector1_dev <<= vector2_dev;                                                                   \
    vector1_dev = vector1_dev ^ vector2_dev;                                                       \
    vector1_dev = vector1_dev | vector2_dev;                                                       \
    vector1_dev = vector1_dev & vector2_dev;                                                       \
    vector1_dev = vector1_dev >> vector2_dev;                                                      \
    vector1_dev = vector1_dev << vector2_dev;                                                      \
  }                                                                                                \
  void BitwiseHost(float1& vector1_host, float1& vector2_host) {                                       \
    vector1_host = ~vector1_host;                                                                  \
    vector1_host %= vector2_host;                                                                  \
    vector1_host ^= vector2_host;                                                                  \
    vector1_host |= vector2_host;                                                                  \
    vector1_host &= vector2_host;                                                                  \
    vector1_host >>= vector2_host;                                                                 \
    vector1_host <<= vector2_host;                                                                 \
    vector1_host = vector1_host ^ vector2_host;                                                    \
    vector1_host = vector1_host | vector2_host;                                                    \
    vector1_host = vector1_host & vector2_host;                                                    \
    vector1_host = vector1_host >> vector2_host;                                                   \
    vector1_host = vector1_host << vector2_host;                                                   \
  }

  __global__ void BitwiseDevice(float2* vector1_dev_ptr, float2* vector2_dev_ptr) {                    \
    float2 vector1_dev = *vector1_dev_ptr;                                                           \
    float2 vector2_dev = *vector2_dev_ptr;                                                           \
    vector1_dev = ~vector1_dev;                                                                    \
    vector1_dev %= vector2_dev;                                                                    \
    vector1_dev ^= vector2_dev;                                                                    \
    vector1_dev |= vector2_dev;                                                                    \
    vector1_dev &= vector2_dev;                                                                    \
    vector1_dev >>= vector2_dev;                                                                   \
    vector1_dev <<= vector2_dev;                                                                   \
    vector1_dev = vector1_dev ^ vector2_dev;                                                       \
    vector1_dev = vector1_dev | vector2_dev;                                                       \
    vector1_dev = vector1_dev & vector2_dev;                                                       \
    vector1_dev = vector1_dev >> vector2_dev;                                                      \
    vector1_dev = vector1_dev << vector2_dev;                                                      \
  }                                                                                                \
  void BitwiseHost(float2& vector1_host, float2& vector2_host) {                                       \
    vector1_host = ~vector1_host;                                                                  \
    vector1_host %= vector2_host;                                                                  \
    vector1_host ^= vector2_host;                                                                  \
    vector1_host |= vector2_host;                                                                  \
    vector1_host &= vector2_host;                                                                  \
    vector1_host >>= vector2_host;                                                                 \
    vector1_host <<= vector2_host;                                                                 \
    vector1_host = vector1_host ^ vector2_host;                                                    \
    vector1_host = vector1_host | vector2_host;                                                    \
    vector1_host = vector1_host & vector2_host;                                                    \
    vector1_host = vector1_host >> vector2_host;                                                   \
    vector1_host = vector1_host << vector2_host;                                                   \
  }

  __global__ void BitwiseDevice(float3* vector1_dev_ptr, float3* vector2_dev_ptr) {                    \
    float3 vector1_dev = *vector1_dev_ptr;                                                           \
    float3 vector2_dev = *vector2_dev_ptr;                                                           \
    vector1_dev = ~vector1_dev;                                                                    \
    vector1_dev %= vector2_dev;                                                                    \
    vector1_dev ^= vector2_dev;                                                                    \
    vector1_dev |= vector2_dev;                                                                    \
    vector1_dev &= vector2_dev;                                                                    \
    vector1_dev >>= vector2_dev;                                                                   \
    vector1_dev <<= vector2_dev;                                                                   \
    vector1_dev = vector1_dev ^ vector2_dev;                                                       \
    vector1_dev = vector1_dev | vector2_dev;                                                       \
    vector1_dev = vector1_dev & vector2_dev;                                                       \
    vector1_dev = vector1_dev >> vector2_dev;                                                      \
    vector1_dev = vector1_dev << vector2_dev;                                                      \
  }                                                                                                \
  void BitwiseHost(float3& vector1_host, float3& vector2_host) {                                       \
    vector1_host = ~vector1_host;                                                                  \
    vector1_host %= vector2_host;                                                                  \
    vector1_host ^= vector2_host;                                                                  \
    vector1_host |= vector2_host;                                                                  \
    vector1_host &= vector2_host;                                                                  \
    vector1_host >>= vector2_host;                                                                 \
    vector1_host <<= vector2_host;                                                                 \
    vector1_host = vector1_host ^ vector2_host;                                                    \
    vector1_host = vector1_host | vector2_host;                                                    \
    vector1_host = vector1_host & vector2_host;                                                    \
    vector1_host = vector1_host >> vector2_host;                                                   \
    vector1_host = vector1_host << vector2_host;                                                   \
  }

  __global__ void BitwiseDevice(float4* vector1_dev_ptr, float4* vector2_dev_ptr) {                    \
    float4 vector1_dev = *vector1_dev_ptr;                                                           \
    float4 vector2_dev = *vector2_dev_ptr;                                                           \
    vector1_dev = ~vector1_dev;                                                                    \
    vector1_dev %= vector2_dev;                                                                    \
    vector1_dev ^= vector2_dev;                                                                    \
    vector1_dev |= vector2_dev;                                                                    \
    vector1_dev &= vector2_dev;                                                                    \
    vector1_dev >>= vector2_dev;                                                                   \
    vector1_dev <<= vector2_dev;                                                                   \
    vector1_dev = vector1_dev ^ vector2_dev;                                                       \
    vector1_dev = vector1_dev | vector2_dev;                                                       \
    vector1_dev = vector1_dev & vector2_dev;                                                       \
    vector1_dev = vector1_dev >> vector2_dev;                                                      \
    vector1_dev = vector1_dev << vector2_dev;                                                      \
  }                                                                                                \
  void BitwiseHost(float4& vector1_host, float4& vector2_host) {                                       \
    vector1_host = ~vector1_host;                                                                  \
    vector1_host %= vector2_host;                                                                  \
    vector1_host ^= vector2_host;                                                                  \
    vector1_host |= vector2_host;                                                                  \
    vector1_host &= vector2_host;                                                                  \
    vector1_host >>= vector2_host;                                                                 \
    vector1_host <<= vector2_host;                                                                 \
    vector1_host = vector1_host ^ vector2_host;                                                    \
    vector1_host = vector1_host | vector2_host;                                                    \
    vector1_host = vector1_host & vector2_host;                                                    \
    vector1_host = vector1_host >> vector2_host;                                                   \
    vector1_host = vector1_host << vector2_host;                                                   \
  }
)"};

static constexpr auto kBitwiseDouble{R"(
  __global__ void BitwiseDevice(double1* vector1_dev_ptr, double1* vector2_dev_ptr) {                    \
    double1 vector1_dev = *vector1_dev_ptr;                                                           \
    double1 vector2_dev = *vector2_dev_ptr;                                                           \
    vector1_dev = ~vector1_dev;                                                                    \
    vector1_dev %= vector2_dev;                                                                    \
    vector1_dev ^= vector2_dev;                                                                    \
    vector1_dev |= vector2_dev;                                                                    \
    vector1_dev &= vector2_dev;                                                                    \
    vector1_dev >>= vector2_dev;                                                                   \
    vector1_dev <<= vector2_dev;                                                                   \
    vector1_dev = vector1_dev ^ vector2_dev;                                                       \
    vector1_dev = vector1_dev | vector2_dev;                                                       \
    vector1_dev = vector1_dev & vector2_dev;                                                       \
    vector1_dev = vector1_dev >> vector2_dev;                                                      \
    vector1_dev = vector1_dev << vector2_dev;                                                      \
  }                                                                                                \
  void BitwiseHost(double1& vector1_host, double1& vector2_host) {                                       \
    vector1_host = ~vector1_host;                                                                  \
    vector1_host %= vector2_host;                                                                  \
    vector1_host ^= vector2_host;                                                                  \
    vector1_host |= vector2_host;                                                                  \
    vector1_host &= vector2_host;                                                                  \
    vector1_host >>= vector2_host;                                                                 \
    vector1_host <<= vector2_host;                                                                 \
    vector1_host = vector1_host ^ vector2_host;                                                    \
    vector1_host = vector1_host | vector2_host;                                                    \
    vector1_host = vector1_host & vector2_host;                                                    \
    vector1_host = vector1_host >> vector2_host;                                                   \
    vector1_host = vector1_host << vector2_host;                                                   \
  }

  __global__ void BitwiseDevice(double2* vector1_dev_ptr, double2* vector2_dev_ptr) {                    \
    double2 vector1_dev = *vector1_dev_ptr;                                                           \
    double2 vector2_dev = *vector2_dev_ptr;                                                           \
    vector1_dev = ~vector1_dev;                                                                    \
    vector1_dev %= vector2_dev;                                                                    \
    vector1_dev ^= vector2_dev;                                                                    \
    vector1_dev |= vector2_dev;                                                                    \
    vector1_dev &= vector2_dev;                                                                    \
    vector1_dev >>= vector2_dev;                                                                   \
    vector1_dev <<= vector2_dev;                                                                   \
    vector1_dev = vector1_dev ^ vector2_dev;                                                       \
    vector1_dev = vector1_dev | vector2_dev;                                                       \
    vector1_dev = vector1_dev & vector2_dev;                                                       \
    vector1_dev = vector1_dev >> vector2_dev;                                                      \
    vector1_dev = vector1_dev << vector2_dev;                                                      \
  }                                                                                                \
  void BitwiseHost(double2& vector1_host, double2& vector2_host) {                                       \
    vector1_host = ~vector1_host;                                                                  \
    vector1_host %= vector2_host;                                                                  \
    vector1_host ^= vector2_host;                                                                  \
    vector1_host |= vector2_host;                                                                  \
    vector1_host &= vector2_host;                                                                  \
    vector1_host >>= vector2_host;                                                                 \
    vector1_host <<= vector2_host;                                                                 \
    vector1_host = vector1_host ^ vector2_host;                                                    \
    vector1_host = vector1_host | vector2_host;                                                    \
    vector1_host = vector1_host & vector2_host;                                                    \
    vector1_host = vector1_host >> vector2_host;                                                   \
    vector1_host = vector1_host << vector2_host;                                                   \
  }

  __global__ void BitwiseDevice(double3* vector1_dev_ptr, double3* vector2_dev_ptr) {                    \
    double3 vector1_dev = *vector1_dev_ptr;                                                           \
    double3 vector2_dev = *vector2_dev_ptr;                                                           \
    vector1_dev = ~vector1_dev;                                                                    \
    vector1_dev %= vector2_dev;                                                                    \
    vector1_dev ^= vector2_dev;                                                                    \
    vector1_dev |= vector2_dev;                                                                    \
    vector1_dev &= vector2_dev;                                                                    \
    vector1_dev >>= vector2_dev;                                                                   \
    vector1_dev <<= vector2_dev;                                                                   \
    vector1_dev = vector1_dev ^ vector2_dev;                                                       \
    vector1_dev = vector1_dev | vector2_dev;                                                       \
    vector1_dev = vector1_dev & vector2_dev;                                                       \
    vector1_dev = vector1_dev >> vector2_dev;                                                      \
    vector1_dev = vector1_dev << vector2_dev;                                                      \
  }                                                                                                \
  void BitwiseHost(double3& vector1_host, double3& vector2_host) {                                       \
    vector1_host = ~vector1_host;                                                                  \
    vector1_host %= vector2_host;                                                                  \
    vector1_host ^= vector2_host;                                                                  \
    vector1_host |= vector2_host;                                                                  \
    vector1_host &= vector2_host;                                                                  \
    vector1_host >>= vector2_host;                                                                 \
    vector1_host <<= vector2_host;                                                                 \
    vector1_host = vector1_host ^ vector2_host;                                                    \
    vector1_host = vector1_host | vector2_host;                                                    \
    vector1_host = vector1_host & vector2_host;                                                    \
    vector1_host = vector1_host >> vector2_host;                                                   \
    vector1_host = vector1_host << vector2_host;                                                   \
  }

  __global__ void BitwiseDevice(double4* vector1_dev_ptr, double4* vector2_dev_ptr) {                    \
    double4 vector1_dev = *vector1_dev_ptr;                                                           \
    double4 vector2_dev = *vector2_dev_ptr;                                                           \
    vector1_dev = ~vector1_dev;                                                                    \
    vector1_dev %= vector2_dev;                                                                    \
    vector1_dev ^= vector2_dev;                                                                    \
    vector1_dev |= vector2_dev;                                                                    \
    vector1_dev &= vector2_dev;                                                                    \
    vector1_dev >>= vector2_dev;                                                                   \
    vector1_dev <<= vector2_dev;                                                                   \
    vector1_dev = vector1_dev ^ vector2_dev;                                                       \
    vector1_dev = vector1_dev | vector2_dev;                                                       \
    vector1_dev = vector1_dev & vector2_dev;                                                       \
    vector1_dev = vector1_dev >> vector2_dev;                                                      \
    vector1_dev = vector1_dev << vector2_dev;                                                      \
  }                                                                                                \
  void BitwiseHost(double4& vector1_host, double4& vector2_host) {                                       \
    vector1_host = ~vector1_host;                                                                  \
    vector1_host %= vector2_host;                                                                  \
    vector1_host ^= vector2_host;                                                                  \
    vector1_host |= vector2_host;                                                                  \
    vector1_host &= vector2_host;                                                                  \
    vector1_host >>= vector2_host;                                                                 \
    vector1_host <<= vector2_host;                                                                 \
    vector1_host = vector1_host ^ vector2_host;                                                    \
    vector1_host = vector1_host | vector2_host;                                                    \
    vector1_host = vector1_host & vector2_host;                                                    \
    vector1_host = vector1_host >> vector2_host;                                                   \
    vector1_host = vector1_host << vector2_host;                                                   \
  }
)"};
