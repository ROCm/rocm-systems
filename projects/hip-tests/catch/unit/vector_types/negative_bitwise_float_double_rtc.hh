/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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
