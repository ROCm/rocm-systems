/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#define NEGATE_UNSIGNED_VECTOR_FUNCTIONS(type)                                                     \
  __global__ void NegateDevice(type* vector_dev_ptr) {                                             \
    type vector_dev = *vector_dev_ptr;                                                             \
    vector_dev = -vector_dev;                                                                      \
  }                                                                                                \
  void NegateHost(type& vector_host) { vector_host = -vector_host; }

#define BITWISE_FLOATING_POINT_VECTOR_FUNCTIONS(type)                                              \
  __global__ void BitwiseDevice(type* vector1_dev_ptr, type* vector2_dev_ptr) {                    \
    type vector1_dev = *vector1_dev_ptr;                                                           \
    type vector2_dev = *vector2_dev_ptr;                                                           \
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
  void BitwiseHost(type& vector1_host, type& vector2_host) {                                       \
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

#define CALCULATE_ASSIGN_VECTOR_FUNCTIONS(type)                                                    \
  __global__ void CalculateAssignDevice(type* vector_dev_ptr, decltype(type().x) value) {          \
    type vector_dev = *vector_dev_ptr;                                                             \
    vector_dev %= value;                                                                           \
    vector_dev ^= value;                                                                           \
    vector_dev |= value;                                                                           \
    vector_dev &= value;                                                                           \
    vector_dev >>= value;                                                                          \
    vector_dev <<= value;                                                                          \
  }                                                                                                \
  void CalculateAssignHost(type& vector_host, decltype(type().x) value) {                          \
    vector_host %= value;                                                                          \
    vector_host ^= value;                                                                          \
    vector_host |= value;                                                                          \
    vector_host &= value;                                                                          \
    vector_host >>= value;                                                                         \
    vector_host <<= value;                                                                         \
  }
