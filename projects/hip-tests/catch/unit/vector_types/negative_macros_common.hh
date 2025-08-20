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
