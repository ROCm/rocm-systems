/**
 * MIT License
 *
 * Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*!\file
 * \brief hip_bfloat16.h provides struct for hip_bfloat16 typedef
 */

#ifndef _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_BFLOAT16_H_
#define _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_BFLOAT16_H_

#include "host_defines.h"
#if defined(__HIPCC_RTC__)
#define __HOST_DEVICE__ __device__
#else
#define __HOST_DEVICE__ __host__ __device__
#endif

#if __cplusplus < 201103L || !defined(__HIPCC__)

// If this is a C compiler, C++ compiler below C++11, or a host-only compiler, we only
// include a minimal definition of hip_bfloat16

static_assert(sizeof(unsigned short) == 2, "size of unsigned short should be 2 bytes");
static_assert(sizeof(__bf16) == sizeof(unsigned short));

#include <stdint.h>
/*! \brief Struct to represent a 16 bit brain floating point number. */
typedef struct {
  unsigned short data;
} hip_bfloat16;

#else  // __cplusplus < 201103L || !defined(__HIPCC__)

#include <hip/hip_runtime.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow"
struct hip_bfloat16 {
  union {
    unsigned short data;
    __bf16 data_bf16;
  };

  enum truncate_t { truncate };

  __HOST_DEVICE__ hip_bfloat16() = default;

  // round upper 16 bits of IEEE float to convert to bfloat16
  explicit __HOST_DEVICE__ hip_bfloat16(float f) : data_bf16(f) {}

  explicit __HOST_DEVICE__ hip_bfloat16(float f, truncate_t)
      : data(truncate_float_to_bfloat16(f)) {}

  // zero extend lower 16 bits of bfloat16 to convert to IEEE float
  __HOST_DEVICE__ operator float() const { return static_cast<float>(data_bf16); }

  __HOST_DEVICE__ hip_bfloat16& operator=(const float& f) {
    data_bf16 = f;
    return *this;
  }

  static __HOST_DEVICE__ hip_bfloat16 round_to_bfloat16(float f) {
    hip_bfloat16 output;
    output.data_bf16 = f;
    return output;
  }

  static __HOST_DEVICE__ hip_bfloat16 round_to_bfloat16(float f, truncate_t) {
    hip_bfloat16 output;
    output.data = truncate_float_to_bfloat16(f);
    return output;
  }

 private:
  // Truncate instead of rounding, preserving SNaN
  static __HOST_DEVICE__ __hip_uint16_t truncate_float_to_bfloat16(float f) {
    union {
      float fp32;
      __hip_uint32_t int32;
    } u = {f};
    return __hip_uint16_t(u.int32 >> 16) | (!(~u.int32 & 0x7f800000) && (u.int32 & 0xffff));
  }
};
#pragma clang diagnostic pop

typedef struct {
  unsigned short data;
} hip_bfloat16_public;

static_assert(__hip_internal::is_standard_layout<hip_bfloat16>{},
              "hip_bfloat16 is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(__hip_internal::is_trivial<hip_bfloat16>{},
              "hip_bfloat16 is not a trivial type, and thus is "
              "incompatible with C.");
#if !defined(__HIPCC_RTC__)
static_assert(sizeof(hip_bfloat16) == sizeof(hip_bfloat16_public) &&
                  offsetof(hip_bfloat16, data) == offsetof(hip_bfloat16_public, data),
              "internal hip_bfloat16 does not match public hip_bfloat16");

inline std::ostream& operator<<(std::ostream& os, const hip_bfloat16& bf16) {
  return os << float(bf16);
}
#endif

inline __HOST_DEVICE__ hip_bfloat16 operator+(hip_bfloat16 a) { return a; }
inline __HOST_DEVICE__ hip_bfloat16 operator-(hip_bfloat16 a) {
  a.data ^= 0x8000;
  return a;
}
inline __HOST_DEVICE__ hip_bfloat16 operator+(hip_bfloat16 a, hip_bfloat16 b) {
  hip_bfloat16 ret;
  ret.data_bf16 = a.data_bf16 + b.data_bf16;
  return ret;
}
inline __HOST_DEVICE__ hip_bfloat16 operator-(hip_bfloat16 a, hip_bfloat16 b) {
  hip_bfloat16 ret;
  ret.data_bf16 = a.data_bf16 - b.data_bf16;
  return ret;
}
inline __HOST_DEVICE__ hip_bfloat16 operator*(hip_bfloat16 a, hip_bfloat16 b) {
  hip_bfloat16 ret;
  ret.data_bf16 = a.data_bf16 * b.data_bf16;
  return ret;
}
inline __HOST_DEVICE__ hip_bfloat16 operator/(hip_bfloat16 a, hip_bfloat16 b) {
  hip_bfloat16 ret;
  ret.data_bf16 = a.data_bf16 / b.data_bf16;
  return ret;
}
inline __HOST_DEVICE__ bool operator<(hip_bfloat16 a, hip_bfloat16 b) {
  hip_bfloat16 ret;
  return a.data_bf16 < b.data_bf16;
}
inline __HOST_DEVICE__ bool operator==(hip_bfloat16 a, hip_bfloat16 b) {
  hip_bfloat16 ret;
  return a.data_bf16 == b.data_bf16;
}
inline __HOST_DEVICE__ bool operator>(hip_bfloat16 a, hip_bfloat16 b) { return b < a; }
inline __HOST_DEVICE__ bool operator<=(hip_bfloat16 a, hip_bfloat16 b) { return !(a > b); }
inline __HOST_DEVICE__ bool operator!=(hip_bfloat16 a, hip_bfloat16 b) { return !(a == b); }
inline __HOST_DEVICE__ bool operator>=(hip_bfloat16 a, hip_bfloat16 b) { return !(a < b); }
inline __HOST_DEVICE__ hip_bfloat16& operator+=(hip_bfloat16& a, hip_bfloat16 b) {
  return a = a + b;
}
inline __HOST_DEVICE__ hip_bfloat16& operator-=(hip_bfloat16& a, hip_bfloat16 b) {
  return a = a - b;
}
inline __HOST_DEVICE__ hip_bfloat16& operator*=(hip_bfloat16& a, hip_bfloat16 b) {
  return a = a * b;
}
inline __HOST_DEVICE__ hip_bfloat16& operator/=(hip_bfloat16& a, hip_bfloat16 b) {
  return a = a / b;
}
inline __HOST_DEVICE__ hip_bfloat16& operator++(hip_bfloat16& a) { return a += hip_bfloat16(1.0f); }
inline __HOST_DEVICE__ hip_bfloat16& operator--(hip_bfloat16& a) { return a -= hip_bfloat16(1.0f); }
inline __HOST_DEVICE__ hip_bfloat16 operator++(hip_bfloat16& a, int) {
  hip_bfloat16 orig = a;
  ++a;
  return orig;
}
inline __HOST_DEVICE__ hip_bfloat16 operator--(hip_bfloat16& a, int) {
  hip_bfloat16 orig = a;
  --a;
  return orig;
}

namespace std {
constexpr __HOST_DEVICE__ bool isinf(hip_bfloat16 a) {
  return !(~a.data & 0x7f80) && !(a.data & 0x7f);
}
constexpr __HOST_DEVICE__ bool isnan(hip_bfloat16 a) {
  return !(~a.data & 0x7f80) && +(a.data & 0x7f);
}
constexpr __HOST_DEVICE__ bool iszero(hip_bfloat16 a) { return !(a.data & 0x7fff); }
}  // namespace std

#endif  // __cplusplus < 201103L || !defined(__HIPCC__)

#endif  // _HIP_BFLOAT16_H_
