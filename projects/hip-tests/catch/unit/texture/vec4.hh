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

#include <hip/hip_runtime_api.h>

template <typename T> struct vec4_struct { using type = void; };

#define DEFINE_VEC4_OVERLOAD(base_type, vec_type)                                                  \
  template <> struct vec4_struct<base_type> { using type = vec_type; }

DEFINE_VEC4_OVERLOAD(char, char4);
DEFINE_VEC4_OVERLOAD(short, short4);
DEFINE_VEC4_OVERLOAD(int, int4);
DEFINE_VEC4_OVERLOAD(long, long4);
DEFINE_VEC4_OVERLOAD(long long, longlong4);

DEFINE_VEC4_OVERLOAD(unsigned char, uchar4);
DEFINE_VEC4_OVERLOAD(unsigned short, ushort4);
DEFINE_VEC4_OVERLOAD(unsigned int, uint4);
DEFINE_VEC4_OVERLOAD(unsigned long, ulong4);
DEFINE_VEC4_OVERLOAD(unsigned long long, ulonglong4);

DEFINE_VEC4_OVERLOAD(float, float4);
DEFINE_VEC4_OVERLOAD(double, float4);

template <typename T> using vec4 = typename vec4_struct<T>::type;

template <typename T> inline void SetVec4(vec4<T>& vec, const T val) {
  vec.x = val;
  vec.y = val;
  vec.z = val;
  vec.w = val;
}

template <typename T>
inline void SetVec4(vec4<T>& vec, const T x, const T y, const T z, const T w) {
  vec.x = x;
  vec.y = y;
  vec.z = z;
  vec.w = w;
}

template <typename T> inline auto MakeVec4(const T val) {
  vec4<T> vec;
  SetVec4(vec, val);

  return vec;
}

template <typename T> inline void MakeVec4(const T x, const T y, const T z, const T w) {
  vec4<T> vec;
  SetVec4(vec, x, y, z, w);

  return vec;
}

template <typename T> inline vec4<float> Vec4Map(const T& vec) {
  vec4<float> ret;
  ret.x = NormalizeInteger(vec.x);
  ret.y = NormalizeInteger(vec.y);
  ret.z = NormalizeInteger(vec.z);
  ret.w = NormalizeInteger(vec.w);
  return ret;
}

template <typename T> inline __host__ __device__ auto Vec4Scale(float s, const T& vec) {
  T ret;
  ret.x = s * vec.x;
  ret.y = s * vec.y;
  ret.z = s * vec.z;
  ret.w = s * vec.w;

  return ret;
}

template <typename T> inline __host__ __device__ auto Vec4Add(const T& vec1, const T& vec2) {
  T ret;
  ret.x = vec1.x + vec2.x;
  ret.y = vec1.y + vec2.y;
  ret.z = vec1.z + vec2.z;
  ret.w = vec1.w + vec2.w;

  return ret;
}
