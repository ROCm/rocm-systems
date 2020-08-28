/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef INC_ROCTRACER_HIP_H_
#define INC_ROCTRACER_HIP_H_

#ifdef __cplusplus
#include <iostream>

inline static std::ostream& operator<<(std::ostream& out, const unsigned char& v) {
  out  << (unsigned int)v;
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const char& v) {
  out  << (unsigned char)v;
  return out;
}
#endif  // __cplusplus

#include <hip_ostream_ops.h>
#include <hip/hip_runtime.h>
#include <hip/hcc_detail/hip_prof_str.h>

#include <roctracer.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Traced calls ID enumeration
typedef enum hip_api_id_t roctracer_hip_api_cid_t;

#ifdef __cplusplus
}  // extern "C" block
#endif  // __cplusplus

#endif  // INC_ROCTRACER_HIP_H_
