/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef INCLUDE_RDC_LIB_RDC_COMMON_H_
#define INCLUDE_RDC_LIB_RDC_COMMON_H_
#include <cstring>
#include <iostream>
#include <map>
#include <utility>

#include "rdc/rdc.h"

//<! The key to identify the field with <gpu_id, field_id>
typedef std::pair<uint32_t, rdc_field_t> RdcFieldKey;

//<! The key to identify the field with <gpu_id, field_group_id>
typedef std::pair<uint32_t, uint32_t> RdcFieldGroupKey;

//!< The gauge metrics do not require aggregations
typedef std::map<RdcFieldKey, uint64_t> rdc_gpu_gauges_t;

//!< Initial metadata key with which rdcd admits a RegisterPolicy stream
inline constexpr char kRdcPolicyAdmissionKey[] = "rdc-policy-admission";

//!< The size of the C API's job_id parameters, char job_id[64]
inline constexpr size_t kRdcJobIdSize = 64;

//!< Whether a job ID of length bytes fits the C API's char job_id[64] with its
//!< NUL and has no control characters, which logs would carry.
inline bool rdc_valid_job_id(const char* job_id, size_t length) {
  if (length >= kRdcJobIdSize) {
    return false;
  }
  for (size_t i = 0; i < length; ++i) {
    const unsigned char c = static_cast<unsigned char>(job_id[i]);
    if (c < 0x20 || c == 0x7f) {
      return false;
    }
  }
  return true;
}

//!< Whether a group or field group name of length bytes fits RDC's
//!< char[RDC_MAX_STR_LENGTH] name buffers with its NUL and has no NUL inside
//!< (which a std::string from a client can have, and its c_str() would cut).
inline bool rdc_valid_name(const char* name, size_t length) {
  return length < RDC_MAX_STR_LENGTH && (length == 0 || std::memchr(name, '\0', length) == nullptr);
}

/**
 *  @brief The strncpy but with null terminated
 *
 *  @details It will copy at most n-1 bytes from src to dst, and
 *  always adds a null terminator following the bytes copied to dst.
 *
 *  @param[out] dest The destination string to copy
 *
 *  @param[in] src The source string to be copied
 *
 *  @param[in] n At most n-1 bytes will be copied
 *
 *  @retval Return a pointer to the destination string.
 */
char* strncpy_with_null(char* dest, const char* src, size_t n);

#endif  // INCLUDE_RDC_LIB_RDC_COMMON_H_
