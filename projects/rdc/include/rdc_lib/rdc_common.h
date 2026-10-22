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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <utility>

#include "rdc/rdc.h"

//<! The key to identify the field with <gpu_id, field_id>
typedef std::pair<uint32_t, rdc_field_t> RdcFieldKey;

//<! The key to identify the field with <gpu_id, field_group_id>
typedef std::pair<uint32_t, uint32_t> RdcFieldGroupKey;

//!< The gauge metrics do not require aggregations
typedef std::map<RdcFieldKey, uint64_t> rdc_gpu_gauges_t;

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

/**
 *  @brief The length of the well-formed UTF-8 character (RFC 3629) at s[0..size), or 0 when
 *  the bytes there do not start one: no overlong form, no UTF-16 surrogate, nothing above
 *  U+10FFFF, no truncated sequence.
 */
inline size_t rdc_utf8_char_length(const unsigned char* s, size_t size) {
  if (size == 0) return 0;
  const unsigned char c = s[0];
  if (c < 0x80) return 1;
  size_t n = 0;
  unsigned char lo = 0x80;  // the second byte's range
  unsigned char hi = 0xBF;
  if (c >= 0xC2 && c <= 0xDF) {
    n = 2;
  } else if (c >= 0xE0 && c <= 0xEF) {
    n = 3;
    if (c == 0xE0) lo = 0xA0;
    if (c == 0xED) hi = 0x9F;
  } else if (c >= 0xF0 && c <= 0xF4) {
    n = 4;
    if (c == 0xF0) lo = 0x90;
    if (c == 0xF4) hi = 0x8F;
  } else {
    return 0;
  }
  if (size < n || s[1] < lo || s[1] > hi) return 0;
  for (size_t k = 2; k < n; ++k) {
    if (s[k] < 0x80 || s[k] > 0xBF) return 0;
  }
  return n;
}

/**
 *  @brief Whether the UTF-8 character s[0..n) is a C0 control (U+0000-U+001F), DEL (U+007F) or
 *  a C1 control (U+0080-U+009F).
 */
inline bool rdc_is_control_char(const unsigned char* s, size_t n) {
  if (n == 1) return s[0] < 0x20 || s[0] == 0x7F;
  return n == 2 && s[0] == 0xC2 && s[1] <= 0x9F;
}

/**
 *  @brief Whether text[0..size) is clean text: valid UTF-8 (rdc_utf8_char_length()) with no
 *  control character (rdc_is_control_char()).
 */
inline bool rdc_is_clean_text(const char* text, size_t size) {
  const auto* s = reinterpret_cast<const unsigned char*>(text);
  for (size_t i = 0; i < size;) {
    const size_t n = rdc_utf8_char_length(s + i, size - i);
    if (n == 0 || rdc_is_control_char(s + i, n)) return false;
    i += n;
  }
  return true;
}

/**
 *  @brief Text from rdcd, safe to print: control characters and each byte that is not part of
 *  valid UTF-8 are escaped, so the text cannot send escape sequences to a terminal.
 *
 *  @details In text, each such byte becomes \xHH. With json, a control character becomes
 *  \u00HH (one escape for a two-byte C1 control), an invalid byte \u00HH, and '"' and '\' are
 *  escaped too, so the result fits in a JSON string.
 */
inline std::string rdc_escape_text(const std::string& text, bool json) {
  static const char kHex[] = "0123456789abcdef";
  const auto* s = reinterpret_cast<const unsigned char*>(text.data());
  std::string out;
  out.reserve(text.size());
  auto escape_byte = [&out, json](unsigned char b) {
    out += json ? "\\u00" : "\\x";
    out += kHex[b >> 4];
    out += kHex[b & 0xF];
  };
  for (size_t i = 0; i < text.size();) {
    const size_t n = rdc_utf8_char_length(s + i, text.size() - i);
    if (n == 0) {
      escape_byte(s[i]);  // not UTF-8: each byte on its own
      i += 1;
      continue;
    }
    if (rdc_is_control_char(s + i, n)) {
      if (n == 1) {
        escape_byte(s[i]);
      } else if (json) {
        escape_byte(s[i + 1]);  // C2 80-9F is U+0080-U+009F
      } else {
        escape_byte(s[i]);
        escape_byte(s[i + 1]);
      }
    } else {
      if (json && n == 1 && (s[i] == '"' || s[i] == '\\')) out += '\\';
      out.append(text, i, n);
    }
    i += n;
  }
  return out;
}

/**
 *  @brief The number in a field ID that a caller passed to RDC.
 *
 *  @details A caller can pass any 32-bit value as an ::rdc_field_t, and loading a value that
 *  ::rdc_field_t does not define as that enum is undefined behavior. This copies the bytes
 *  instead, so that the number can be checked before the ID is used as an ::rdc_field_t.
 */
inline uint32_t rdc_field_id_number(const rdc_field_t& field_id) {
  static_assert(sizeof(uint32_t) == sizeof(rdc_field_t), "rdc_field_t is 32 bits");
  uint32_t id;
  std::memcpy(&id, &field_id, sizeof(id));
  return id;
}

#endif  // INCLUDE_RDC_LIB_RDC_COMMON_H_
