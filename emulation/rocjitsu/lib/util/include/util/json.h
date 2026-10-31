// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <string_view>

namespace util {

/// @brief Escape @p value for use inside a JSON string literal.
inline std::string json_escape(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size());
  constexpr char hex[] = "0123456789abcdef";
  for (const unsigned char ch : value) {
    switch (ch) {
    case '\"':
      escaped += "\\\"";
      break;
    case '\\':
      escaped += "\\\\";
      break;
    case '\b':
      escaped += "\\b";
      break;
    case '\f':
      escaped += "\\f";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      if (ch < 0x20) {
        escaped += "\\u00";
        escaped += hex[ch >> 4];
        escaped += hex[ch & 0xf];
      } else {
        escaped += static_cast<char>(ch);
      }
    }
  }
  return escaped;
}

} // namespace util
