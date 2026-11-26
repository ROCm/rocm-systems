// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <ostream>
#include <span>
#include <string>
#include <string_view>

namespace rocjitsu::consan::render_detail {

inline std::string hex_bytes(std::span<const uint8_t> bytes) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (const uint8_t byte : bytes) {
    result += digits[byte >> 4];
    result += digits[byte & 15];
  }
  return result;
}

inline void write_json_string(std::ostream &out, std::string_view value) {
  constexpr char digits[] = "0123456789abcdef";
  out << '"';
  for (const unsigned char ch : value) {
    if (ch == '"' || ch == '\\')
      out << '\\' << ch;
    else if (ch < 32)
      out << "\\u00" << digits[ch >> 4] << digits[ch & 15];
    else
      out << ch;
  }
  out << '"';
}

} // namespace rocjitsu::consan::render_detail
