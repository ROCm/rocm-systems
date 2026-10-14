// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "json_output.h"

namespace rocjitsu::cli {

std::string json_quoted(std::string_view text) {
  std::string out;
  out.reserve(text.size() + 2);
  out.push_back('"');
  for (unsigned char c : text) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\b':
      out += "\\b";
      break;
    case '\f':
      out += "\\f";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (c < 0x20) {
        constexpr char kHex[] = "0123456789abcdef";
        out += "\\u00";
        out.push_back(kHex[c >> 4]);
        out.push_back(kHex[c & 0x0f]);
      } else {
        out.push_back(static_cast<char>(c));
      }
      break;
    }
  }
  out.push_back('"');
  return out;
}

void write_json_string(std::ostream &out, std::string_view text) { out << json_quoted(text); }

void write_json_string_field(std::ostream &out, std::string_view key, std::string_view value,
                             bool comma) {
  out << "  ";
  write_json_string(out, key);
  out << ": ";
  write_json_string(out, value);
  if (comma)
    out << ",";
  out << "\n";
}

} // namespace rocjitsu::cli
