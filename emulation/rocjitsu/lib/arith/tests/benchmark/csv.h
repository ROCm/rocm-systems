// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CSV escaping and assembly of workload variant labels.

#pragma once

#include <string>
#include <vector>

namespace rocjitsu::arith::bench {

// Join variant descriptions into one CSV field, quoting embedded CSV delimiters.
inline std::string variant_field(const std::vector<std::string> &variants) {
  std::string value;
  for (int i = 0; i < int(variants.size()); ++i) {
    if (i) {
      value += ", ";
    }
    value += variants[i];
  }
  if (value.find_first_of(",\"\r\n") == std::string::npos) {
    return value;
  }
  std::string quoted = "\"";
  for (char c : value) {
    if (c == '"') {
      quoted += '"';
    }
    quoted += c;
  }
  return quoted + '"';
}

} // namespace rocjitsu::arith::bench
