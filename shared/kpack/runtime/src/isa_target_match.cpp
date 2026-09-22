// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "isa_target_match.h"

#include <cstring>

namespace kpack {

namespace {
constexpr char kTargetTriplePrefix[] = "amdgcn-amd-amdhsa--";
constexpr size_t kTargetTriplePrefixLen = sizeof(kTargetTriplePrefix) - 1;
}  // namespace

std::string strip_target_prefix(const char* isa) {
  if (!isa) return "";
  std::string s(isa);
  if (s.size() > kTargetTriplePrefixLen &&
      s.compare(0, kTargetTriplePrefixLen, kTargetTriplePrefix) == 0) {
    return s.substr(kTargetTriplePrefixLen);
  }
  return s;
}

ParsedTarget parse_target_id(const std::string& target) {
  ParsedTarget result;
  if (target.empty()) return result;

  // Split on ':' — first element is processor, rest are features
  size_t pos = 0;
  size_t colon = target.find(':');

  result.processor = target.substr(0, colon);

  while (colon != std::string::npos) {
    pos = colon + 1;
    colon = target.find(':', pos);
    std::string feature = target.substr(pos, colon - pos);
    if (!feature.empty()) {
      result.features.push_back(std::move(feature));
    }
  }

  return result;
}

}  // namespace kpack
