/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdio>
#include <string_view>

namespace hip::aqllog {

// Kept separate from the Catch-based AQL helpers so standalone test
// executables can delimit the portion of the trace their parent will inspect.
inline constexpr char kRegionBegin[] = "HIP_TEST_AQL_REGION_BEGIN:";
inline constexpr char kRegionEnd[] = "HIP_TEST_AQL_REGION_END:";

inline void BeginRegion(std::string_view name) {
  std::fflush(nullptr);
  std::fprintf(stderr, "%s%.*s\n", kRegionBegin, static_cast<int>(name.size()), name.data());
  std::fflush(stderr);
}

inline void EndRegion(std::string_view name) {
  std::fflush(nullptr);
  std::fprintf(stderr, "%s%.*s\n", kRegionEnd, static_cast<int>(name.size()), name.data());
  std::fflush(stderr);
}

}  // namespace hip::aqllog
