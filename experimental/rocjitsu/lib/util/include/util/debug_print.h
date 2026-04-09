// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DEBUG_PRINT_H_
#define UTIL_DEBUG_PRINT_H_

#include <iostream>
#include <ostream>

namespace util {
namespace debug {

#ifndef NDEBUG
inline constexpr bool DEBUG_ENABLE = true;
#else
inline constexpr bool DEBUG_ENABLE = false;
#endif

template <typename... Args> static void print(Args &&...args) {
  if constexpr (DEBUG_ENABLE) {
    std::ostream &trace_stream(std::cerr);
    (trace_stream << ... << args);
    trace_stream << std::endl;
    trace_stream.flush();
  }
}

} // namespace debug
} // namespace util

#endif // UTIL_DEBUG_PRINT_H_
