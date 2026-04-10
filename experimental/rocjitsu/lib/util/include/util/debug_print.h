// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DEBUG_PRINT_H_
#define UTIL_DEBUG_PRINT_H_

#include <cstdlib>
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

namespace trace {

/// @brief Runtime-controlled trace output. Always compiled in, but only
/// prints when ROCJITSU_TRACE environment variable is set.
/// Zero cost when not enabled (single branch on cached flag).
inline bool enabled() {
  static const bool on = (std::getenv("ROCJITSU_TRACE") != nullptr);
  return on;
}

template <typename... Args> static void print(Args &&...args) {
  if (enabled()) {
    std::ostream &s(std::cerr);
    s << "[rj] ";
    (s << ... << args);
    s << std::endl;
    s.flush();
  }
}

} // namespace trace
} // namespace util

#endif // UTIL_DEBUG_PRINT_H_
