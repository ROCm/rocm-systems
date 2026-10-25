// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>

#include <array>
#include <string>
#include <string_view>

namespace rocjitsu::test {

inline bool has_sanitizer_diagnostic(std::string_view output) {
  constexpr std::array markers = {
      "AddressSanitizer",           "LeakSanitizer", "ThreadSanitizer", "MemorySanitizer",
      "UndefinedBehaviorSanitizer", "runtime error:"};
  for (std::string_view marker : markers)
    if (output.find(marker) != std::string_view::npos)
      return true;
  return false;
}

// Sanitizers can exit with status 1 or SIGABRT, just like a fatal GPU failure.
// Check captured stderr as well as the exit status to keep their reports visible.
class SanitizerFreeDiagnostic {
public:
  using is_gtest_matcher = void;
  explicit SanitizerFreeDiagnostic(const std::string &pattern)
      : diagnostic_(::testing::ContainsRegex(pattern)) {}

  bool MatchAndExplain(const std::string &output, ::testing::MatchResultListener *listener) const {
    if (has_sanitizer_diagnostic(output)) {
      *listener << "contains a sanitizer diagnostic";
      return false;
    }
    return diagnostic_.MatchAndExplain(output, listener);
  }

  void DescribeTo(std::ostream *stream) const {
    diagnostic_.DescribeTo(stream);
    *stream << " without sanitizer diagnostics";
  }

  void DescribeNegationTo(std::ostream *stream) const {
    *stream << "contains sanitizer diagnostics or ";
    diagnostic_.DescribeNegationTo(stream);
  }

private:
  ::testing::Matcher<const std::string &> diagnostic_;
};

} // namespace rocjitsu::test
