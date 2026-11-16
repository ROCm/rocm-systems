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

inline bool has_assertion_diagnostic(std::string_view output) {
  while (!output.empty()) {
    const size_t end = output.find('\n');
    const std::string_view line = output.substr(0, end);
    if ((line.find("Assertion") != std::string_view::npos ||
         line.find("assertion") != std::string_view::npos) &&
        (line.find("failed") != std::string_view::npos ||
         line.find("failure") != std::string_view::npos))
      return true;
    if (end == std::string_view::npos)
      break;
    output.remove_prefix(end + 1);
  }
  return false;
}

// libstdc++ and libc++ report std::terminate() before aborting.
inline bool has_terminate_diagnostic(std::string_view output) {
  return output.find("terminate called") != std::string_view::npos ||
         output.find("libc++abi: terminating") != std::string_view::npos;
}

inline bool has_unexpected_failure_diagnostic(std::string_view output) {
  return has_sanitizer_diagnostic(output) || has_assertion_diagnostic(output) ||
         has_terminate_diagnostic(output);
}

// Names every report kind rejected by has_unexpected_failure_diagnostic().
inline constexpr char kUnexpectedFailureDiagnostics[] =
    "sanitizer, assertion or termination diagnostics";

// Sanitizers, assertions and std::terminate() can exit with status 1 or SIGABRT, like a
// fatal GPU failure. Check captured stderr as well as the exit status to keep their
// reports visible.
class ExpectedFailureDiagnostic {
public:
  using is_gtest_matcher = void;
  explicit ExpectedFailureDiagnostic(const std::string &pattern)
      : diagnostic_(::testing::ContainsRegex(pattern)) {}

  bool MatchAndExplain(const std::string &output, ::testing::MatchResultListener *listener) const {
    if (has_unexpected_failure_diagnostic(output)) {
      *listener << "contains " << kUnexpectedFailureDiagnostics;
      return false;
    }
    return diagnostic_.MatchAndExplain(output, listener);
  }

  void DescribeTo(std::ostream *stream) const {
    diagnostic_.DescribeTo(stream);
    *stream << " without " << kUnexpectedFailureDiagnostics;
  }

  void DescribeNegationTo(std::ostream *stream) const {
    *stream << "contains " << kUnexpectedFailureDiagnostics << " or ";
    diagnostic_.DescribeNegationTo(stream);
  }

private:
  ::testing::Matcher<const std::string &> diagnostic_;
};

} // namespace rocjitsu::test
