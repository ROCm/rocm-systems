// SPDX-License-Identifier: MIT
/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * Pass/fail counting shared by the standalone NIC suites (test_nic_*_suite.cpp). Each suite is its
 * own executable, so the inline counters are per process.
 */

#ifndef AMDSMI_TESTS_NIC_NIC_CHECK_H_
#define AMDSMI_TESTS_NIC_NIC_CHECK_H_

#include <iostream>
#include <string>

namespace nic_test {

inline auto g_tests_run = int{0};
inline auto g_tests_failed = int{0};

inline auto check(const std::string& name, bool is_passed, const std::string& detail = "") -> void {
  ++g_tests_run;
  if (!is_passed) {
    ++g_tests_failed;
  }
  std::cout << (is_passed ? "  PASS: " : "  FAIL: ") << name;
  if (!detail.empty()) {
    std::cout << " - " << detail;
  }
  std::cout << "\n";
}

// Prints the totals; the return value is the process exit code.
inline auto report() -> int {
  std::cout << "\n" << g_tests_run << " tests run, " << g_tests_failed << " failed\n";
  return (g_tests_failed == 0) ? 0 : 1;
}

}  // namespace nic_test

#endif  // AMDSMI_TESTS_NIC_NIC_CHECK_H_
