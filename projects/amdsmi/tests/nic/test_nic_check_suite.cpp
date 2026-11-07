// SPDX-License-Identifier: MIT
/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * Unit tests for the pass/fail harness (nic_check.h) that every standalone NIC suite reports
 * through. A harness that miscounted, or exited 0 after a failure, would turn all of them green.
 *
 * This suite cannot report through the harness it tests, so it keeps its own count.
 */

#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>

#include "nic_check.h"

namespace {

auto g_failures = int{0};

auto expect(const std::string& name, bool is_passed) -> void {
  if (!is_passed) {
    ++g_failures;
  }
  std::cout << (is_passed ? "  PASS: " : "  FAIL: ") << name << "\n";
}

struct StdoutRestore_t {
  std::streambuf* m_original;
  ~StdoutRestore_t() { std::cout.rdbuf(m_original); }
};

// Runs fn with std::cout redirected and returns what it printed.
template <typename Fn>
auto capture_stdout(Fn&& fn) -> std::string {
  auto buffer = std::ostringstream{};
  auto restore = StdoutRestore_t{std::cout.rdbuf(buffer.rdbuf())};
  fn();
  return buffer.str();
}

auto reset_counters() -> void {
  nic_test::g_tests_run = 0;
  nic_test::g_tests_failed = 0;
}

}  // namespace

auto main() -> int {
  std::cout << "NIC check harness tests\n";
  std::cout << "=======================\n";

  {
    reset_counters();
    const auto text = capture_stdout([] { nic_test::check("alpha", true); });
    expect("a passing check prints PASS and its name", text == "  PASS: alpha\n");
    expect("a passing check counts as run", nic_test::g_tests_run == 1);
    expect("a passing check does not count as failed", nic_test::g_tests_failed == 0);
  }

  {
    reset_counters();
    const auto text = capture_stdout([] { nic_test::check("beta", false); });
    expect("a failing check prints FAIL and its name", text == "  FAIL: beta\n");
    expect("a failing check counts as run", nic_test::g_tests_run == 1);
    expect("a failing check counts as failed", nic_test::g_tests_failed == 1);
  }

  {
    reset_counters();
    const auto text = capture_stdout([] {
      nic_test::check("gamma", true, "got 3");
      nic_test::check("delta", true, "");
    });
    expect("a detail follows the name after a dash",
           text.find("  PASS: gamma - got 3\n") != std::string::npos);
    expect("an empty detail adds nothing", text.find("  PASS: delta\n") != std::string::npos);
  }

  {
    reset_counters();
    auto exit_code = int{-1};
    const auto text = capture_stdout([&exit_code] {
      nic_test::check("epsilon", true);
      exit_code = nic_test::report();
    });
    expect("report exits 0 when nothing failed", exit_code == 0);
    expect("report prints the totals", text.find("\n1 tests run, 0 failed\n") != std::string::npos);
  }

  {
    reset_counters();
    auto exit_code = int{-1};
    const auto text = capture_stdout([&exit_code] {
      nic_test::check("zeta", false);
      nic_test::check("eta", true);
      exit_code = nic_test::report();
    });
    expect("report exits 1 when any check failed", exit_code == 1);
    expect("report counts every check, passed or failed",
           text.find("\n2 tests run, 1 failed\n") != std::string::npos);
  }

  std::cout << "\n" << g_failures << " harness failures\n";
  return (g_failures == 0) ? 0 : 1;
}
