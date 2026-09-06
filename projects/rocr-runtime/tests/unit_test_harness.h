/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

/* Neither libhsakmt nor hsa-runtime has a unit-test framework, and pulling one
 * in for a handful of pure-logic checks is not worth the dependency, so this is
 * the whole harness: register a case with TEST_CASE, assert with CHECK, return
 * RunAllTests() from main(). Failures report file, line and expression and keep
 * going, so one run shows every broken case rather than only the first.
 *
 * It lives above both sub-projects because both use it and neither may include
 * the other's tree. Header-only and standard-library-only on purpose: a test
 * that needs a build system is a test nobody runs by hand.
 */

#include <cstdio>
#include <string>
#include <vector>

namespace unittest {

struct TestCase {
  const char* name;
  void (*fn)();
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> cases;
  return cases;
}

inline int& failure_count() {
  static int failures = 0;
  return failures;
}

struct Registrar {
  Registrar(const char* name, void (*fn)()) { registry().push_back(TestCase{name, fn}); }
};

inline void ReportFailure(const char* file, int line, const char* expr) {
  ++failure_count();
  std::fprintf(stderr, "  FAIL %s:%d: %s\n", file, line, expr);
}

inline int RunAllTests() {
  int failed_cases = 0;
  for (const auto& test : registry()) {
    const int before = failure_count();
    std::printf("[ RUN      ] %s\n", test.name);
    test.fn();
    if (failure_count() == before) {
      std::printf("[       OK ] %s\n", test.name);
    } else {
      std::printf("[   FAILED ] %s\n", test.name);
      ++failed_cases;
    }
  }
  std::printf("%d test(s) run, %d failed, %d assertion failure(s)\n",
              static_cast<int>(registry().size()), failed_cases, failure_count());
  return failed_cases == 0 ? 0 : 1;
}

}  // namespace unittest

#define TEST_CASE(name)                                                                            \
  static void name();                                                                              \
  static ::unittest::Registrar name##_registrar(#name, &name);                                     \
  static void name()

#define CHECK(expr)                                                                                \
  do {                                                                                             \
    if (!(expr)) ::unittest::ReportFailure(__FILE__, __LINE__, #expr);                             \
  } while (0)

#define CHECK_EQ(lhs, rhs)                                                                         \
  do {                                                                                             \
    if (!((lhs) == (rhs))) ::unittest::ReportFailure(__FILE__, __LINE__, #lhs " == " #rhs);        \
  } while (0)
