/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Registers a base transport's tests in a derived binary, minus that binary's generated exclusions.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_DERIVED_TRANSPORT_TESTS_H_
#define RCCL_TEST_HOST_DERIVED_TRANSPORT_TESTS_H_

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace rccl_derived_tests {

enum class Tag { replaced, lagging, diverged };

struct ExcludedTest {
  const char* suite;
  const char* name;
  Tag tag;
};

struct EnabledTest {
  const char* suite;
  const char* name;  // without DISABLED_
};

struct BaseTest {
  std::string suite;
  std::string name;
};

constexpr char kDisabledPrefix[] = "DISABLED_";

#define RCCL_DERIVED_EXCLUDED_TEST(suite, name, tag) {suite, name, Tag::tag},
inline const std::vector<ExcludedTest>& ExcludedTests() {
  static const std::vector<ExcludedTest> tests = {
#include "derived_tests_excluded.inc"
  };
  return tests;
}
#undef RCCL_DERIVED_EXCLUDED_TEST

#define RCCL_DERIVED_ENABLED_TEST(suite, name) {suite, name},
inline const std::vector<EnabledTest>& EnabledTests() {
  static const std::vector<EnabledTest> tests = {
#include "derived_tests_enabled.inc"
  };
  return tests;
}
#undef RCCL_DERIVED_ENABLED_TEST

// Every base TEST_F seen, registered or not, in definition order.
inline std::vector<BaseTest>& BaseTests() {
  static std::vector<BaseTest> tests;
  return tests;
}

inline bool IsExcluded(const std::string& suite, const std::string& name) {
  for (const ExcludedTest& e : ExcludedTests()) {
    if (suite == e.suite && name == e.name) return true;
  }
  return false;
}

inline std::string RunName(const std::string& suite, const std::string& name) {
  const std::string prefix = kDisabledPrefix;
  if (name.compare(0, prefix.size(), prefix) != 0) return name;
  for (const EnabledTest& e : EnabledTests()) {
    if (suite == e.suite && name.substr(prefix.size()) == e.name) return e.name;
  }
  return name;
}

template <class Fixture, class Test>
bool RegisterBaseTest(const char* suite, const char* name, const char* file, int line) {
  BaseTests().push_back({suite, name});
  if (IsExcluded(suite, name)) return false;
  ::testing::RegisterTest(suite, RunName(suite, name).c_str(), nullptr, nullptr, file, line,
                          []() -> Fixture* { return new Test(); });
  return true;
}

}  // namespace rccl_derived_tests

#endif  // RCCL_TEST_HOST_DERIVED_TRANSPORT_TESTS_H_
