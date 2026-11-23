/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Fails on a stale or wrong entry in a derived binary's base-test exclusion lists.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "transport/derived_transport_tests.h"

#include <cstring>
#include <string>
#include <vector>

namespace {

using rccl_derived_tests::BaseTests;
using rccl_derived_tests::EnabledTest;
using rccl_derived_tests::EnabledTests;
using rccl_derived_tests::ExcludedTest;
using rccl_derived_tests::ExcludedTests;
using rccl_derived_tests::kDisabledPrefix;
using rccl_derived_tests::Tag;

std::vector<const ::testing::TestInfo*> Registered(const std::string& suite, const std::string& name) {
  std::vector<const ::testing::TestInfo*> found;
  const ::testing::UnitTest* unit = ::testing::UnitTest::GetInstance();
  for (int i = 0; i < unit->total_test_suite_count(); ++i) {
    const ::testing::TestSuite* s = unit->GetTestSuite(i);
    for (int j = 0; j < s->total_test_count(); ++j) {
      const ::testing::TestInfo* t = s->GetTestInfo(j);
      if (suite == t->test_suite_name() && name == t->name()) found.push_back(t);
    }
  }
  return found;
}

bool FromBaseFile(const ::testing::TestInfo* t) { return std::strstr(t->file(), RCCL_DERIVED_BASE_DIR) != nullptr; }

bool BaseFileHas(const std::string& suite, const std::string& name) {
  for (const auto& t : BaseTests()) {
    if (t.suite == suite && t.name == name) return true;
  }
  return false;
}

std::string FullName(const std::string& suite, const std::string& name) { return suite + "." + name; }

#define RCCL_DERIVED_CHECK_TEST(test) TEST(RCCL_DERIVED_SELF_CHECK_SUITE, test)

RCCL_DERIVED_CHECK_TEST(NoExcludedBaseTestRuns) {
  for (const ExcludedTest& e : ExcludedTests()) {
    EXPECT_TRUE(BaseFileHas(e.suite, e.name))
        << FullName(e.suite, e.name) << " is excluded but the base file has no such test";
    for (const ::testing::TestInfo* t : Registered(e.suite, e.name)) {
      EXPECT_FALSE(FromBaseFile(t)) << FullName(e.suite, e.name) << " is excluded but registered from " << t->file();
    }
    if (e.tag != Tag::replaced) {
      EXPECT_TRUE(Registered(e.suite, e.name).empty())
          << FullName(e.suite, e.name) << " has a derived test; tag it replaced";
    }
  }
}

RCCL_DERIVED_CHECK_TEST(EveryReplacedTestHasADerivedTest) {
  // Only an unfiltered run can tell that the derived test runs; other runs check it is registered.
  const std::string filter = ::testing::GTEST_FLAG(filter);
  const bool unfiltered = filter.empty() || filter == "*";
  for (const ExcludedTest& e : ExcludedTests()) {
    if (e.tag != Tag::replaced) continue;
    const auto tests = Registered(e.suite, e.name);
    ASSERT_EQ(tests.size(), 1u) << FullName(e.suite, e.name) << " is tagged replaced but has no derived test";
    EXPECT_FALSE(FromBaseFile(tests[0])) << FullName(e.suite, e.name) << " comes from " << tests[0]->file();
    if (unfiltered && std::strncmp(e.name, kDisabledPrefix, sizeof(kDisabledPrefix) - 1) != 0) {
      EXPECT_TRUE(tests[0]->should_run() || tests[0]->is_in_another_shard()) << FullName(e.suite, e.name);
    }
  }
}

RCCL_DERIVED_CHECK_TEST(EveryReEnabledBaseTestIsRegistered) {
  for (const EnabledTest& e : EnabledTests()) {
    const std::string disabled = std::string(kDisabledPrefix) + e.name;
    EXPECT_TRUE(BaseFileHas(e.suite, disabled))
        << FullName(e.suite, disabled) << " is re-enabled but the base file has no such test";
    const auto tests = Registered(e.suite, e.name);
    ASSERT_EQ(tests.size(), 1u) << FullName(e.suite, e.name) << " is re-enabled but not registered";
    EXPECT_TRUE(FromBaseFile(tests[0])) << FullName(e.suite, e.name) << " comes from " << tests[0]->file();
    EXPECT_TRUE(Registered(e.suite, disabled).empty()) << FullName(e.suite, disabled);
  }
}

}  // namespace
