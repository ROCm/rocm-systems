// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "hip_test_support.hpp"
#include "race_log_expectation.hpp"

#include <cstdio>

namespace rocjitsu::test {

class RaceTestBase : public HipHazardTestBase {
protected:
  void ExpectNoRace() {
    const RaceLogParseResult parsed = parseRaceLogFromEnvironment();
    ASSERT_TRUE(parsed.ok()) << parsed.error;
    if (!parsed.records.empty()) {
      for (const auto &record : parsed.records) {
        std::fprintf(stderr, "  [%s symbol=%s dispatch=%d %s reg=%d wg=%s] %s\n",
                     record.kernel.c_str(), record.symbol.c_str(), record.dispatch,
                     record.type.c_str(), record.reg, record.workgroup.c_str(),
                     record.message.c_str());
      }
    }
    EXPECT_TRUE(parsed.records.empty()) << "Expected no races, got " << parsed.records.size();
  }

  void ExpectRace(const RaceExpectation &expected) {
    const RaceLogParseResult parsed = parseRaceLogFromEnvironment();
    ASSERT_TRUE(parsed.ok()) << parsed.error;
    const RaceExpectationMatchResult matched = matchRaceExpectation(parsed.records, expected);
    EXPECT_TRUE(matched.ok()) << matched.message();
  }
};

} // namespace rocjitsu::test
