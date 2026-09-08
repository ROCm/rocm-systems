// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "race_log_expectation.hpp"

#include <cstdio>
#include <vector>

namespace rocjitsu::test {

class RaceTestBase : public ::testing::Test {
protected:
  template <typename T> T *alloc(int count) {
    T *pointer = nullptr;
    (void)hipMalloc(&pointer, count * sizeof(T));
    return pointer;
  }

  template <typename T> T *allocWithData(int count) {
    T *pointer = alloc<T>(count);
    std::vector<T> host(count);
    for (int index = 0; index < count; ++index)
      host[index] = static_cast<T>(index);
    (void)hipMemcpy(pointer, host.data(), count * sizeof(T), hipMemcpyHostToDevice);
    return pointer;
  }

  void sync() { (void)hipDeviceSynchronize(); }

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
