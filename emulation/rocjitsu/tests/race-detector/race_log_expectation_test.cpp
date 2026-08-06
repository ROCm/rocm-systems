// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "race_log_expectation.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>

namespace rocjitsu::test {
namespace {

constexpr std::string_view kValidRaceLog =
    "[rocjitsu] Kernel dispatch: \"racy_kernel\" symbol=\"racy_kernel\"\n"
    "RACE kernel=racy_kernel symbol=_Z11racy_kernelPKfPf dispatch=3 "
    "type=VGPR access=read reg=2 wave=0 lane=1 wg=2,3,4 conflict=unknown\n"
    "Race on VGPR v2 [workgroup (2, 3, 4), wave 0, lane 1]\n"
    "  ==>  0x100  global_load_dword v2, v[0:1], off\n"
    "       0x104  s_waitcnt vmcnt(1)\n"
    "  ==>  0x108  global_store_dword v[0:1], v2, off\n"
    "END_RACE\n"
    "\n"
    "========================================\n"
    " ROCJITSU RACE DETECTION SUMMARY\n"
    "========================================\n"
    "  1 race(s) detected\n"
    "========================================\n";

RaceLogParseResult parse(std::string_view log) {
  std::istringstream input{std::string(log)};
  return parseRaceLog(input);
}

class EnvironmentGuard {
public:
  explicit EnvironmentGuard(const char *name) : name_(name) {
    if (const char *value = std::getenv(name); value != nullptr) {
      was_set_ = true;
      value_ = value;
    }
  }

  ~EnvironmentGuard() {
    if (was_set_)
      set(value_);
    else
      unset();
  }

  void set(const std::string &value) {
#ifdef _WIN32
    (void)_putenv_s(name_.c_str(), value.c_str());
#else
    (void)setenv(name_.c_str(), value.c_str(), 1);
#endif
  }

  void unset() {
#ifdef _WIN32
    (void)_putenv_s(name_.c_str(), "");
#else
    (void)unsetenv(name_.c_str());
#endif
  }

private:
  std::string name_;
  std::string value_;
  bool was_set_ = false;
};

TEST(RaceLogExpectationTest, ParsesValidEmptyLog) {
  const RaceLogParseResult parsed =
      parse("[rocjitsu] Kernel dispatch: \"clean_kernel\" symbol=\"clean_kernel\"\n"
            " ROCJITSU RACE DETECTION SUMMARY\n"
            " No races detected.\n");

  ASSERT_TRUE(parsed.ok()) << parsed.error;
  EXPECT_TRUE(parsed.records.empty());
}

TEST(RaceLogExpectationTest, RejectsEmptyLog) {
  const RaceLogParseResult parsed = parse("");

  EXPECT_FALSE(parsed.ok());
  EXPECT_NE(parsed.error.find("no recognizable race-plugin output"), std::string::npos);
}

TEST(RaceLogExpectationTest, RejectsUnrecognizedLog) {
  const RaceLogParseResult parsed = parse("unrelated output\nwithout any race plugin evidence\n");

  EXPECT_FALSE(parsed.ok());
  EXPECT_NE(parsed.error.find("no recognizable race-plugin output"), std::string::npos);
}

TEST(RaceLogExpectationTest, ParsesValidRaceAndTrace) {
  const RaceLogParseResult parsed = parse(kValidRaceLog);

  ASSERT_TRUE(parsed.ok()) << parsed.error;
  ASSERT_EQ(parsed.records.size(), 1u);
  const RaceRecord &record = parsed.records.front();
  EXPECT_EQ(record.kernel, "racy_kernel");
  EXPECT_EQ(record.symbol, "_Z11racy_kernelPKfPf");
  EXPECT_EQ(record.dispatch, 3);
  EXPECT_EQ(record.type, "VGPR");
  EXPECT_EQ(record.access, "read");
  EXPECT_EQ(record.reg, 2);
  EXPECT_EQ(record.wave, 0);
  EXPECT_EQ(record.lane, 1);
  EXPECT_EQ(record.workgroup, "2,3,4");
  EXPECT_EQ(record.conflict, "unknown");

  const TraceSections trace = parseTrace(record);
  EXPECT_EQ(trace.marker_count, 2u);
  EXPECT_NE(trace.header.find("VGPR"), std::string::npos);
  EXPECT_NE(trace.producer.find("global_load_dword"), std::string::npos);
  ASSERT_EQ(trace.between.size(), 1u);
  EXPECT_NE(trace.between.front().find("s_waitcnt"), std::string::npos);
  EXPECT_NE(trace.consumer.find("global_store_dword"), std::string::npos);
}

TEST(RaceLogExpectationTest, RejectsUnterminatedRaceBlock) {
  const std::string truncated =
      std::string(kValidRaceLog.substr(0, kValidRaceLog.find("END_RACE")));
  const RaceLogParseResult parsed = parse(truncated);

  EXPECT_FALSE(parsed.ok());
  EXPECT_NE(parsed.error.find("before END_RACE"), std::string::npos);
}

TEST(RaceLogExpectationTest, RejectsMalformedInteger) {
  std::string malformed(kValidRaceLog);
  malformed.replace(malformed.find("dispatch=3"), std::string("dispatch=3").size(),
                    "dispatch=three");
  const RaceLogParseResult parsed = parse(malformed);

  EXPECT_FALSE(parsed.ok());
  EXPECT_NE(parsed.error.find("invalid dispatch"), std::string::npos);
}

TEST(RaceLogExpectationTest, RejectsMissingRequiredField) {
  std::string malformed(kValidRaceLog);
  malformed.erase(malformed.find(" access=read"), std::string(" access=read").size());
  const RaceLogParseResult parsed = parse(malformed);

  EXPECT_FALSE(parsed.ok());
  EXPECT_NE(parsed.error.find("missing one or more required fields"), std::string::npos);
}

TEST(RaceLogExpectationTest, RejectsDuplicateField) {
  std::string malformed(kValidRaceLog);
  malformed.insert(malformed.find(" type=VGPR"), " dispatch=4");
  const RaceLogParseResult parsed = parse(malformed);

  EXPECT_FALSE(parsed.ok());
  EXPECT_NE(parsed.error.find("duplicate header field 'dispatch'"), std::string::npos);
}

TEST(RaceLogExpectationTest, MissingEnvironmentFails) {
  EnvironmentGuard environment("RJ_SINK_DIR");
  environment.unset();

  const RaceLogParseResult parsed = parseRaceLogFromEnvironment();

  EXPECT_FALSE(parsed.ok());
  EXPECT_EQ(parsed.error, "RJ_SINK_DIR is not set");
}

TEST(RaceLogExpectationTest, MissingFileFails) {
  const std::filesystem::path missing =
      std::filesystem::temp_directory_path() / "rocjitsu-race-log-expectation-missing" / "race.log";
  const RaceLogParseResult parsed = parseRaceLogFile(missing);

  EXPECT_FALSE(parsed.ok());
  EXPECT_NE(parsed.error.find("could not open race log"), std::string::npos);
}

TEST(RaceLogExpectationTest, MatchesStructuredExpectation) {
  const RaceLogParseResult parsed = parse(kValidRaceLog);
  ASSERT_TRUE(parsed.ok()) << parsed.error;
  const RaceExpectation expected{
      .kernel = "racy_kernel",
      .type = "VGPR",
      .access = "read",
      .wave = 0,
      .lane = 1,
      .producer = "global_load_dword",
      .between = "s_waitcnt",
      .consumer = "global_store_dword",
  };

  const RaceExpectationMatchResult matched = matchRaceExpectation(parsed.records, expected);

  EXPECT_TRUE(matched.ok()) << matched.message();
}

TEST(RaceLogExpectationTest, MissingExpectedTraceMarkerFails) {
  const RaceLogParseResult parsed = parse(kValidRaceLog);
  ASSERT_TRUE(parsed.ok()) << parsed.error;
  const RaceExpectation expected{
      .type = "VGPR",
      .consumer = "v_add_f32",
  };

  const RaceExpectationMatchResult matched = matchRaceExpectation(parsed.records, expected);

  EXPECT_FALSE(matched.ok());
  EXPECT_NE(matched.message().find("consumer trace"), std::string::npos);
}

} // namespace
} // namespace rocjitsu::test
