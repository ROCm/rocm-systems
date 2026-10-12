/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <catch2/catch_test_macros.hpp>
#include <hip_test_level.hh>

#include <set>
#include <string>
#include <vector>

using HipTestLevel::LevelSource;

namespace {
/// Build a single-level resolution without going through the priority chain.
HipTestLevel::LevelResolution resolutionFrom(LevelSource source, const std::string& level) {
  HipTestLevel::LevelResolution resolution;
  resolution.levels = HipTestLevel::collectLevels(level, true);
  resolution.level = level;
  resolution.source = source;
  return resolution;
}

/// Build a multi-level resolution; @p levels supplies selection, the highest
/// supplies parameters.
HipTestLevel::LevelResolution resolutionFrom(LevelSource source, const std::set<int>& levels) {
  HipTestLevel::LevelResolution resolution;
  resolution.levels = levels;
  resolution.level = HipTestLevel::highestLevel(levels);
  resolution.source = source;
  return resolution;
}
}  // namespace

TEST_CASE("LevelParse_OrFiltersAreUnioned") {
  REQUIRE(HipTestLevel::collectLevels("[level_1],[level_2]", false) == std::set<int>{1, 2});
}

TEST_CASE("LevelParse_ExclusionOnlyLeavesTheRest") {
  REQUIRE(HipTestLevel::collectLevels("~[level_3]~[level_4]", false) == std::set<int>{0, 1, 2});
}

TEST_CASE("LevelParse_SelfCancellingFilterSelectsNothing") {
  REQUIRE(HipTestLevel::collectLevels("[level_0]~[level_0]", false).empty());
}

TEST_CASE("LevelParse_MalformedTagIsIgnored") {
  REQUIRE(HipTestLevel::collectLevels("[level_2x]", false).empty());
}

TEST_CASE("LevelParse_TestNameIsIgnored") {
  REQUIRE(HipTestLevel::collectLevels("Unit_atomicAnd_Positive_SameAddress", false).empty());
}

TEST_CASE("LevelParse_BareTagRejectedWhenNotAllowed") {
  REQUIRE(HipTestLevel::collectLevels("level_2", false).empty());
}

TEST_CASE("LevelParse_BareTagAcceptedWhenAllowed") {
  REQUIRE(HipTestLevel::collectLevels("level_2", true) == std::set<int>{2});
}

TEST_CASE("LevelParse_BareExclusionAcceptedWhenAllowed") {
  REQUIRE(HipTestLevel::collectLevels("~level_4", true) == std::set<int>{0, 1, 2, 3});
}

TEST_CASE("LevelResolve_DefaultsWhenNothingProvided") {
  const auto resolution = HipTestLevel::resolveLevel({}, nullptr);
  REQUIRE(resolution.level == "level_2");
  REQUIRE(resolution.source == LevelSource::kDefault);
  REQUIRE_FALSE(resolution.envRejected);
}

TEST_CASE("LevelResolve_EnvironmentUsedWhenCommandLineSilent") {
  const auto resolution = HipTestLevel::resolveLevel({"Unit_foo"}, "level_1");
  REQUIRE(resolution.level == "level_1");
  REQUIRE(resolution.source == LevelSource::kEnvironment);
}

TEST_CASE("LevelResolve_CommandLineBeatsEnvironment") {
  const auto resolution = HipTestLevel::resolveLevel({"[level_0]"}, "level_1");
  REQUIRE(resolution.level == "level_0");
  REQUIRE(resolution.source == LevelSource::kCommandLine);
}

TEST_CASE("LevelResolve_HighestWinsForParametersButAllLevelsAreKept") {
  // Parameters need a single level, so the highest wins. Selection must not
  // silently drop the others - they are kept for applyLevelFilter to use.
  const auto resolution = HipTestLevel::resolveLevel({"[level_1]", "[level_3]"}, nullptr);
  REQUIRE(resolution.level == "level_3");
  REQUIRE(resolution.levels == std::set<int>{1, 3});
  REQUIRE(resolution.multipleSelected());
}

TEST_CASE("LevelResolve_SingleLevelIsNotFlaggedAsMultiple") {
  const auto resolution = HipTestLevel::resolveLevel({}, "level_1");
  REQUIRE(resolution.levels == std::set<int>{1});
  REQUIRE_FALSE(resolution.multipleSelected());
}

TEST_CASE("LevelResolve_EnvironmentExclusionKeepsEveryRemainingLevel") {
  // "~level_4" asks for everything below level_4: all four must run, while the
  // parameters come from the highest of them.
  const auto resolution = HipTestLevel::resolveLevel({}, "~level_4");
  REQUIRE(resolution.levels == std::set<int>{0, 1, 2, 3});
  REQUIRE(resolution.level == "level_3");
  REQUIRE(resolution.source == LevelSource::kEnvironment);
}

TEST_CASE("LevelResolve_DefaultSelectsExactlyTheDefaultLevel") {
  const auto resolution = HipTestLevel::resolveLevel({"Unit_foo"}, nullptr);
  REQUIRE(resolution.levels == std::set<int>{2});
  REQUIRE(resolution.level == HipTestLevel::kDefaultLevel);
}

TEST_CASE("LevelResolve_GarbageEnvironmentFallsBackToDefault") {
  const auto resolution = HipTestLevel::resolveLevel({}, "not_a_level");
  REQUIRE(resolution.level == "level_2");
  REQUIRE(resolution.source == LevelSource::kDefault);
  REQUIRE(resolution.envRejected);
}

TEST_CASE("LevelResolve_OutOfRangeEnvironmentIsReportedUnsupported") {
  const auto resolution = HipTestLevel::resolveLevel({}, "level_9");
  REQUIRE(resolution.level == "level_9");
  REQUIRE(resolution.source == LevelSource::kEnvironment);
  REQUIRE(resolution.firstUnsupportedLevel() == "level_9");
}

TEST_CASE("LevelResolve_UnsupportedLevelIsCaughtEvenAlongsideSupportedOnes") {
  // Selection now uses every level, so validation must cover every level - not
  // just the highest that supplies the parameters.
  const auto resolution = HipTestLevel::resolveLevel({}, "level_1,level_7");
  REQUIRE(resolution.levels == std::set<int>{1, 7});
  REQUIRE(resolution.firstUnsupportedLevel() == "level_7");
}

TEST_CASE("LevelResolve_AllSupportedLevelsPassValidation") {
  REQUIRE(HipTestLevel::resolveLevel({}, "~level_4").firstUnsupportedLevel().empty());
  REQUIRE(HipTestLevel::resolveLevel({}, nullptr).firstUnsupportedLevel().empty());
}

TEST_CASE("LevelDescribe_ReportsSourceAndLevel") {
  const auto lines = HipTestLevel::resolveLevel({}, "level_1").describe();
  REQUIRE(lines.size() == 1);
  REQUIRE(lines.back() == "[Level Filter] Detected from HIP_TEST_LEVEL: level_1");
}

TEST_CASE("LevelDescribe_ReportsEveryLevelAndWhichSuppliesParameters") {
  const auto lines = HipTestLevel::resolveLevel({}, "level_1,level_3").describe();
  REQUIRE(lines.size() == 2);
  REQUIRE(lines.front() ==
          "[Level Filter] Multiple levels selected via HIP_TEST_LEVEL (level_1, level_3); "
          "all of them run, parameters use the highest (level_3)");
}

TEST_CASE("LevelDescribe_ReportsARejectedEnvironmentValue") {
  const auto lines = HipTestLevel::resolveLevel({}, "not_a_level").describe();
  REQUIRE(lines.size() == 2);
  REQUIRE(lines.front() ==
          "[Level Filter] HIP_TEST_LEVEL='not_a_level' has no valid level, "
          "ignoring");
  REQUIRE(lines.back() == "[Level Filter] Detected from default: level_2");
}

TEST_CASE("LevelSupport_KnownAndUnknown") {
  REQUIRE(HipTestLevel::isSupportedLevel("level_0"));
  REQUIRE(HipTestLevel::isSupportedLevel("level_4"));
  REQUIRE_FALSE(HipTestLevel::isSupportedLevel("level_5"));
  REQUIRE_FALSE(HipTestLevel::isSupportedLevel(""));
}

TEST_CASE("LevelResolve_SourceIsReportedForEveryPath") {
  // main() branches on the source: a command-line level is already expressed
  // in the spec, so it must be distinguishable from the other two.
  REQUIRE(HipTestLevel::resolveLevel({"[level_1]"}, nullptr).source == LevelSource::kCommandLine);
  REQUIRE(HipTestLevel::resolveLevel({"Unit_foo"}, "level_1").source == LevelSource::kEnvironment);
  REQUIRE(HipTestLevel::resolveLevel({"Unit_foo"}, nullptr).source == LevelSource::kDefault);
}

TEST_CASE("LevelFilter_DefaultLevelNeverNarrowsABareRun") {
  // A bare "./AtomicsTest" must run every test in the binary and take its
  // parameters from level_2. Narrowing here would empty any binary whose tests
  // all sit at another level - the contract tier is entirely level_0.
  std::vector<std::string> spec;
  REQUIRE_FALSE(
      HipTestLevel::applyLevelFilter(spec, resolutionFrom(LevelSource::kDefault, "level_2")));
  REQUIRE(spec.empty());
}

TEST_CASE("LevelFilter_EmptySpecGetsTheEnvironmentLevelTag") {
  std::vector<std::string> spec;
  REQUIRE(
      HipTestLevel::applyLevelFilter(spec, resolutionFrom(LevelSource::kEnvironment, "level_0")));
  REQUIRE(spec == std::vector<std::string>{"[level_0]"});
}

TEST_CASE("LevelFilter_DefaultLevelLeavesANamedTestAlone") {
  // ./AtomicsTest Unit_foo and plain ctest must keep running the named test
  // whatever its tags; the default level only supplies the parameters.
  std::vector<std::string> spec{"Unit_foo"};
  REQUIRE_FALSE(
      HipTestLevel::applyLevelFilter(spec, resolutionFrom(LevelSource::kDefault, "level_2")));
  REQUIRE(spec == std::vector<std::string>{"Unit_foo"});
}

TEST_CASE("LevelFilter_CommandLineLevelLeavesTheSpecAlone") {
  std::vector<std::string> spec{"[level_0]"};
  REQUIRE_FALSE(
      HipTestLevel::applyLevelFilter(spec, resolutionFrom(LevelSource::kCommandLine, "level_0")));
  REQUIRE(spec == std::vector<std::string>{"[level_0]"});
}

TEST_CASE("LevelFilter_EnvironmentLevelIsAndedWithANamedTest") {
  // Patterns inside one filter are ANDed by Catch2, so this narrows the run
  // to the named test rather than widening it to every level_0 test.
  std::vector<std::string> spec{"Unit_atomicAnd_Positive_SameAddress"};
  REQUIRE(
      HipTestLevel::applyLevelFilter(spec, resolutionFrom(LevelSource::kEnvironment, "level_0")));
  REQUIRE(spec == std::vector<std::string>{"Unit_atomicAnd_Positive_SameAddress[level_0]"});
}

TEST_CASE("LevelFilter_EachOrFilterGetsTheTag") {
  // Appending once to the whole string would bind the tag to the last filter only.
  std::vector<std::string> spec{"Unit_a,Unit_b"};
  REQUIRE(
      HipTestLevel::applyLevelFilter(spec, resolutionFrom(LevelSource::kEnvironment, "level_1")));
  REQUIRE(spec == std::vector<std::string>{"Unit_a[level_1],Unit_b[level_1]"});
}

TEST_CASE("LevelFilter_EscapedCommaIsNotASeparator") {
  // CatchAddTests.cmake escapes \ , [ ] ; in generated test names.
  std::vector<std::string> spec{"Unit_a\\,b"};
  REQUIRE(
      HipTestLevel::applyLevelFilter(spec, resolutionFrom(LevelSource::kEnvironment, "level_1")));
  REQUIRE(spec == std::vector<std::string>{"Unit_a\\,b[level_1]"});
}

TEST_CASE("LevelFilter_TrailingUnpairedBackslashIsKeptAndTagged") {
  // Not reachable from CatchAddTests.cmake-generated specs, which always emit
  // matched backslash pairs; pinned here only so a change in this corner does
  // not go unnoticed. A trailing backslash has no following character to form
  // a pair with, so the loop's escape branch never triggers for it: it is
  // copied through like an ordinary character, and the tag is appended after
  // it rather than being swallowed into an escape of the tag itself.
  std::vector<std::string> spec{"Unit_a\\"};
  REQUIRE(
      HipTestLevel::applyLevelFilter(spec, resolutionFrom(LevelSource::kEnvironment, "level_1")));
  REQUIRE(spec == std::vector<std::string>{"Unit_a\\[level_1]"});
}

TEST_CASE("LevelFilter_ExclusionKeepsItsScope") {
  // Catch2 resets exclusion per pattern, so the appended tag is required,
  // not forbidden.
  std::vector<std::string> spec{"~[disabled]"};
  REQUIRE(
      HipTestLevel::applyLevelFilter(spec, resolutionFrom(LevelSource::kEnvironment, "level_2")));
  REQUIRE(spec == std::vector<std::string>{"~[disabled][level_2]"});
}

TEST_CASE("LevelFilter_MultipleSpecEntriesAreAllRewritten") {
  std::vector<std::string> spec{"Unit_a", "Unit_b"};
  REQUIRE(
      HipTestLevel::applyLevelFilter(spec, resolutionFrom(LevelSource::kEnvironment, "level_4")));
  REQUIRE(spec == std::vector<std::string>{"Unit_a[level_4]", "Unit_b[level_4]"});
}

TEST_CASE("LevelFilter_EmptySpecGetsEverySelectedLevel") {
  // Catch2 ORs comma-separated filters, so this selects level_1 or level_3.
  std::vector<std::string> spec;
  REQUIRE(HipTestLevel::applyLevelFilter(
      spec, resolutionFrom(LevelSource::kEnvironment, std::set<int>{1, 3})));
  REQUIRE(spec == std::vector<std::string>{"[level_1],[level_3]"});
}

TEST_CASE("LevelFilter_NamedTestIsRepeatedOncePerSelectedLevel") {
  // Catch2 has no OR inside a filter, so "Unit_foo AND (level_1 OR level_3)"
  // has to be written as two filters.
  std::vector<std::string> spec{"Unit_foo"};
  REQUIRE(HipTestLevel::applyLevelFilter(
      spec, resolutionFrom(LevelSource::kEnvironment, std::set<int>{1, 3})));
  REQUIRE(spec == std::vector<std::string>{"Unit_foo[level_1],Unit_foo[level_3]"});
}

TEST_CASE("LevelFilter_CrossProductCoversEveryFilterAndLevel") {
  std::vector<std::string> spec{"Unit_a,Unit_b"};
  REQUIRE(HipTestLevel::applyLevelFilter(
      spec, resolutionFrom(LevelSource::kEnvironment, std::set<int>{0, 2})));
  REQUIRE(spec == std::vector<std::string>{"Unit_a[level_0],Unit_a[level_2],"
                                           "Unit_b[level_0],Unit_b[level_2]"});
}

TEST_CASE("LevelFilter_EverySelectedLevelSurvivesARoundTrip") {
  std::vector<std::string> spec;
  const auto resolution = HipTestLevel::resolveLevel({}, "~level_4");
  REQUIRE(HipTestLevel::applyLevelFilter(spec, resolution));
  REQUIRE(HipTestLevel::collectLevels(spec.at(0), false) == std::set<int>{0, 1, 2, 3});
}

TEST_CASE("LevelFilter_TagRoundTrips") {
  std::vector<std::string> spec;
  const auto resolution = HipTestLevel::resolveLevel({}, "level_3");
  REQUIRE(HipTestLevel::applyLevelFilter(spec, resolution));
  REQUIRE(HipTestLevel::collectLevels(spec.at(0), false) == std::set<int>{3});
}

TEST_CASE("LevelExitCode_NoTestsBecomesSkippedWhenFiltered") {
  // Catch2 returns NoTestsRunExitCode when the level filter matched nothing;
  // ctest must see a skip, not a failure.
  REQUIRE(HipTestLevel::mapLevelFilterExitCode(2, true) == 4);
}

TEST_CASE("LevelExitCode_NoTestsIsLeftAloneWhenNotFiltered") {
  // No filter was applied, so an empty run means the user's own spec matched
  // nothing - a genuine failure worth reporting.
  REQUIRE(HipTestLevel::mapLevelFilterExitCode(2, false) == 2);
}

TEST_CASE("LevelExitCode_OtherCodesPassThrough") {
  REQUIRE(HipTestLevel::mapLevelFilterExitCode(0, true) == 0);
  REQUIRE(HipTestLevel::mapLevelFilterExitCode(1, true) == 1);
  REQUIRE(HipTestLevel::mapLevelFilterExitCode(42, true) == 42);
}
