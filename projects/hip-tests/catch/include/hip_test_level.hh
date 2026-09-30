/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <set>
#include <string>
#include <vector>

/**
 * @brief Resolution of the single test level used for a whole run.
 *
 * The active level is resolved once, before the run starts, using a strict
 * priority order:
 *   1. Catch2 command-line tag filter, e.g. ./test "[level_2]"
 *   2. HIP_TEST_LEVEL environment variable, e.g. HIP_TEST_LEVEL=level_2
 *   3. Hardcoded default (kDefaultLevel)
 *
 * A higher-priority source is used whenever it yields a level; the lower ones
 * are then ignored (strict precedence, no merging across sources).
 */
namespace HipTestLevel {

/// Levels understood by the test suite (must match definitions.yaml).
inline constexpr const char* kSupportedLevels[] = {"level_0", "level_1", "level_2", "level_3",
                                                   "level_4"};

/// Level used when neither the command line nor HIP_TEST_LEVEL specify one.
inline constexpr const char* kDefaultLevel = "level_2";

/// Which source the active level came from.
enum class LevelSource { kCommandLine, kEnvironment, kDefault };

/**
 * @brief Outcome of resolving the active level, including what to report.
 *
 * A request may select more than one level ("HIP_TEST_LEVEL=level_1,level_3",
 * "~level_4"). The two consumers treat that differently:
 *
 *   - Test parameters take the single highest level (@ref level). There is one
 *     TestParameterStore per run, so one set of sizes and iteration counts has
 *     to win, and the highest is the widest.
 *   - Test selection uses every selected level (@ref levels), OR'd together, so
 *     nothing the request asked for is silently dropped from the run.
 */
struct LevelResolution {
  /// Every selected level, ascending. Never empty after resolveLevel().
  std::set<int> levels;
  /// Highest of @ref levels as "level_N" - the level whose parameters are used.
  /// Not necessarily supported; callers must still validate (see
  /// @ref firstUnsupportedLevel).
  std::string level;
  /// Where @ref levels came from.
  LevelSource source = LevelSource::kDefault;
  /// Value of HIP_TEST_LEVEL as seen by resolveLevel(), empty when unset.
  /// Retained so @ref describe can report a rejected value without re-reading
  /// the environment.
  std::string envValue;
  /// HIP_TEST_LEVEL was set but named no valid level, so it was skipped.
  bool envRejected = false;

  /// @brief Whether more than one level was selected.
  bool multipleSelected() const;

  /// @brief First selected level the suite does not support, as "level_N".
  /// @return "" when every selected level is supported.
  std::string firstUnsupportedLevel() const;

  /// @brief Human-readable lines describing how the level was resolved.
  /// @return One to three lines, most specific last. Never empty.
  std::vector<std::string> describe() const;
};

/// @brief Whether @p level is one of kSupportedLevels.
bool isSupportedLevel(const std::string& level);

/**
 * @brief Collect every level a tag expression can select.
 *
 * The expression is read the way Catch2 reads a test spec: a comma separated
 * list of filters OR'd together, each filter a sequence of patterns AND'd
 * together, each pattern optionally negated with '~'. Patterns that do not read
 * exactly "level_N" (test names, other tags, and malformed tags such as
 * "[level_2x]") place no constraint on the level and are ignored.
 *
 * A filter naming levels contributes those levels, minus the ones it excludes;
 * a filter that only excludes levels contributes every supported level except
 * the excluded ones; a filter mentioning no level at all contributes nothing.
 * Note that filters are OR'd, so "[level_1],~[level_2]" also selects level_3 and
 * level_4, exactly as Catch2 would.
 *
 * @param allowBareTags when true, a pattern may also be written without the
 *        surrounding brackets ("level_2", "~level_2"), which HIP_TEST_LEVEL
 *        accepts. Brackets, when present, must still be balanced.
 * @return Distinct level numbers selectable, sorted ascending (empty if the
 *         expression places no constraint on the level).
 */
std::set<int> collectLevels(const std::string& text, bool allowBareTags);

/**
 * @brief Reduce a set of levels to a single level string ("level_N").
 * @return "level_N" for the highest level, or "" if the set is empty.
 */
std::string highestLevel(const std::set<int>& levels);

/**
 * @brief Resolve the active level using the strict priority order.
 * @param testsOrTags the Catch2 test spec entries, as parsed from the command line.
 * @param envLevel value of HIP_TEST_LEVEL, or nullptr when unset.
 */
LevelResolution resolveLevel(const std::vector<std::string>& testsOrTags, const char* envLevel);

/**
 * @brief Let the active level select tests, when it was explicitly requested.
 *
 * An explicitly requested level narrows the run; the hardcoded default only
 * supplies parameters. Concretely:
 *
 *   - kCommandLine: the spec already names the levels. Nothing to do.
 *   - kDefault:     nothing to do either. The default level supplies parameters
 *                   and never narrows, so a bare "./test" and plain ctest run
 *                   every test in the binary, as they always have.
 *   - kEnvironment, empty spec: becomes "[level_A],[level_B],..." over every
 *                   selected level.
 *   - kEnvironment, non-empty spec: every filter of every entry is expanded once
 *                   per selected level, so "Unit_foo" with levels {1,3} becomes
 *                   "Unit_foo[level_1],Unit_foo[level_3]".
 *
 * Selection uses every level in LevelResolution::levels, not just the highest.
 * Catch2 has no OR inside a single filter, so a multi-level request is encoded
 * by repeating each filter once per level - the cross product of the spec's
 * filters and the selected levels.
 *
 * Only an explicit request selects. A test binary cannot tell a hand-typed
 * "./test Unit_foo" from the identical command line ctest generates for each
 * registered test, so "was a level explicitly asked for?" is the only signal
 * available. It also matters that level tags partition the suite rather than
 * nesting: a default that narrowed would empty any binary whose tests all sit
 * at some level other than the default.
 *
 * Patterns inside one Catch2 filter are AND'd, and filters (separated by
 * unescaped commas) are OR'd, so the tag must be appended per filter. Appending
 * it once as a new spec entry would instead OR in every test carrying the level.
 *
 * Backslash escapes are respected when splitting: CatchAddTests.cmake escapes
 * '\', ',', '[', ']' and ';' in generated test names, so a name may legitimately
 * contain "\,", which is not a filter separator.
 *
 * @param testsOrTags spec entries, rewritten in place.
 * @param resolution the active level and where it came from.
 * @return true if a filter was applied, so that a zero-test run means "nothing
 *         matched the level" rather than "nothing matched the user's spec".
 */
bool applyLevelFilter(std::vector<std::string>& testsOrTags, const LevelResolution& resolution);

/**
 * @brief Translate "no tests ran" into "skipped" when a level filter caused it.
 *
 * When applyLevelFilter() narrowed the spec and nothing matched, Catch2 exits
 * with NoTestsRunExitCode, which CTest reports as a failure. The run is really
 * a skip: the test exists, it just does not belong to the active level. Mapping
 * it onto Catch2's own AllTestsSkippedExitCode lets a single CTest
 * SKIP_RETURN_CODE cover both cases.
 *
 * Both codes are file-local constants in Catch2's catch_session.cpp:37-42 and
 * are not exported, so they are restated in the implementation.
 *
 * @param exitCode value returned by Catch::Session::run().
 * @param levelFilterApplied whether applyLevelFilter() rewrote the spec.
 * @return the exit code to return from main().
 */
int mapLevelFilterExitCode(int exitCode, bool levelFilterApplied);

}  // namespace HipTestLevel
