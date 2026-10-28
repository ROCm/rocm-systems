/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <string>

/**
 * @brief Resolution of the single test parameter level used for a whole run.
 *
 * The level only picks the test parameters (memory sizes, block sizes,
 * iteration counts, ...). It never selects which tests run - that is left
 * entirely to the Catch2 test spec, e.g. ./test "[level_0]", or to ctest -L.
 * The two are independent knobs; to run the level_0 tests with level_0
 * parameters, set both:
 *
 *   HIP_TEST_LEVEL=level_0 ./test "[level_0]"
 *   HIP_TEST_LEVEL=level_0 ctest -L level_0
 *
 * The level is resolved once, before the run starts:
 *   1. HIP_TEST_LEVEL environment variable, e.g. HIP_TEST_LEVEL=level_0
 *   2. Hardcoded default (kDefaultLevel)
 *
 * HIP_TEST_LEVEL must name exactly one supported level or one test category -
 * "quick", "standard", "comprehensive", "full", as defined in
 * test_categories.yaml (see the generated hip_test_parameters.hh). A category
 * stands for the highest level it covers, so "HIP_TEST_LEVEL=quick" is
 * "HIP_TEST_LEVEL=level_0" and "HIP_TEST_LEVEL=full" is "HIP_TEST_LEVEL=level_4".
 * It pairs with the matching label filter: "HIP_TEST_LEVEL=quick ctest -L quick"
 * selects the category's tests and runs them with its parameters.
 *
 * Anything else - a list, an exclusion, a bracketed tag, an unsupported level,
 * a typo - is rejected and the run aborts, rather than silently falling back to
 * the default parameters. An empty value is treated as unset.
 */
namespace HipTestLevel {

/// Levels understood by the test suite (must match definitions.yaml).
inline constexpr const char* kSupportedLevels[] = {"level_0", "level_1", "level_2", "level_3",
                                                   "level_4"};

/// Level used when HIP_TEST_LEVEL is unset.
inline constexpr const char* kDefaultLevel = "level_2";

/// Which source the active level came from.
enum class LevelSource { kEnvironment, kDefault };

/// @brief Outcome of resolving the active level, including what to report.
struct LevelResolution {
  /// Level whose parameters are used, as "level_N". Always a supported level:
  /// kDefaultLevel when HIP_TEST_LEVEL is unset or rejected.
  std::string level = kDefaultLevel;
  /// Where @ref level came from.
  LevelSource source = LevelSource::kDefault;
  /// Test category HIP_TEST_LEVEL named, empty when it named a level directly.
  std::string category;
  /// Why HIP_TEST_LEVEL was rejected; empty when it was accepted or unset.
  std::string error;

  /// @brief Whether HIP_TEST_LEVEL was rejected. The run must not start.
  bool rejected() const { return !error.empty(); }

  /// @brief Human-readable line saying which level supplies the parameters.
  std::string describe() const;
};

/// @brief Whether @p level is one of kSupportedLevels.
bool isSupportedLevel(const std::string& level);

/**
 * @brief Resolve the active level.
 * @param envLevel value of HIP_TEST_LEVEL, or nullptr when unset.
 */
LevelResolution resolveLevel(const char* envLevel);

}  // namespace HipTestLevel
