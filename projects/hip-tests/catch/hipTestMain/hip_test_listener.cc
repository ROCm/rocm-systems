/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <catch2/interfaces/catch_interfaces_config.hpp>
#include <hip_test_params.hh>
#include <hip_test_context.hh>
#include <hip_test_level.hh>
#include <string>
#include <vector>
#include <cstdlib>

/**
 * @brief Loads the active level's test parameters into TestParameterStore.
 *
 * Only the parameters live here. Test *selection* cannot: Catch2 has already
 * parsed the test spec by the time any listener event fires, so the spec is
 * rewritten in main() instead (see HipTestLevel::applyLevelFilter).
 *
 * The level is resolved the same way main() resolves it, from the same two
 * inputs, so the two agree. Where main() rewrote the spec, that rewrite names
 * the levels it applied, so resolving again from the rewritten spec yields the
 * same answer.
 */
class HipTestParameterListener : public Catch::EventListenerBase {
 public:
  using Catch::EventListenerBase::EventListenerBase;

  /**
   * @brief Called once when the test run begins.
   * Initializes TestParameterStore and loads the resolved level's parameters.
   */
  void testRunStarting(Catch::TestRunInfo const&) override {
    const std::vector<std::string> testsOrTags =
        (m_config != nullptr) ? m_config->getTestsOrTags() : std::vector<std::string>{};
    const auto resolution = HipTestLevel::resolveLevel(testsOrTags, std::getenv("HIP_TEST_LEVEL"));

    for (const auto& line : resolution.describe()) {
      LogPrintf("%s\n", line.c_str());
    }

    // An unsupported level is rejected in main() before the run starts; by the
    // time we get here the level is known good, so load it and move on.
    auto& params = TestParameterStore::instance();
    params.initialize();
    LogPrintf("[Level Filter] Applying test parameters level: %s\n", resolution.level.c_str());
    params.loadLevelConfig(resolution.level);
  }

  /// @brief Called when the test run ends. Releases the loaded parameters.
  void testRunEnded(Catch::TestRunStats const&) override { TestParameterStore::instance().clear(); }
};

// Register the listener - it will be automatically activated
CATCH_REGISTER_LISTENER(HipTestParameterListener)
