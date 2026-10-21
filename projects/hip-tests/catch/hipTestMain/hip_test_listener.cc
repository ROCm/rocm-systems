/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <hip_test_params.hh>
#include <hip_test_context.hh>
#include <hip_test_level.hh>
#include <cstdlib>
#include <stdexcept>

/**
 * @brief Loads the active level's test parameters into TestParameterStore.
 *
 * The level only picks parameters; it never narrows the run, so the Catch2 test
 * spec is not consulted. Catch2 never starts a run for --help or the --list-*
 * options, so those work whatever HIP_TEST_LEVEL holds.
 */
class HipTestParameterListener : public Catch::EventListenerBase {
 public:
  using Catch::EventListenerBase::EventListenerBase;

  /**
   * @brief Called once when the test run begins, before any test.
   * Initializes TestParameterStore and loads the resolved level's parameters.
   * @throws std::runtime_error when HIP_TEST_LEVEL is rejected. Catch2 prints
   *         the message and Session::run() returns 1 without running any test,
   *         rather than silently running with the default parameters.
   */
  void testRunStarting(Catch::TestRunInfo const&) override {
    const auto resolution = HipTestLevel::resolveLevel(std::getenv("HIP_TEST_LEVEL"));
    if (resolution.rejected()) {
      throw std::runtime_error("[Test Level] ERROR: " + resolution.error + ". Aborting.");
    }
    LogPrintf("%s\n", resolution.describe().c_str());

    auto& params = TestParameterStore::instance();
    params.initialize();
    params.loadLevelConfig(resolution.level);
  }

  /// @brief Called when the test run ends. Releases the loaded parameters.
  void testRunEnded(Catch::TestRunStats const&) override { TestParameterStore::instance().clear(); }
};

// Register the listener - it will be automatically activated
CATCH_REGISTER_LISTENER(HipTestParameterListener)
