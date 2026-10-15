/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#define CATCH_CONFIG_RUNNER
#include <cmd_options.hh>
#include <hip_test_common.hh>
#include <hip_test_level.hh>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

CmdOptions cmd_options;

namespace {

/// @brief Whether Catch2 will only print information instead of running tests.
bool isListingOnly(const Catch::ConfigData& configData) {
  return configData.showHelp || configData.libIdentify || configData.listTests ||
         configData.listTags || configData.listReporters || configData.listListeners;
}

}  // namespace

int main(int argc, char** argv) {
  auto& context = TestContext::get();

  Catch::Session session;

  using namespace Catch::Clara;
  // clang-format off
  auto cli = session.cli()
    | Opt(cmd_options.no_display)
        ["-S"]["--no-display"]
        ("Do not display the output of performance tests")
    | Opt(cmd_options.progress)
        ["-P"]["--progress"]
        ("Show progress bar when running performance tests")
    | Opt(cmd_options.reduce_iterations, "reduce_iterations")
        ["-R"]["--reduce-iterations"]
        ("Number of iterations for fuzzing reduce operations (default: 1)")
    | Opt(cmd_options.reduce_input_size, "reduce_input_size")
        ["-Z"]["--reduce-input-size"]
        ("Size of the input for the reduce sync operations performance test (megabytes) (default: 50)")
  ;
  // clang-format on

  session.cli(cli);

  int out = session.applyCommandLine(argc, argv);
  if (out == 0) {
    auto& configData = session.configData();
    if (isListingOnly(configData)) {
      // Nothing runs, so the level is irrelevant
      out = session.run();
    } else {
#ifdef ENABLE_YAML_TAGS
      // Test selection by level has to happen here: Catch2 parses the test spec
      // before any listener event fires, so a listener cannot narrow the run.
      // Loading the level's parameters can wait, and does - see
      // hip_test_listener.cc, which is not part of the standalone build.
      //
      // Without YAML tags every TEST_CASE has an empty tag string, so a level
      // filter would match nothing; the whole block is compiled out, which is
      // also what keeps the standalone build free of hip_test_level.cc.
      const auto resolution =
          HipTestLevel::resolveLevel(configData.testsOrTags, std::getenv("HIP_TEST_LEVEL"));

      const std::string unsupported = resolution.firstUnsupportedLevel();
      if (!unsupported.empty()) {
        std::fprintf(stderr, "[Level Filter] ERROR: '%s' is not a supported level. Aborting.\n",
                     unsupported.c_str());
        context.cleanContext();
        return EXIT_FAILURE;
      }

      const bool levelFilterApplied =
          HipTestLevel::applyLevelFilter(configData.testsOrTags, resolution);

      out = session.run();

      const int reported = HipTestLevel::mapLevelFilterExitCode(out, levelFilterApplied);
      if (reported != out) {
        LogPrintf("[Level Filter] No test matched the selected level(s); reporting as skipped%s",
                  "");
        out = reported;
      }
#else
      out = session.run();
#endif
    }
  }

  context.cleanContext();
  return out;
}
