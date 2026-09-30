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
    | Opt(cmd_options.iterations, "iterations")
        ["-I"]["--iterations"]
        ("Number of iterations used for performance tests (default: 1000)")
    | Opt(cmd_options.warmups, "warmups")
        ["-W"]["--warmups"]
        ("Number of warmup iterations used for performance tests (default: 100)")
    | Opt(cmd_options.no_display)
        ["-S"]["--no-display"]
        ("Do not display the output of performance tests")
    | Opt(cmd_options.progress)
        ["-P"]["--progress"]
        ("Show progress bar when running performance tests")
    | Opt(cmd_options.cg_iterations, "cg_iterations")
        ["-C"]["--cg-iterations"]
        ("Number of iterations used for cooperative groups sync tests (default: 5)")
    | Opt(cmd_options.cg_reduction_factor, "cg_reduction_factor")
        ["-C"]["--cg-reduction-factor"]
        ("Percentage of warp sizes for shuffle tests to be actually tested (default: 10)")
    | Opt(cmd_options.warp_reduction_factor, "warp_reduction_factor")
        ["-F"]["--warp-reduction-factor"]
        ("Percentage of lane mask iterations for warp shuffle tests to be tested (default: 6.25)")
    | Opt(cmd_options.accuracy_iterations, "accuracy_iterations")
        ["-A"]["--accuracy-iterations"]
        ("Number of iterations used for math accuracy tests with randomly generated inputs (default: 2^32)")
    | Opt(cmd_options.accuracy_max_memory, "accuracy_max_memory")
        ["-M"]["--accuracy-max-memory"]
        ("Percentage of global device memory allowed for math accuracy tests in case the global device memory is lower than max_memory (default: 80%)")
    | Opt(cmd_options.reduce_iterations, "reduce_iterations")
        ["-R"]["--reduce-iterations"]
        ("Number of iterations for fuzzing reduce operations (default: 1)")
    | Opt(cmd_options.reduce_input_size, "reduce_input_size")
        ["-Z"]["--reduce-input-size"]
        ("Size of the input for the reduce sync operations performance test (megabytes) (default: 50)")
    | Opt(cmd_options.max_memory, "max_memory")
        ["-X"]["--max-memory"]
        ("Maximum amount of memory to use for math accuracy tests (default: 2GB)")
    | Opt(cmd_options.reduction_factor, "reduction_factor")
        ["-R"]["--reduction-factor"]
        ("Percentage of test data to be actually tested (default: 0.1%)")
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
