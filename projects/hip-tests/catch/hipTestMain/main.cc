/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#define CATCH_CONFIG_RUNNER
#include <cmd_options.hh>
#include <hip_test_common.hh>
#include <iostream>

CmdOptions cmd_options;

int main(int argc, char** argv) {
  auto& context = TestContext::get(argc, argv);
  if (context.skipTest()) {
    // CTest uses this regex to figure out if the test has been skipped
    std::cout << "HIP_SKIP_THIS_TEST" << std::endl;
    return 0;
  }

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

  int out = session.run(argc, argv);
  TestContext::get().cleanContext();
  return out;
}
