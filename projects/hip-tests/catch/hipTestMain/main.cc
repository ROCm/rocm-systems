/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#define CATCH_CONFIG_RUNNER
#include <cmd_options.hh>
#include <hip_test_common.hh>

CmdOptions cmd_options;

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

  int out = session.run(argc, argv);
  context.cleanContext();
  return out;
}
