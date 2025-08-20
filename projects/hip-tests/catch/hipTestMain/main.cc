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

  using namespace Catch::clara;
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
    | Opt(cmd_options.accuracy_iterations, "accuracy_iterations")
        ["-A"]["--accuracy-iterations"]
        ("Number of iterations used for math accuracy tests with randomly generated inputs (default: 2^32)")
    | Opt(cmd_options.accuracy_max_memory, "accuracy_max_memory")
        ["-M"]["--accuracy-max-memory"]
        ("Percentage of global device memory allowed for math accuracy tests (default: 80%)")
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
  TestContext::get().cleanContext();
  return out;
}
