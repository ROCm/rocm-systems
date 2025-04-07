#include <gtest/gtest.h>
#include <string_view>

#include "src/core/hardware/hsa_info.h"

int main(int argc, char** argv) {
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  // Disable ATT test fir gfx10 GPUs until its supported
  // read the command line arguments after above filters so it
  // does not override the command-line --gtest_filter argument
  bool skipInit = false;
  for (int i = 0; i < argc; i++) {
    if (std::string_view("--gtest_list_tests").compare(argv[i]) == 0 ||
        std::string_view("-h").compare(argv[i]) == 0 ||
        std::string_view("--help").compare(argv[i]) == 0) {
      skipInit = true;
      break;
    }
  }
  if (!skipInit) hsa_init();
  testing::InitGoogleTest(&argc, argv);
  // hsa_shut_down(); // Waiting for hsa_shutdown bug to fix
  // Append filter above to disable any problematic test
  return RUN_ALL_TESTS();
}
