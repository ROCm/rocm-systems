#include "test_common.hpp"

TestConfig g_config;

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  // Parse custom flags that GoogleTest didn't consume.
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--test-all")
      g_config.testAll = true;
    else if (arg.rfind("--corpus-dir=", 0) == 0)
      g_config.corpusDirs.push_back(arg.substr(13));
    else if (arg.rfind("--corpus-limit=", 0) == 0)
      g_config.corpusLimit = atoi(arg.substr(15).c_str());
    else if (arg.rfind("--raise-dir=", 0) == 0)
      g_config.raiseDirs.push_back(arg.substr(12));
  }

  return RUN_ALL_TESTS();
}
