#include <gtest/gtest.h>

// Entry Point for Gtests Infra

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  //testing::GTEST_FLAG(filter)="-HSATest.*";
  return RUN_ALL_TESTS();
}
