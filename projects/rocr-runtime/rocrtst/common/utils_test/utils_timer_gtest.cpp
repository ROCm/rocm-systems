

#include <iostream>

#include "gtest/gtest.h"

#include "utils_timer_test.hpp"

using namespace std;

class rocrtstUtilsTimerGtest : public ::testing::Test {

 protected:

  // No argument constructor called from Google Test Framework
  rocrtstUtilsTimerGtest() { };

};

TEST_F(rocrtstUtilsTimerGtest, TestingTimer101) {

  // Create a Hsa Perf Utils Timer Test object.
  // The test will iterate 108 times with sleep
  // time of 3 milliseconds per iteration
  rocrtstUtilsTimerTest* timer = new rocrtstUtilsTimerTest(108, 3);

  // Let the timer object collect data
  timer->run();

  // Print the statistics of timer object
  timer->print();
}
