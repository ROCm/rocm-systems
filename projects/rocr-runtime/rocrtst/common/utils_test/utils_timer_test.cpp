
#include <iostream>
#include "hsatimer.h"
#include <unistd.h>
#include "utils_timer_test.hpp"

using namespace std;



// Destructor method of test driver
rocrtstUtilsTimerTest::~rocrtstUtilsTimerTest() { }

// Constructor method of test driver
//
// @brief loopCnt number of times to call sleep Api
//
// @brief sleepTimer time to sleep in milliseconds
rocrtstUtilsTimerTest::rocrtstUtilsTimerTest(uint32_t loopCnt, uint32_t sleepTime) :
  loopCnt_(loopCnt), sleepTime_(sleepTime), total_time_(0) { }

// Execute user defined number of sleep calls and collect the
// total time taken by such calls
void rocrtstUtilsTimerTest::run() {

  double time;
  PerfTimer timer;
  uint32_t index = timer.CreateTimer();

  for (uint32_t idx; idx < loopCnt_; idx++) {

    timer.StartTimer(index);
    usleep(sleepTime_);
    timer.StopTimer(index);
    time = timer.ReadTimer(index);
    total_time_ += time;
  }
}

// Print time reported by Hsa Perf Utils Timer service
void rocrtstUtilsTimerTest::print() {

  std::cout << "Time taken by " << loopCnt_;
  std::cout << " iterations of sleep is: " << total_time_ << std::endl;
}
