#ifndef ROCRTST_UTILS_TIMER_TEST_H_
#define ROCRTST_UTILS_TIMER_TEST_H_

// Encapsulates Api's to access Timer service of rocrtst Utils library
class rocrtstUtilsTimerTest {

 public:

  // Destructor method of test driver
  ~rocrtstUtilsTimerTest();

  // Constructor method of test driver
  //
  // @brief loopCnt number of times to call sleep Api
  //
  // @brief sleepTimer time to sleep in milliseconds
  rocrtstUtilsTimerTest(uint32_t loopCnt, uint32_t sleepTime);

  // Execute user defined number of sleep calls and collect the
  // total time taken by such calls
  void run();

  // Print time reported by rocrtst Utils Timer service
  void print();

 private:

  // Number of times to invoke sleep Api
  uint32_t loopCnt_;

  // Time to sleep per cycle, in milliseconds
  uint32_t sleepTime_;

  // Time taken by sleep Api
  double total_time_;
};

#endif
