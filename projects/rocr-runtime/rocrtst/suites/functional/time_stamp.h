/*
* Copyright © Advanced Micro Devices, Inc., or its affiliates.
*
* SPDX-License-Identifier: MIT
*/
#ifndef ROCRTST_SUITES_FUNCTIONAL_TIME_STAMP_H_
#define ROCRTST_SUITES_FUNCTIONAL_TIME_STAMP_H_


#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

class TimeStamp : public TestBase {
 public:
    TimeStamp();

  // @Brief: Destructor for test case of TimeStamp
  virtual ~TimeStamp();

  // @Brief: Setup the environment for measurement
  virtual void SetUp();

  // @Brief: Core measurement execution
  virtual void Run();

  // @Brief: Clean up and retrive the resource
  virtual void Close();

  // @Brief: Display  results
  virtual void DisplayResults() const;

  // @Brief: Display information about what this test does
  virtual void DisplayTestInfo(void);

  void TimeStampTest (void);

  void BarrierPacketTimestampValidationTest(void);
private:
  void FreeResources();  // Helper to clean up queue and signals

  const static uint32_t NUM_BARRIERS = 100;
  hsa_signal_t completion_signals[NUM_BARRIERS] = {0};
  hsa_queue_t *queue = NULL;
  bool resources_free = false;  // Track if cleanup happened
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_TIME_STAMP_H_
