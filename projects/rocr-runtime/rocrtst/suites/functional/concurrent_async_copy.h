/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2024, Advanced Micro Devices, Inc.
 * All rights reserved.
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_CONCURRENT_ASYNC_COPY_H_
#define ROCRTST_SUITES_FUNCTIONAL_CONCURRENT_ASYNC_COPY_H_

#include <pthread.h>
#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

/**
 * @brief Test class for concurrent async copy operations
 *
 * This test reproduces the TSAN race condition on sdma_blit_used_mask_
 * by spawning multiple threads that concurrently perform async memory
 * copies, all targeting the same GPU agent.
 *
 * Expected TSAN report (before fix):
 *   Read/Write race at amd_gpu_agent.cpp:2292 in GetBlitObject()
 */
class ConcurrentAsyncCopy : public TestBase {
 public:
  ConcurrentAsyncCopy(void);
  virtual ~ConcurrentAsyncCopy(void);

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

  void TestConcurrentAsyncCopy(void);
  void TestSdmaDoorbellRace(void);
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_CONCURRENT_ASYNC_COPY_H_
