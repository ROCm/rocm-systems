/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ROCRTST_SUITES_FUNCTIONAL_MEMORY_FILL_H_
#define ROCRTST_SUITES_FUNCTIONAL_MEMORY_FILL_H_


#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

class MemoryFill : public TestBase {
 public:
  explicit MemoryFill(void);

  // @Brief: Destructor for test case of MemoryFill
  virtual ~MemoryFill();

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

  void MemoryFillTest(void);


 private:
  void MemoryFillTest(hsa_agent_t cpuAgent, hsa_agent_t gpuAgent);
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_MEMORY_FILL_H_

