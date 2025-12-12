/*
* Copyright © Advanced Micro Devices, Inc., or its affiliates. 
* 
* SPDX-License-Identifier: MIT
*/
#ifndef ROCRTST_SUITES_FUNCTIONAL_FILTER_DEVICES_H_
#define ROCRTST_SUITES_FUNCTIONAL_FILTER_DEVICES_H_

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

class FilterDevicesTest : public TestBase {
 public:
  FilterDevicesTest();

  // @Brief: Destructor for test case of FilterDevicesTest
  virtual ~FilterDevicesTest();

  // @Brief: Setup the environment for FilterDevicesTest
  virtual void SetUp();

  // @Brief: Core test execution
  virtual void Run();

  // @Brief: Clean up and retrive the resource
  virtual void Close();

  // @Brief: Display results
  virtual void DisplayResults() const;

  // @Brief: Display information about what this test does
  virtual void DisplayTestInfo(void);

  // @Brief: Test ROCR_VISIBLE_DEVICES filtering functionality
  void TestRocrVisibleDevicesFiltering();

 private:
  uint32_t initial_gpu_count_;
  std::vector<hsa_agent_t> available_gpus_;
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_FILTER_DEVICES_H_