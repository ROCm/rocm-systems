// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef TESTS_AMD_SMI_TEST_FUNCTIONAL_POWER_CAP_READ_WRITE_H_
#define TESTS_AMD_SMI_TEST_FUNCTIONAL_POWER_CAP_READ_WRITE_H_

#include <string>

#include "test_base.h"

class TestPowerCapReadWrite : public TestBase {
 public:
  TestPowerCapReadWrite();

  void SetCheckPowerCap(std::string msg, uint32_t dv_ind, uint32_t sensor_ind, uint64_t& curr_cap,
                        uint64_t& new_cap, amdsmi_status_t& ret);

  // @Brief: Destructor for test case of TestPowerCapReadWrite
  virtual ~TestPowerCapReadWrite();

  // @Brief: Setup the environment for measurement
  virtual void SetUp();

  // @Brief: Core measurement execution
  virtual void Run();

  // @Brief: Clean up and retrieve the resource
  virtual void Close();

  // @Brief: Display  results
  virtual void DisplayResults() const;

  // @Brief: Display information about what this test does
  virtual void DisplayTestInfo(void);
};

#endif  // TESTS_AMD_SMI_TEST_FUNCTIONAL_POWER_CAP_READ_WRITE_H_
