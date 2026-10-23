// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef TESTS_AMD_SMI_TEST_FUNCTIONAL_MEMORYPARTITION_READ_WRITE_H_
#define TESTS_AMD_SMI_TEST_FUNCTIONAL_MEMORYPARTITION_READ_WRITE_H_

#include <functional>

#include "test_base.h"

class TestMemoryPartitionReadWrite : public TestBase {
 public:
  TestMemoryPartitionReadWrite();

  // @Brief: Destructor for test case of TestMemoryPartitionReadWrite
  virtual ~TestMemoryPartitionReadWrite();

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

 protected:
  struct Api {
    std::function<decltype(amdsmi_get_gpu_accelerator_partition_profile)> get_profile =
        amdsmi_get_gpu_accelerator_partition_profile;
    std::function<decltype(amdsmi_get_gpu_accelerator_partition_profile_config)>
        get_profile_config = amdsmi_get_gpu_accelerator_partition_profile_config;
    std::function<decltype(amdsmi_get_gpu_memory_partition)> get_memory =
        amdsmi_get_gpu_memory_partition;
    std::function<decltype(amdsmi_get_gpu_memory_partition_config)> get_memory_config =
        amdsmi_get_gpu_memory_partition_config;
    std::function<decltype(amdsmi_set_gpu_memory_partition_mode)> set_memory =
        amdsmi_set_gpu_memory_partition_mode;
    std::function<decltype(amdsmi_get_gpu_xcd_counter)> get_xcd_counter =
        amdsmi_get_gpu_xcd_counter;
    std::function<void(amdsmi_processor_handle)> print_device_header;
  } api_;
};

#endif  // TESTS_AMD_SMI_TEST_FUNCTIONAL_MEMORYPARTITION_READ_WRITE_H_
