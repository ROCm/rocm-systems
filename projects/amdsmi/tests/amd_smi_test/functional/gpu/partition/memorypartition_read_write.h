// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef TESTS_AMD_SMI_TEST_FUNCTIONAL_MEMORYPARTITION_READ_WRITE_H_
#define TESTS_AMD_SMI_TEST_FUNCTIONAL_MEMORYPARTITION_READ_WRITE_H_

#include <functional>
#include <map>
#include <string>
#include <vector>

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
    std::function<decltype(amdsmi_set_gpu_accelerator_partition_profile)> set_profile =
        amdsmi_set_gpu_accelerator_partition_profile;
    std::function<void(uint32_t)> sleep_ms;
  } api_;

  // Applying an NPS mode needs a driver reload, so the cycle only runs when
  // AMDSMI_ALLOW_DESTRUCTIVE_TESTS=1, the same opt-in as test_memory_partition_cycle.py.
  bool allow_destructive_ = false;

  // One entry per physical GPU, read from its partition 0 handle.
  struct PhysicalGpu {
    amdsmi_processor_handle handle = nullptr;
    std::string memory;
    amdsmi_accelerator_partition_profile_t profile = {};
    bool profile_valid = false;
    amdsmi_nps_caps_t caps = {};
  };

  // Re-reads every device after amdsmi_init; false when the library is unavailable.
  virtual bool RefreshHandles(std::string* error);
  // False when another process holds amdgpu, so a staged mode could not be applied.
  virtual bool DriverIdle(std::string* error);
  // Applies a staged NPS mode. On failure before the unload the library is initialized
  // again and the mode stays staged; if amdgpu does not load again, no devices remain.
  virtual bool ReloadDriver(std::string* error);
  // Physical GPU key; empty for a handle that is not partition 0.
  virtual std::string PhysicalKey(amdsmi_processor_handle handle);

  std::map<std::string, PhysicalGpu> ReadPhysicalGpus(bool* complete);
  void RunMemoryPartitionCycle();
  bool StageOnAll(amdsmi_memory_partition_type_t mode, bool* any_staged);
  // The mode every GPU reports after a reload, or empty when they disagree or are unreadable.
  std::string VerifyMode(const std::string& expected, size_t physical_count);
  void RestoreAcceleratorProfiles(const std::map<std::string, PhysicalGpu>& original);
};

#endif  // TESTS_AMD_SMI_TEST_FUNCTIONAL_MEMORYPARTITION_READ_WRITE_H_
