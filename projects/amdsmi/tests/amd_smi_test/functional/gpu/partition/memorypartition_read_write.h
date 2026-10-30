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

  // Rounds the accelerator profile restore makes; BUSY is retried.
  static constexpr int kProfileAttempts = 10;

  enum class RunResult { kOk, kFailed, kTimedOut, kStuck };

  // Runs `path args...` with a minimal environment and no shell. A child that
  // ignores SIGKILL after the timeout is reported as kStuck instead of awaited.
  static RunResult RunProgram(const std::string& path, const std::vector<std::string>& args,
                              int timeout_sec, int reap_sec, std::string* error);

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
    std::function<decltype(amdsmi_get_gpu_kfd_info)> get_kfd_info = amdsmi_get_gpu_kfd_info;
    std::function<decltype(amdsmi_get_gpu_device_bdf)> get_bdf = amdsmi_get_gpu_device_bdf;
    std::function<decltype(amdsmi_shut_down)> shut_down = amdsmi_shut_down;
    // Host access for the reload cycle.
    std::function<std::string(const std::string&)> read_file;  // empty when unreadable
    std::function<bool(const std::string&)> is_executable;
    std::function<RunResult(const std::string&, const std::vector<std::string>&, std::string*)> run;
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
    bool caps_valid = false;
  };

  struct StageResult {
    size_t accepted = 0;
    size_t total = 0;
    bool complete = true;
  };

  enum class Reload {
    kApplied,      // amdgpu reloaded and the library enumerated the devices again
    kNotUnloaded,  // amdgpu stayed loaded, so a staged mode is still pending
    kNotLoaded,    // amdgpu is not loaded; no device is usable
    kNoDevices,    // amdgpu is live, but the library could not enumerate the devices
  };

  // Shuts the library down, then initializes it and enumerates the devices again.
  // The fixture must hold the only amdsmi_init reference, or the handles go stale.
  virtual bool RefreshHandles(std::string* error);
  // Why amdgpu cannot be reloaded, or empty when it can. Checked before any mode is
  // staged, so a failed check leaves nothing pending. Leaves the library initialized.
  std::string ReloadBlocker();
  Reload ReloadDriver(std::string* error);
  // modprobe at a fixed path, or empty when kmod is not installed there.
  std::string ModprobePath();
  // amdgpu's module refcount, or -1 when it is built in, not loaded, or unreadable.
  int AmdgpuRefcount();
  // Physical GPU key; empty for a handle that is not partition 0.
  std::string PhysicalKey(amdsmi_processor_handle handle);

  std::map<std::string, PhysicalGpu> ReadPhysicalGpus(bool* complete);
  // The mode every GPU reports, or empty when they disagree, a read failed or GPUs are missing.
  static std::string AgreedMode(const std::map<std::string, PhysicalGpu>& gpus, bool complete,
                                size_t physical_count);
  void RunMemoryPartitionCycle();
  StageResult StageOnAll(amdsmi_memory_partition_type_t mode);
  // Reloads to apply `staged`; reports a failure and returns false unless it was applied.
  bool ApplyStaged(const std::string& staged, const std::string& start);
  // Waits for every GPU to report a mode after a reload; returns the agreed mode.
  std::string VerifyMode(const std::string& expected, size_t physical_count);
  void RestoreAcceleratorProfiles(const std::map<std::string, PhysicalGpu>& original);
};

#endif  // TESTS_AMD_SMI_TEST_FUNCTIONAL_MEMORYPARTITION_READ_WRITE_H_
