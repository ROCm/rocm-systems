// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "functional/gpu/partition/memorypartition_read_write.h"

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

struct PartitionDevice {
  amdsmi_accelerator_partition_profile_t profile = {};
  amdsmi_status_t initial_status = AMDSMI_STATUS_SUCCESS;
  amdsmi_accelerator_partition_type_t final_type = AMDSMI_ACCELERATOR_PARTITION_QPX;
  amdsmi_status_t final_status = AMDSMI_STATUS_SUCCESS;
  uint32_t final_index = 2;
  uint32_t profile_reads = 0;
  std::string memory = "NPS2";
  bool unterminated_memory = false;
  uint32_t memory_reads = 0;
  uint32_t fail_memory_read = 0;
  amdsmi_status_t memory_status = AMDSMI_STATUS_SUCCESS;
  std::string memory_after_set;
  std::string memory_before_set;
  amdsmi_memory_partition_type_t config_mode = AMDSMI_MEMORY_PARTITION_NPS2;
  amdsmi_status_t set_status = AMDSMI_STATUS_SUCCESS;
  amdsmi_memory_partition_type_t pending = AMDSMI_MEMORY_PARTITION_UNKNOWN;
  std::vector<amdsmi_memory_partition_type_t> memory_requests;
};

const char* MemoryModeName(amdsmi_memory_partition_type_t mode) {
  switch (mode) {
    case AMDSMI_MEMORY_PARTITION_NPS1:
      return "NPS1";
    case AMDSMI_MEMORY_PARTITION_NPS2:
      return "NPS2";
    case AMDSMI_MEMORY_PARTITION_NPS4:
      return "NPS4";
    case AMDSMI_MEMORY_PARTITION_NPS8:
      return "NPS8";
    default:
      return "UNKNOWN";
  }
}

class MemoryPartitionHarness : public TestMemoryPartitionReadWrite {
 public:
  explicit MemoryPartitionHarness(uint32_t count = 1) : devices(count) {
    set_verbosity(0);
    set_dont_fail(false);
    set_num_monitor_devs(count);
    for (uint32_t i = 0; i < count; ++i) {
      processor_handles_[i] = &devices[i];
      devices[i].profile.profile_type = AMDSMI_ACCELERATOR_PARTITION_QPX;
      devices[i].profile.profile_index = 2;
    }
    api_.print_device_header = [](amdsmi_processor_handle) {};
    api_.get_profile = [](amdsmi_processor_handle handle,
                          amdsmi_accelerator_partition_profile_t* profile, uint32_t* ids) {
      auto& device = *static_cast<PartitionDevice*>(handle);
      *profile = device.profile;
      std::memset(ids, 0, 8 * sizeof(*ids));
      if (device.profile_reads++ != 0) {
        profile->profile_type = device.final_type;
        profile->profile_index = device.final_index;
        return device.final_status;
      }
      return device.initial_status;
    };
    api_.get_profile_config = [](amdsmi_processor_handle,
                                 amdsmi_accelerator_partition_profile_config_t* config) {
      *config = {};
      return AMDSMI_STATUS_SUCCESS;
    };
    api_.get_xcd_counter = [](amdsmi_processor_handle, uint16_t* count) {
      *count = 8;
      return AMDSMI_STATUS_SUCCESS;
    };
    api_.get_memory = [this](amdsmi_processor_handle handle, char* value, uint32_t size) {
      if (!value || !size) return AMDSMI_STATUS_INVAL;
      if (change_monitor_count) set_num_monitor_devs(0);
      auto& device = *static_cast<PartitionDevice*>(handle);
      if (size == 255 && ++device.memory_reads == device.fail_memory_read) {
        return device.memory_status;
      }
      if (device.memory_reads == 2 && !device.memory_before_set.empty()) {
        device.memory = device.memory_before_set;
      }
      if (device.unterminated_memory) {
        std::memset(value, 'X', size);
        return AMDSMI_STATUS_SUCCESS;
      }
      const auto& memory = device.memory;
      if (size <= memory.size()) return AMDSMI_STATUS_INSUFFICIENT_SIZE;
      std::memcpy(value, memory.c_str(), memory.size() + 1);
      return AMDSMI_STATUS_SUCCESS;
    };
    api_.get_memory_config = [](amdsmi_processor_handle handle,
                                amdsmi_memory_partition_config_t* config) {
      if (!config) return AMDSMI_STATUS_INVAL;
      *config = {};
      config->mp_mode = static_cast<PartitionDevice*>(handle)->config_mode;
      config->partition_caps.nps_flags.nps1_cap = 1;
      config->partition_caps.nps_flags.nps2_cap = 1;
      config->partition_caps.nps_flags.nps4_cap = 1;
      config->partition_caps.nps_flags.nps8_cap = 1;
      return AMDSMI_STATUS_SUCCESS;
    };
    api_.set_memory = [](amdsmi_processor_handle handle, amdsmi_memory_partition_type_t mode) {
      auto& device = *static_cast<PartitionDevice*>(handle);
      device.memory_requests.push_back(mode);
      if (mode == AMDSMI_MEMORY_PARTITION_UNKNOWN) return AMDSMI_STATUS_INVAL;
      if (device.set_status != AMDSMI_STATUS_SUCCESS) return device.set_status;
      // Active-mode requests do not cancel a previously staged alternate mode.
      if (MemoryModeName(mode) != device.memory) device.pending = mode;
      if (!device.memory_after_set.empty()) device.memory = device.memory_after_set;
      return AMDSMI_STATUS_SUCCESS;
    };
  }

  void UseNullHandle() {
    processor_handles_[0] = nullptr;
    api_.get_profile = [this](amdsmi_processor_handle, amdsmi_accelerator_partition_profile_t*,
                              uint32_t*) {
      ++null_handle_calls;
      return AMDSMI_STATUS_NOT_SUPPORTED;
    };
    api_.get_memory = [this](amdsmi_processor_handle, char*, uint32_t) {
      ++null_handle_calls;
      return AMDSMI_STATUS_NOT_SUPPORTED;
    };
  }

  std::vector<PartitionDevice> devices;
  int null_handle_calls = 0;
  bool change_monitor_count = false;
};

int FailureCount(TestMemoryPartitionReadWrite* test) {
  testing::TestPartResultArray results;
  {
    testing::ScopedFakeTestPartResultReporter reporter(
        testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &results);
    test->Run();
  }
  return results.size();
}

TEST(GpuUnit, MemoryPartitionOnlyRequestsCurrentMode) {
  MemoryPartitionHarness test;
  test.Run();
  for (auto mode : test.devices[0].memory_requests) {
    EXPECT_TRUE(mode == AMDSMI_MEMORY_PARTITION_UNKNOWN || mode == AMDSMI_MEMORY_PARTITION_NPS2);
  }
  EXPECT_EQ(test.devices[0].pending, AMDSMI_MEMORY_PARTITION_UNKNOWN);
}

TEST(GpuUnit, MemoryPartitionVerifiesEveryComputeSnapshotWithoutRepair) {
  for (uint32_t verbosity = 0; verbosity <= 2; ++verbosity) {
    MemoryPartitionHarness test(2);
    test.set_verbosity(verbosity);
    test.devices[0].final_type = AMDSMI_ACCELERATOR_PARTITION_DPX;
    test.devices[1].final_status = AMDSMI_STATUS_NO_PERM;
    EXPECT_EQ(FailureCount(&test), 2);
    for (const auto& device : test.devices) {
      EXPECT_EQ(device.profile_reads, 2);
    }
  }
}

TEST(GpuUnit, MemoryPartitionRejectsInvalidMemorySnapshot) {
  for (const auto& mode : {"", "NPS3"}) {
    MemoryPartitionHarness test;
    test.devices[0].memory = mode;
    EXPECT_EQ(FailureCount(&test), 1);
    EXPECT_TRUE(test.devices[0].memory_requests.empty());
  }
}

TEST(GpuUnit, MemoryPartitionRejectsFailedComputeSnapshotBeforeOtherQueries) {
  MemoryPartitionHarness test(2);
  test.devices[0].profile.profile_type = AMDSMI_ACCELERATOR_PARTITION_CPX;
  test.devices[0].initial_status = AMDSMI_STATUS_FILE_ERROR;
  EXPECT_EQ(FailureCount(&test), 1);
  EXPECT_TRUE(test.devices[0].memory_requests.empty());
  EXPECT_EQ(test.devices[0].profile_reads, 1);
  EXPECT_FALSE(test.devices[1].memory_requests.empty());
  EXPECT_EQ(test.devices[1].profile_reads, 2);
}

TEST(GpuUnit, MemoryPartitionRejectsInvalidComputeType) {
  MemoryPartitionHarness test;
  test.devices[0].profile.profile_type = AMDSMI_ACCELERATOR_PARTITION_INVALID;
  test.devices[0].final_type = AMDSMI_ACCELERATOR_PARTITION_INVALID;
  EXPECT_EQ(FailureCount(&test), 1);
  EXPECT_EQ(test.devices[0].profile_reads, 1);
  EXPECT_TRUE(test.devices[0].memory_requests.empty());
}

TEST(GpuUnit, MemoryPartitionVerifiesEveryMemorySnapshot) {
  MemoryPartitionHarness test(2);
  test.devices[0].memory_after_set = "NPS4";
  test.devices[1].fail_memory_read = 3;
  test.devices[1].memory_status = AMDSMI_STATUS_INVAL;
  EXPECT_EQ(FailureCount(&test), 2);
  for (const auto& device : test.devices) {
    EXPECT_EQ(device.memory_reads, 3);
    EXPECT_EQ(device.profile_reads, 2);
  }
}

TEST(GpuUnit, MemoryPartitionDoesNotOverwriteConcurrentMemoryChange) {
  MemoryPartitionHarness test;
  test.devices[0].memory_before_set = "NPS4";
  EXPECT_EQ(FailureCount(&test), 2);
  EXPECT_EQ(test.devices[0].memory_requests,
            (std::vector<amdsmi_memory_partition_type_t>{AMDSMI_MEMORY_PARTITION_UNKNOWN,
                                                         AMDSMI_MEMORY_PARTITION_UNKNOWN}));
  EXPECT_EQ(test.devices[0].memory, "NPS4");
}

TEST(GpuUnit, MemoryPartitionRejectsNullHandleBeforeApiCalls) {
  MemoryPartitionHarness test;
  test.UseNullHandle();
  EXPECT_EQ(FailureCount(&test), 1);
  EXPECT_EQ(test.null_handle_calls, 0);
}

TEST(GpuUnit, MemoryPartitionKeepsInitialDeviceCountDuringQueries) {
  MemoryPartitionHarness test(2);
  test.change_monitor_count = true;
  test.Run();
  EXPECT_EQ(test.num_monitor_devs(), 0);
  for (const auto& device : test.devices) {
    EXPECT_EQ(device.memory_requests.size(), 3);
    EXPECT_EQ(device.memory_reads, 3);
  }
}

TEST(GpuUnit, MemoryPartitionFailedMemorySnapshotContinuesOtherDevices) {
  MemoryPartitionHarness test(2);
  test.set_dont_fail(true);
  test.devices[0].fail_memory_read = 1;
  test.devices[0].memory_status = AMDSMI_STATUS_FILE_ERROR;
  EXPECT_EQ(FailureCount(&test), 1);
  EXPECT_TRUE(test.devices[0].memory_requests.empty());
  EXPECT_EQ(test.devices[1].memory_requests.size(), 3);
  EXPECT_EQ(test.devices[0].profile_reads, 2);
  EXPECT_EQ(test.devices[1].profile_reads, 2);
}

TEST(GpuUnit, MemoryPartitionRejectsObservedConfigDriftBeforeSet) {
  MemoryPartitionHarness test;
  test.devices[0].config_mode = AMDSMI_MEMORY_PARTITION_NPS4;
  EXPECT_EQ(FailureCount(&test), 1);
  EXPECT_EQ(test.devices[0].memory_requests,
            (std::vector<amdsmi_memory_partition_type_t>{AMDSMI_MEMORY_PARTITION_UNKNOWN,
                                                         AMDSMI_MEMORY_PARTITION_UNKNOWN}));
}

TEST(GpuUnit, MemoryPartitionPreservesMixedStartingModes) {
  MemoryPartitionHarness test(2);
  test.devices[0].memory = "NPS1";
  test.devices[0].config_mode = AMDSMI_MEMORY_PARTITION_NPS1;
  test.devices[0].profile.profile_type = AMDSMI_ACCELERATOR_PARTITION_DPX;
  test.devices[0].final_type = AMDSMI_ACCELERATOR_PARTITION_DPX;
  test.devices[0].profile.profile_index = 1;
  test.devices[0].final_index = 1;
  test.devices[1].memory = "NPS4";
  test.devices[1].config_mode = AMDSMI_MEMORY_PARTITION_NPS4;
  test.Run();
  for (const auto& device : test.devices) {
    EXPECT_EQ(device.memory_requests, (std::vector<amdsmi_memory_partition_type_t>{
                                          AMDSMI_MEMORY_PARTITION_UNKNOWN,
                                          AMDSMI_MEMORY_PARTITION_UNKNOWN, device.config_mode}));
    EXPECT_EQ(device.pending, AMDSMI_MEMORY_PARTITION_UNKNOWN);
  }
}

TEST(GpuUnit, MemoryPartitionRejectsUnterminatedMemorySnapshot) {
  MemoryPartitionHarness test;
  test.devices[0].unterminated_memory = true;
  EXPECT_EQ(FailureCount(&test), 1);
  EXPECT_TRUE(test.devices[0].memory_requests.empty());
}

TEST(GpuUnit, MemoryPartitionDoesNotInventUnsupportedComputeSnapshot) {
  MemoryPartitionHarness test;
  test.devices[0].initial_status = AMDSMI_STATUS_NOT_SUPPORTED;
  test.Run();
  EXPECT_EQ(test.devices[0].profile_reads, 1);
  EXPECT_EQ(test.devices[0].memory_requests.size(), 3);
}

TEST(GpuUnit, MemoryPartitionDoesNotInventUnsupportedMemorySnapshot) {
  MemoryPartitionHarness test;
  test.devices[0].fail_memory_read = 1;
  test.devices[0].memory_status = AMDSMI_STATUS_NOT_SUPPORTED;
  test.Run();
  EXPECT_EQ(test.devices[0].memory_reads, 1);
  EXPECT_TRUE(test.devices[0].memory_requests.empty());
  EXPECT_EQ(test.devices[0].profile_reads, 2);
}

TEST(GpuUnit, MemoryPartitionChecksStateAfterUnsupportedSet) {
  MemoryPartitionHarness test;
  test.devices[0].set_status = AMDSMI_STATUS_NOT_SUPPORTED;
  test.Run();
  EXPECT_EQ(test.devices[0].memory_reads, 3);
  EXPECT_EQ(test.devices[0].profile_reads, 2);
}

TEST(GpuUnit, MemoryPartitionAcceptsUnavailableProfileIndexWithoutSettingIt) {
  MemoryPartitionHarness test;
  test.devices[0].profile.profile_index = std::numeric_limits<uint32_t>::max();
  test.devices[0].final_index = std::numeric_limits<uint32_t>::max();
  test.Run();
  EXPECT_EQ(test.devices[0].profile_reads, 2);
}

TEST(GpuUnit, MemoryPartitionRejectsProfileIndexDrift) {
  MemoryPartitionHarness test;
  test.devices[0].final_index = 3;
  EXPECT_EQ(FailureCount(&test), 1);
}

TEST(GpuUnit, MemoryPartitionHandlesNoDevices) {
  MemoryPartitionHarness test(0);
  test.Run();
}

}  // namespace
