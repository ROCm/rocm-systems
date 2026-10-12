// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "functional/gpu/partition/memorypartition_read_write.h"

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace {

const char* ModeName(amdsmi_memory_partition_type_t mode) {
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

amdsmi_memory_partition_type_t ModeOf(const std::string& name) {
  for (auto mode : {AMDSMI_MEMORY_PARTITION_NPS1, AMDSMI_MEMORY_PARTITION_NPS2,
                    AMDSMI_MEMORY_PARTITION_NPS4, AMDSMI_MEMORY_PARTITION_NPS8}) {
    if (name == ModeName(mode)) {
      return mode;
    }
  }
  return AMDSMI_MEMORY_PARTITION_UNKNOWN;
}

struct PartitionDevice {
  std::string memory = "NPS2";
  amdsmi_status_t memory_status = AMDSMI_STATUS_SUCCESS;
  amdsmi_accelerator_partition_profile_t profile = {};
  amdsmi_status_t profile_status = AMDSMI_STATUS_SUCCESS;
  amdsmi_memory_partition_type_t staged = AMDSMI_MEMORY_PARTITION_UNKNOWN;
  std::vector<amdsmi_memory_partition_type_t> requests;
  // Runs after every set request, to change the device partway through a run.
  std::function<void(PartitionDevice&, amdsmi_memory_partition_type_t)> on_set;

  bool RequestedValidMode() const {
    return std::any_of(requests.begin(), requests.end(), [](amdsmi_memory_partition_type_t mode) {
      return mode != AMDSMI_MEMORY_PARTITION_UNKNOWN;
    });
  }
};

// Follows the driver: a mode other than the active one stays staged until the next reload,
// and a request for the active mode is ignored.
class MemoryPartitionHarness : public TestMemoryPartitionReadWrite {
 public:
  explicit MemoryPartitionHarness(uint32_t count) : devices(count) {
    set_verbosity(0);
    set_dont_fail(false);
    set_num_monitor_devs(count);
    for (uint32_t i = 0; i < count; ++i) {
      devices[i].profile.profile_type = AMDSMI_ACCELERATOR_PARTITION_SPX;
      processor_handles_[i] = &devices[i];
    }
    api_.get_profile = [](amdsmi_processor_handle handle,
                          amdsmi_accelerator_partition_profile_t* profile, uint32_t* ids) {
      const auto& device = Device(handle);
      *profile = device.profile;
      std::memset(ids, 0, 8 * sizeof(*ids));
      return device.profile_status;
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
    api_.get_memory = [](amdsmi_processor_handle handle, char* value, uint32_t size) {
      const auto& device = Device(handle);
      if (!value || !size) {
        return AMDSMI_STATUS_INVAL;
      }
      if (device.memory_status != AMDSMI_STATUS_SUCCESS) {
        return device.memory_status;
      }
      if (size <= device.memory.size()) {
        return AMDSMI_STATUS_INSUFFICIENT_SIZE;
      }
      std::memcpy(value, device.memory.c_str(), device.memory.size() + 1);
      return AMDSMI_STATUS_SUCCESS;
    };
    api_.get_memory_config = [](amdsmi_processor_handle handle,
                                amdsmi_memory_partition_config_t* config) {
      if (!config) {
        return AMDSMI_STATUS_INVAL;
      }
      *config = {};
      config->mp_mode = ModeOf(Device(handle).memory);
      config->partition_caps.nps_flags.nps1_cap = 1;
      config->partition_caps.nps_flags.nps2_cap = 1;
      config->partition_caps.nps_flags.nps4_cap = 1;
      config->partition_caps.nps_flags.nps8_cap = 1;
      return AMDSMI_STATUS_SUCCESS;
    };
    api_.set_memory = [](amdsmi_processor_handle handle, amdsmi_memory_partition_type_t mode) {
      auto& device = Device(handle);
      device.requests.push_back(mode);
      const bool valid = ModeOf(ModeName(mode)) != AMDSMI_MEMORY_PARTITION_UNKNOWN;
      if (valid && device.memory != ModeName(mode)) {
        device.staged = mode;
      }
      if (device.on_set) {
        device.on_set(device, mode);
      }
      return valid ? AMDSMI_STATUS_SUCCESS : AMDSMI_STATUS_INVAL;
    };
  }

  void PrintDeviceHeader(amdsmi_processor_handle) override {}

  std::vector<PartitionDevice> devices;

 private:
  static PartitionDevice& Device(amdsmi_processor_handle handle) {
    return *static_cast<PartitionDevice*>(handle);
  }
};

int Failures(TestMemoryPartitionReadWrite* test) {
  testing::TestPartResultArray results;
  {
    testing::ScopedFakeTestPartResultReporter reporter(
        testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &results);
    test->Run();
  }
  return results.size();
}

TEST(GpuUnit, MemoryPartitionRequestsOnlyTheActiveMode) {
  MemoryPartitionHarness test(2);
  test.devices[0].memory = "NPS1";
  test.devices[1].memory = "NPS4";
  EXPECT_EQ(Failures(&test), 0);
  for (const auto& device : test.devices) {
    EXPECT_EQ(device.staged, AMDSMI_MEMORY_PARTITION_UNKNOWN) << "left a mode staged";
    EXPECT_NE(std::find(device.requests.begin(), device.requests.end(), ModeOf(device.memory)),
              device.requests.end())
        << "never requested the active mode";
  }
}

TEST(GpuUnit, MemoryPartitionFailsOnChangedOrUnreadableFinalState) {
  const std::vector<std::function<void(PartitionDevice&)>> changes = {
      [](PartitionDevice& device) { device.memory = "NPS4"; },
      [](PartitionDevice& device) {
        device.profile.profile_type = AMDSMI_ACCELERATOR_PARTITION_DPX;
      },
      [](PartitionDevice& device) { device.profile.profile_index = 3; },
      [](PartitionDevice& device) { device.memory_status = AMDSMI_STATUS_FILE_ERROR; },
      [](PartitionDevice& device) { device.profile_status = AMDSMI_STATUS_NO_PERM; },
  };
  for (size_t i = 0; i < changes.size(); ++i) {
    SCOPED_TRACE(::testing::Message() << "change " << i);
    MemoryPartitionHarness test(1);
    test.devices[0].on_set = [&change = changes[i]](PartitionDevice& device,
                                                    amdsmi_memory_partition_type_t mode) {
      if (mode != AMDSMI_MEMORY_PARTITION_UNKNOWN) {
        change(device);
      }
    };
    EXPECT_GT(Failures(&test), 0);
  }
}

TEST(GpuUnit, MemoryPartitionRejectsUnusableSnapshotsWithoutSetting) {
  for (const char* memory : {"", "NPS3"}) {
    SCOPED_TRACE(::testing::Message() << "memory=\"" << memory << "\"");
    MemoryPartitionHarness test(1);
    test.devices[0].memory = memory;
    EXPECT_GT(Failures(&test), 0);
    EXPECT_FALSE(test.devices[0].RequestedValidMode());
  }

  MemoryPartitionHarness test(2);
  test.devices[0].profile_status = AMDSMI_STATUS_FILE_ERROR;
  EXPECT_GT(Failures(&test), 0);
  EXPECT_FALSE(test.devices[0].RequestedValidMode());
  EXPECT_TRUE(test.devices[1].RequestedValidMode()) << "one unreadable device stopped the rest";
}

TEST(GpuUnit, MemoryPartitionDoesNotRequestAfterExternalChange) {
  MemoryPartitionHarness test(1);
  // Another process applies NPS4 after the test read NPS2.
  test.devices[0].on_set = [](PartitionDevice& device, amdsmi_memory_partition_type_t mode) {
    if (mode == AMDSMI_MEMORY_PARTITION_UNKNOWN) {
      device.memory = "NPS4";
    }
  };
  EXPECT_GT(Failures(&test), 0);
  EXPECT_EQ(test.devices[0].staged, AMDSMI_MEMORY_PARTITION_UNKNOWN);
  EXPECT_FALSE(test.devices[0].RequestedValidMode());
}

TEST(GpuUnit, MemoryPartitionPassesWhenUnsupported) {
  MemoryPartitionHarness no_memory(1);
  no_memory.devices[0].memory_status = AMDSMI_STATUS_NOT_SUPPORTED;
  EXPECT_EQ(Failures(&no_memory), 0);
  EXPECT_FALSE(no_memory.devices[0].RequestedValidMode());

  MemoryPartitionHarness no_compute(1);
  no_compute.devices[0].profile_status = AMDSMI_STATUS_NOT_SUPPORTED;
  EXPECT_EQ(Failures(&no_compute), 0);
  EXPECT_TRUE(no_compute.devices[0].RequestedValidMode());
}

}  // namespace
