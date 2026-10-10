// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "memorypartition_read_write.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "test_base.h"
#include "test_common.h"

const uint32_t MAX_UNSUPPORTED_PARTITIONS = 0;
const uint32_t MAX_SPX_PARTITIONS = 1;  // Single GPU node
const uint32_t MAX_DPX_PARTITIONS = 2;
const uint32_t MAX_TPX_PARTITIONS = 3;
const uint32_t MAX_QPX_PARTITIONS = 4;

TestMemoryPartitionReadWrite::TestMemoryPartitionReadWrite() : TestBase() {
  api_.print_device_header = [this](amdsmi_processor_handle handle) { PrintDeviceHeader(handle); };
  set_title("AMDSMI Memory Partition Read Test");
  set_description(
      "Verify memory partition reads, capabilities, invalid inputs, and current-mode sets. "
      "Alternate NPS transitions require a separate isolated lifecycle harness.");
}

TestMemoryPartitionReadWrite::~TestMemoryPartitionReadWrite(void) {}

void TestMemoryPartitionReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestMemoryPartitionReadWrite::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestMemoryPartitionReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestMemoryPartitionReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

static const std::string memoryPartitionString(amdsmi_memory_partition_type_t memoryPartitionType) {
  switch (memoryPartitionType) {
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

void TestMemoryPartitionReadWrite::Run(void) {
  amdsmi_status_t ret, err, ret_set;
  constexpr uint32_t k255Len = 255;
  constexpr uint32_t k0Len = 0;
  char orig_memory_partition[k255Len] = {};
  amdsmi_memory_partition_config_t current_memory_config;
  const uint32_t kMAX_UINT32 = std::numeric_limits<uint32_t>::max();
  std::map<amdsmi_processor_handle, amdsmi_accelerator_partition_profile_t> original_profiles;
  std::map<amdsmi_processor_handle, std::string> original_memory_modes;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  bool isVerbose =
      (this->verbosity() && this->verbosity() >= (this->TestBase::VERBOSE_STANDARD)) ? true : false;

  // Save each device independently; mixed compute configurations are valid.
  IF_VERB(STANDARD) {
    std::cout << "\t**=========================================================\n";
    std::cout << "\t**Save Original Compute Partition Settings ================\n";
    std::cout << "\t**=========================================================\n";
  }
  auto initial_num_devices = num_monitor_devs();
  ASSERT_LE(initial_num_devices, MAX_MONITOR_DEVICES);
  std::vector<bool> safe_to_test(initial_num_devices, true);
  amdsmi_accelerator_partition_type_t primary_partition_type = AMDSMI_ACCELERATOR_PARTITION_INVALID;
  uint32_t primary_index = 0;
  for (uint32_t dv_ind = 0; dv_ind < initial_num_devices; ++dv_ind) {
    EXPECT_NE(processor_handles_[dv_ind], nullptr) << "Missing processor handle";
    if (processor_handles_[dv_ind] == nullptr) {
      safe_to_test[dv_ind] = false;
      continue;
    }
    if (dv_ind != 0) {
      std::cout << "\n";
    }
    api_.print_device_header(processor_handles_[dv_ind]);
    amdsmi_accelerator_partition_profile_t profile = {};
    uint32_t partition_id[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_profile", "", VERB(STANDARD));
    ret = api_.get_profile(processor_handles_[dv_ind], &profile, &partition_id[0]);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) continue;
    EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS) << "Compute partition snapshot failed";
    if (ret != AMDSMI_STATUS_SUCCESS) {
      safe_to_test[dv_ind] = false;
      continue;
    }
    const bool valid_type = profile.profile_type >= AMDSMI_ACCELERATOR_PARTITION_SPX &&
                            profile.profile_type <= AMDSMI_ACCELERATOR_PARTITION_CPX;
    EXPECT_TRUE(valid_type) << "Invalid compute partition snapshot";
    if (!valid_type) {
      safe_to_test[dv_ind] = false;
      continue;
    }
    original_profiles[processor_handles_[dv_ind]] = profile;
    std::string nps_caps_str = "";
    if ((profile.memory_caps.nps_flags.nps1_cap == 0 &&
         profile.memory_caps.nps_flags.nps2_cap == 0 &&
         profile.memory_caps.nps_flags.nps4_cap == 0 &&
         profile.memory_caps.nps_flags.nps8_cap == 0)) {
      nps_caps_str = "N/A";
    } else {
      nps_caps_str.clear();
      if (profile.memory_caps.nps_flags.nps1_cap) {
        (nps_caps_str.empty()) ? nps_caps_str += "NPS1" : nps_caps_str += ", NPS1";
      }
      if (profile.memory_caps.nps_flags.nps2_cap) {
        (nps_caps_str.empty()) ? nps_caps_str += "NPS2" : nps_caps_str += ", NPS2";
      }
      if (profile.memory_caps.nps_flags.nps4_cap) {
        (nps_caps_str.empty()) ? nps_caps_str += "NPS4" : nps_caps_str += ", NPS4";
      }
      if (profile.memory_caps.nps_flags.nps8_cap) {
        (nps_caps_str.empty()) ? nps_caps_str += "NPS8" : nps_caps_str += ", NPS8";
      }
    }

    std::string profile_type_str = "N/A";
    if (profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_SPX) {
      profile_type_str = "SPX";
    } else if (profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_DPX) {
      profile_type_str = "DPX";
    } else if (profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_TPX) {
      profile_type_str = "TPX";
    } else if (profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_QPX) {
      profile_type_str = "QPX";
    } else if (profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_CPX) {
      profile_type_str = "CPX";
    }

    // save the primary partition type
    if (profile.profile_type != AMDSMI_ACCELERATOR_PARTITION_INVALID) {
      primary_partition_type = profile.profile_type;
      primary_index = dv_ind;
    }

    std::string partition_id_str = "";
    for (int i = 0; i < 8; i++) {
      partition_id_str += std::to_string(partition_id[i]);
      if (i < 7) {
        partition_id_str += ", ";
      }

      switch (primary_partition_type) {
        case AMDSMI_ACCELERATOR_PARTITION_SPX:
          EXPECT_LT(partition_id[i], MAX_SPX_PARTITIONS);
          break;
        case AMDSMI_ACCELERATOR_PARTITION_DPX:
          EXPECT_LT(partition_id[i], MAX_DPX_PARTITIONS);
          break;
        case AMDSMI_ACCELERATOR_PARTITION_TPX:
          EXPECT_LT(partition_id[i], MAX_TPX_PARTITIONS);
          break;
        case AMDSMI_ACCELERATOR_PARTITION_QPX:
          EXPECT_LT(partition_id[i], MAX_QPX_PARTITIONS);
          break;
        case AMDSMI_ACCELERATOR_PARTITION_CPX: {
          uint16_t num_xcd;
          uint32_t max_xcps = 0;
          DISPLAY_AMDSMI_API("amdsmi_get_gpu_xcd_counter", "gpu=" + std::to_string(primary_index),
                             VERB(STANDARD));
          ret = api_.get_xcd_counter(processor_handles_[primary_index], &num_xcd);
          DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
          if (ret == AMDSMI_STATUS_SUCCESS) {
            max_xcps = static_cast<uint32_t>(num_xcd);
          }
          if (!amd::smi::is_vm_guest()) {
            // In BM, we can get the number of XCDs (calculated by getting # of gfx_clocks)
            EXPECT_LT(partition_id[i], max_xcps);
          } else {
            // In guest, we may not be able to get the number of XCDs
            // (calculated by getting # of gfx_clocks)
            EXPECT_LE(partition_id[i], max_xcps);
          }
          break;
        }
        case AMDSMI_ACCELERATOR_PARTITION_INVALID:
          EXPECT_EQ(partition_id[i], MAX_UNSUPPORTED_PARTITIONS);
          break;
        default:
          EXPECT_EQ(partition_id[i], MAX_UNSUPPORTED_PARTITIONS);
          break;
      }
    }

    IF_VERB(STANDARD) {
      std::cout << "\t**amdsmi_get_gpu_accelerator_partition_profile(processor_handles_[" << dv_ind
                << "], &profile, &partition_id[0]):\n"
                << "\t\t" << smi_amdgpu_get_status_string(ret, false)
                << "\n\t**Current profile.profile_type: " << profile_type_str
                << "\n\t**profile.num_partitions: "
                << (profile.num_partitions == kMAX_UINT32 ? "N/A"
                                                          : std::to_string(profile.num_partitions))
                << "\n\t**profile.memory_caps: " << nps_caps_str << "\n\t**profile.profile_index: "
                << (profile.profile_index == kMAX_UINT32 ? "N/A"
                                                         : std::to_string(profile.profile_index))
                << "\n\t**profile.num_resources: " << profile.num_resources
                << "\n\t**partition_id: " << partition_id_str << std::endl;
    }
    EXPECT_TRUE(ret == AMDSMI_STATUS_SUCCESS || ret == AMDSMI_STATUS_NOT_SUPPORTED);
    amdsmi_accelerator_partition_profile_config_t profile_config = {};
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_profile_config",
                       "gpu=" + std::to_string(dv_ind), VERB(STANDARD));
    ret = api_.get_profile_config(processor_handles_[dv_ind], &profile_config);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    IF_VERB(STANDARD) {
      std::cout << "\t**amdsmi_get_gpu_accelerator_partition_profile_config(processor_handles_["
                << dv_ind << "], &profile_config):\n"
                << "\t\t" << smi_amdgpu_get_status_string(ret, false)
                << "\n\t**profile_config.num_profiles: " << profile_config.num_profiles
                << "\n\t**profile_config.num_resource_profiles: "
                << profile_config.num_resource_profiles << std::endl;
    }
    getAvailableProfileConfigs(dv_ind, profile, profile_config, isVerbose);

    IF_VERB(STANDARD) {
      std::cout << "\t**=========================================================\n";
      std::cout << "\t**Checking valid profile Sets =============================\n";
      std::cout << "\t**=========================================================\n";
    }
    int resource_index = 0;
    for (uint32_t i = 0; i < profile_config.num_profiles; i++) {
      auto current_profile = profile_config.profiles[i];
      std::string profile_type_str = "N/A";
      if (current_profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_SPX) {
        profile_type_str = "SPX";
      } else if (current_profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_DPX) {
        profile_type_str = "DPX";
      } else if (current_profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_TPX) {
        profile_type_str = "TPX";
      } else if (current_profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_QPX) {
        profile_type_str = "QPX";
      } else if (current_profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_CPX) {
        profile_type_str = "CPX";
      }

      std::string nps_caps_str = "";
      if ((current_profile.memory_caps.nps_flags.nps1_cap == 0 &&
           current_profile.memory_caps.nps_flags.nps2_cap == 0 &&
           current_profile.memory_caps.nps_flags.nps4_cap == 0 &&
           current_profile.memory_caps.nps_flags.nps8_cap == 0)) {
        nps_caps_str = "N/A";
      } else {
        nps_caps_str.clear();
        if (current_profile.memory_caps.nps_flags.nps1_cap) {
          (nps_caps_str.empty()) ? nps_caps_str += "NPS1" : nps_caps_str += ", NPS1";
        }
        if (current_profile.memory_caps.nps_flags.nps2_cap) {
          (nps_caps_str.empty()) ? nps_caps_str += "NPS2" : nps_caps_str += ", NPS2";
        }
        if (current_profile.memory_caps.nps_flags.nps4_cap) {
          (nps_caps_str.empty()) ? nps_caps_str += "NPS4" : nps_caps_str += ", NPS4";
        }
        if (current_profile.memory_caps.nps_flags.nps8_cap) {
          (nps_caps_str.empty()) ? nps_caps_str += "NPS8" : nps_caps_str += ", NPS8";
        }
      }
      IF_VERB(STANDARD) {
        std::cout << "\t**profile_config.profiles[" << i << "]:\n"
                  << "\t\tprofile_type: " << profile_type_str
                  << "\n\t\tnum_partitions: " << current_profile.num_partitions
                  << "\n\t\tmemory_caps: " << nps_caps_str
                  << "\n\t\tcurrent_profile.num_resources: " << current_profile.num_resources
                  << std::endl;
      }
      for (uint32_t j = 0; j < current_profile.num_resources; j++) {
        auto rp = profile_config.resource_profiles[resource_index];

        IF_VERB(STANDARD) {
          std::cout << "\n\t\t\tprofile_index: " << current_profile.profile_index
                    << "\n\t\t\tresource_index: " << resource_index
                    << "\n\t\t\tprofile_config.resource_profiles[" << resource_index
                    << "].resource_type: " << getResourceType(rp.resource_type)
                    << "\n\t\t\tprofile_config.resource_profiles[" << resource_index
                    << "].partition_resource: " << rp.partition_resource
                    << "\n\t\t\tprofile_config.resource_profiles[" << resource_index
                    << "].num_partitions_share_resource: " << rp.num_partitions_share_resource
                    << std::endl;
        }
        resource_index++;
      }
    }
    EXPECT_TRUE(ret == AMDSMI_STATUS_SUCCESS || ret == AMDSMI_STATUS_NOT_SUPPORTED);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "amdsmi_get_gpu_accelerator_partition_profile_config(): "
                  << "Not supported on this machine" << std::endl;
      }
      continue;
    }
  }

  // Run memory partition tests
  IF_VERB(STANDARD) {
    std::cout << "\t**=========================================================\n";
    std::cout << "\t**Test: Memory Partition Sets =============================\n";
    std::cout << "\t**=========================================================\n";
  }
  for (uint32_t dv_ind = 0; dv_ind < initial_num_devices; ++dv_ind) {
    if (!safe_to_test[dv_ind]) continue;
    if (dv_ind != 0) {
      IF_VERB(STANDARD) { std::cout << std::endl; }
    }
    api_.print_device_header(processor_handles_[dv_ind]);

    // Standard checks to see if API is supported, before running full tests
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    std::memset(orig_memory_partition, 0, sizeof(orig_memory_partition));
    ret = api_.get_memory(processor_handles_[dv_ind], orig_memory_partition, k255Len);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      continue;
    }
    EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS) << "Memory partition snapshot failed";
    if (ret != AMDSMI_STATUS_SUCCESS) continue;
    const std::map<std::string, amdsmi_memory_partition_type_t> memory_modes = {
        {"NPS1", AMDSMI_MEMORY_PARTITION_NPS1},
        {"NPS2", AMDSMI_MEMORY_PARTITION_NPS2},
        {"NPS4", AMDSMI_MEMORY_PARTITION_NPS4},
        {"NPS8", AMDSMI_MEMORY_PARTITION_NPS8}};
    const std::string original_memory(
        orig_memory_partition, strnlen(orig_memory_partition, sizeof(orig_memory_partition)));
    const auto original_mode = memory_modes.find(original_memory);
    EXPECT_NE(original_mode, memory_modes.end()) << "Invalid memory partition snapshot";
    if (original_mode == memory_modes.end()) {
      continue;
    }
    original_memory_modes[processor_handles_[dv_ind]] = original_memory;
    IF_VERB(STANDARD) {
      std::cout << "\t**Current Memory Partition: " << original_memory << std::endl;
    }

    // Verify api support checking functionality is working
    constexpr uint32_t k2Len = 2;
    char smallBuffer[k2Len];
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    err = api_.get_memory(processor_handles_[dv_ind], smallBuffer, k2Len);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INSUFFICIENT_SIZE);
    uint32_t size = static_cast<uint32_t>(sizeof(smallBuffer) / sizeof(*smallBuffer));
    ASSERT_EQ(err, AMDSMI_STATUS_INSUFFICIENT_SIZE);
    ASSERT_EQ(k2Len, size);
    if (err == AMDSMI_STATUS_INSUFFICIENT_SIZE) {
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Confirmed AMDSMI_STATUS_INSUFFICIENT_SIZE was returned "
                  << "and size is 2, as requested." << std::endl;
      }
    }

    // Verify api support checking functionality is working
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    err = api_.get_memory(processor_handles_[dv_ind], nullptr, k255Len);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

    if (err == AMDSMI_STATUS_INVAL) {
      IF_VERB(STANDARD) {
        std::cout << "\t**amdsmi_get_gpu_memory_partition(processor_handles_[" << dv_ind << "], "
                  << "nullptr, 255): "
                  << "Confirmed AMDSMI_STATUS_INVAL was returned." << std::endl;
      }
    }

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition_config", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    err = api_.get_memory_config(processor_handles_[dv_ind], nullptr);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

    if (err == AMDSMI_STATUS_INVAL) {
      IF_VERB(STANDARD) {
        std::cout << "\t**amdsmi_get_gpu_memory_partition(processor_handles_[" << dv_ind
                  << "], nullptr): Confirmed AMDSMI_STATUS_INVAL was returned." << std::endl;
      }
    }

    // Verify api support checking functionality is working
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    err = api_.get_memory(processor_handles_[dv_ind], orig_memory_partition, k0Len);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    ASSERT_TRUE(err == AMDSMI_STATUS_INVAL);
    if (err == AMDSMI_STATUS_INVAL) {
      IF_VERB(STANDARD) {
        std::cout << "\t**amdsmi_get_gpu_memory_partition(processor_handles_[" << dv_ind << "], "
                  << "orig_memory_partition, 0): "
                  << "Confirmed AMDSMI_STATUS_INVAL was returned." << std::endl;
      }
    }

    amdsmi_memory_partition_config_t* null_memory_partition_config = nullptr;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition_config", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    err = api_.get_memory_config(processor_handles_[dv_ind], null_memory_partition_config);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                          AMDSMI_STATUS_NOT_SUPPORTED);
    ASSERT_TRUE((err == AMDSMI_STATUS_INVAL) || (err == AMDSMI_STATUS_NOT_SUPPORTED));
    if (err == AMDSMI_STATUS_INVAL) {
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "amdsmi_get_gpu_memory_partition_config(processor_handles_[" << dv_ind << "], "
                  << "nullptr): "
                  << "Confirmed AMDSMI_STATUS_INVAL was returned." << std::endl;
      }
    }

    /****************************************/
    /* amdsmi_set_gpu_memory_partition_mode(...) */
    /****************************************/
    // Verify api support checking functionality is working
    amdsmi_memory_partition_type_t null_memory_partition = {};
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_memory_partition_mode", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    err = api_.set_memory(processor_handles_[dv_ind], null_memory_partition);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    std::cout << "\t**amdsmi_set_gpu_memory_partition_mode"
              << "(processor_handles_[" << dv_ind
              << "], null_memory_partition): " << smi_amdgpu_get_status_string(err, false) << "\n";
    // Note: new_memory_partition is not set
    ASSERT_TRUE(err == AMDSMI_STATUS_INVAL);
    if (err == AMDSMI_STATUS_INVAL) {
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Confirmed AMDSMI_STATUS_INVAL was returned." << std::endl;
      }
    } else if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << ": "
                  << "amdsmi_set_gpu_memory_partition_mode not supported on this "
                  << "device\n\t    (if amdsmi_get_gpu_memory_partition works, "
                  << "then likely need to set in bios)" << std::endl;
      }
      continue;
    }
    ASSERT_FALSE(err == AMDSMI_STATUS_NO_PERM);

    // Verify api support checking functionality is working
    amdsmi_memory_partition_type_t new_memory_partition = AMDSMI_MEMORY_PARTITION_UNKNOWN;
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_memory_partition_mode", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    err = api_.set_memory(processor_handles_[dv_ind], new_memory_partition);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    ASSERT_TRUE((err == AMDSMI_STATUS_INVAL) || (err == AMDSMI_STATUS_NOT_SUPPORTED) ||
                (err == AMDSMI_STATUS_NO_PERM));
    if (err == AMDSMI_STATUS_INVAL) {
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Confirmed AMDSMI_STATUS_INVAL was returned." << std::endl;
      }
      else if (err == AMDSMI_STATUS_NO_PERM) {
        // tests should not continue if err is a permission issue
        ASSERT_FALSE(err == AMDSMI_STATUS_NO_PERM);
      }
    }

    // Do not turn a same-mode check into a write after an observed external change.
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = api_.get_memory(processor_handles_[dv_ind], orig_memory_partition, k255Len);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS) << "Memory partition pre-set read failed";
    if (ret != AMDSMI_STATUS_SUCCESS) continue;
    const std::string before_set(orig_memory_partition,
                                 strnlen(orig_memory_partition, sizeof(orig_memory_partition)));
    EXPECT_EQ(before_set, original_memory) << "Memory partition changed before same-mode set";
    if (before_set != original_memory) continue;

    {
      ret_set = AMDSMI_STATUS_NOT_SUPPORTED;
      // Active-mode requests cannot cancel a deferred alternate-NPS write.
      new_memory_partition = original_mode->second;
      IF_VERB(STANDARD) {
        std::cout << std::endl;
        std::cout << "\t**"
                  << "======== TEST AMDSMI_MEMORY_PARTITION_"
                  << memoryPartitionString(new_memory_partition) << " ===============" << std::endl;
      }

      // Read capabilities before attempting set; used to validate set return code.
      current_memory_config = {};
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition_config", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      auto ret_caps = api_.get_memory_config(processor_handles_[dv_ind], &current_memory_config);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret_caps, AMDSMI_STATUS_SUCCESS);
      ASSERT_TRUE((ret_caps == AMDSMI_STATUS_NOT_SUPPORTED) || (ret_caps == AMDSMI_STATUS_SUCCESS));

      std::string memory_caps_str = "N/A";
      bool partition_is_supported = false;
      if (ret_caps == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(current_memory_config.mp_mode, new_memory_partition)
            << "Memory partition changed before same-mode set";
        if (current_memory_config.mp_mode != new_memory_partition) continue;
        memory_caps_str.clear();
        if (current_memory_config.partition_caps.nps_flags.nps1_cap) {
          memory_caps_str += (memory_caps_str.empty() ? "NPS1" : ", NPS1");
          if (new_memory_partition == AMDSMI_MEMORY_PARTITION_NPS1) partition_is_supported = true;
        }
        if (current_memory_config.partition_caps.nps_flags.nps2_cap) {
          memory_caps_str += (memory_caps_str.empty() ? "NPS2" : ", NPS2");
          if (new_memory_partition == AMDSMI_MEMORY_PARTITION_NPS2) partition_is_supported = true;
        }
        if (current_memory_config.partition_caps.nps_flags.nps4_cap) {
          memory_caps_str += (memory_caps_str.empty() ? "NPS4" : ", NPS4");
          if (new_memory_partition == AMDSMI_MEMORY_PARTITION_NPS4) partition_is_supported = true;
        }
        if (current_memory_config.partition_caps.nps_flags.nps8_cap) {
          memory_caps_str += (memory_caps_str.empty() ? "NPS8" : ", NPS8");
          if (new_memory_partition == AMDSMI_MEMORY_PARTITION_NPS8) partition_is_supported = true;
        }
      }

      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Available Memory Partition Capabilities: " << memory_caps_str << "\n"
                  << "\t**"
                  << "current_memory_partition_mode: "
                  << (ret_caps == AMDSMI_STATUS_SUCCESS
                          ? memoryPartitionString(current_memory_config.mp_mode)
                          : "N/A")
                  << "\n"
                  << "\t**"
                  << "Requested partition supported by hardware: "
                  << (partition_is_supported ? "YES" : "NO") << "\n"
                  << "\t**"
                  << "Attempting to set memory partition to: "
                  << memoryPartitionString(new_memory_partition) << std::endl;
      }

      DISPLAY_AMDSMI_API("amdsmi_set_gpu_memory_partition_mode", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      ret_set = api_.set_memory(processor_handles_[dv_ind], new_memory_partition);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret_set, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_INVAL, AMDSMI_STATUS_NOT_SUPPORTED);
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "amdsmi_set_gpu_memory_partition_mode(processor_handles_[" << dv_ind << "], "
                  << memoryPartitionString(new_memory_partition)
                  << "): " << smi_amdgpu_get_status_string(ret_set, false) << "\n";
      }
      if (ret_set == AMDSMI_STATUS_NOT_SUPPORTED) {
        continue;
      }
      ASSERT_TRUE((ret_set == AMDSMI_STATUS_SUCCESS) || (ret_set == AMDSMI_STATUS_BUSY) ||
                  (ret_set == AMDSMI_STATUS_AMDGPU_RESTART_ERR) ||
                  (ret_set == AMDSMI_STATUS_INVAL) || (ret_set == AMDSMI_STATUS_NOT_SUPPORTED));

      // If the hardware advertises support for this mode, the set should succeed unless the
      // device is busy (e.g., workloads running).
      if (partition_is_supported) {
        EXPECT_TRUE(ret_set == AMDSMI_STATUS_SUCCESS || ret_set == AMDSMI_STATUS_BUSY)
            << "Set did not succeed for a partition mode advertised as supported by hardware: "
            << memoryPartitionString(new_memory_partition);
      }

      if (ret_set == AMDSMI_STATUS_SUCCESS) {
        std::cout << "\t** Current-mode memory partition request accepted; "
                     "alternate NPS transitions were not tested.\n";
      }

      // Verify capabilities are unchanged after the set attempt.
      amdsmi_memory_partition_config_t post_set_config = {};
      auto ret_caps_post = api_.get_memory_config(processor_handles_[dv_ind], &post_set_config);
      if (ret_caps_post == AMDSMI_STATUS_SUCCESS && ret_caps == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(post_set_config.partition_caps.nps_flags.nps1_cap,
                  current_memory_config.partition_caps.nps_flags.nps1_cap);
        EXPECT_EQ(post_set_config.partition_caps.nps_flags.nps2_cap,
                  current_memory_config.partition_caps.nps_flags.nps2_cap);
        EXPECT_EQ(post_set_config.partition_caps.nps_flags.nps4_cap,
                  current_memory_config.partition_caps.nps_flags.nps4_cap);
        EXPECT_EQ(post_set_config.partition_caps.nps_flags.nps8_cap,
                  current_memory_config.partition_caps.nps_flags.nps8_cap);
      }
    }
  }  // END DEVICE FOR LOOP

  for (const auto& entry : original_memory_modes) {
    SCOPED_TRACE(::testing::Message() << "Memory partition handle " << entry.first);
    char current[k255Len] = {};
    const auto status = api_.get_memory(entry.first, current, sizeof(current));
    EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS) << "Memory partition readback failed";
    if (status != AMDSMI_STATUS_SUCCESS) continue;
    EXPECT_EQ(std::string(current, strnlen(current, sizeof(current))), entry.second)
        << "Memory partition drift";
  }

  // Another actor's changes must fail verification, not be overwritten.
  for (const auto& entry : original_profiles) {
    SCOPED_TRACE(::testing::Message() << "Compute partition handle " << entry.first);
    amdsmi_accelerator_partition_profile_t profile = {};
    uint32_t partition_id[8] = {};
    const auto status = api_.get_profile(entry.first, &profile, partition_id);
    EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS) << "Compute partition readback failed";
    if (status != AMDSMI_STATUS_SUCCESS) continue;
    EXPECT_EQ(profile.profile_type, entry.second.profile_type) << "Compute partition drift";
    EXPECT_EQ(profile.profile_index, entry.second.profile_index) << "Compute profile index drift";
  }
}
