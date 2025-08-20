/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <cstdint>

#include <cstdint>
#include <iostream>
#include <string>
#include <map>
#include <limits>

#include "gtest/gtest.h"
#include "../test_base.h"
#include "../test_common.h"
#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "memorypartition_read_write.h"

const uint32_t MAX_UNSUPPORTED_PARTITIONS = 0;
const uint32_t MAX_SPX_PARTITIONS = 1;  // Single GPU node
const uint32_t MAX_DPX_PARTITIONS = 2;
const uint32_t MAX_TPX_PARTITIONS = 3;
const uint32_t MAX_QPX_PARTITIONS = 4;

void ReloadDriverWithMessages(bool isVerbose,
                              const std::string& preReloadMessage,
                              const std::string& successMessage,
                              const std::string& errorMessage,
                              const std::string& restartErrorMessage,
                              amdsmi_status_t *reload_status) {
    if (isVerbose) {
      std::cout << "\t**" << preReloadMessage << std::endl;
    }

    auto start_time = std::chrono::steady_clock::now();
    auto driver_reload_status = amdsmi_gpu_driver_reload();
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                                            end_time - start_time);
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    *reload_status = driver_reload_status;

    if (isVerbose) {
      std::cout << "\t**"
                << "amdsmi_gpu_driver_reload() took "
                << elapsed_time.count() << " milliseconds ("
                << elapsed_seconds.count() << " seconds)" << std::endl;
    }

    if (driver_reload_status == AMDSMI_STATUS_SUCCESS) {
      if (isVerbose) {
        std::cout << "\t**" << successMessage << std::endl;
      }
      ASSERT_EQ(driver_reload_status, AMDSMI_STATUS_SUCCESS);
    } else if (driver_reload_status == AMDSMI_STATUS_AMDGPU_RESTART_ERR) {
      if (isVerbose) {
        std::cout << "\t**" << restartErrorMessage << std::endl;
      }
      ASSERT_TRUE(driver_reload_status == AMDSMI_STATUS_AMDGPU_RESTART_ERR);
    } else {
      if (isVerbose) {
          std::cout << "\t**" << errorMessage << ": "
                    << smi_amdgpu_get_status_string(driver_reload_status, false) << std::endl;
      }
    }
    // Tests should fail if the driver reload fails
    // TODO(amdsmi_team): This is a temporary solution until CQE can update
    //                    how their containers are ran.
    //                    This is because the driver reload requires:
    //                    1) Containers must run serially
    //                       (i.e. no parallel containers running at the same time)
    //                    2) Containers must run with extra parameters:
    //                       --cap-add=SYS_ADMIN -v /lib/modules:/lib/modules
    //                       See: https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/setup-docker-container.html
    #if 0
      ASSERT_EQ(driver_reload_status, AMDSMI_STATUS_SUCCESS);
    #endif
}

TestMemoryPartitionReadWrite::TestMemoryPartitionReadWrite() : TestBase() {
  set_title("AMDSMI Memory Partition Read Test");
  set_description("The memory partition tests verifies that the memory "
                  "partition settings can be read and updated properly.");
}

TestMemoryPartitionReadWrite::~TestMemoryPartitionReadWrite(void) {
}

void TestMemoryPartitionReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestMemoryPartitionReadWrite::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestMemoryPartitionReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestMemoryPartitionReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

static const std::string
memoryPartitionString(amdsmi_memory_partition_type_t memoryPartitionType) {
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

static const std::map<std::string, amdsmi_memory_partition_type_t>
mapStringToRSMIMemoryPartitionTypes {
  {"NPS1", AMDSMI_MEMORY_PARTITION_NPS1},
  {"NPS2", AMDSMI_MEMORY_PARTITION_NPS2},
  {"NPS4", AMDSMI_MEMORY_PARTITION_NPS4},
  {"NPS8", AMDSMI_MEMORY_PARTITION_NPS8}
};

void TestMemoryPartitionReadWrite::Run(void) {
  amdsmi_status_t ret, err, ret_set;
  constexpr uint32_t k255Len = 255;
  constexpr uint32_t k0Len = 0;
  char orig_memory_partition[k255Len];
  char current_memory_partition[k255Len];
  orig_memory_partition[0] = '\0';
  current_memory_partition[0] = '\0';
  amdsmi_memory_partition_config_t current_memory_config;
  const uint32_t kMAX_UINT32 = std::numeric_limits<uint32_t>::max();
  std::map<uint32_t, AcceleratorProfileConfig> orig_dev_config;  // index, ProfileConfig

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  bool isVerbose = (this->verbosity() &&
        this->verbosity() >= (this->TestBase::VERBOSE_STANDARD)) ? true: false;

  // Save original memory partition settings (see orig_dev_config ^)
  IF_VERB(STANDARD) {
    std::cout << "\t**=========================================================\n";
    std::cout << "\t**Save Original Compute Partition Settings ================\n";
    std::cout << "\t**=========================================================\n";
  }
  auto initial_num_devices = num_monitor_devs();
  amdsmi_accelerator_partition_type_t primary_partition_type = AMDSMI_ACCELERATOR_PARTITION_INVALID;
  uint32_t primary_index = 0;
  for (uint32_t dv_ind = 0; dv_ind < initial_num_devices; ++dv_ind) {
    if (dv_ind != 0) {
      std::cout << "\n";
    }
    PrintDeviceHeader(processor_handles_[dv_ind]);
    amdsmi_accelerator_partition_profile_t profile = {};
    uint32_t partition_id[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    ret = amdsmi_get_gpu_accelerator_partition_profile(processor_handles_[dv_ind],
                                                        &profile, &partition_id[0]);
    std::string nps_caps_str = "";
    if ((profile.memory_caps.nps_flags.nps1_cap == 0
        && profile.memory_caps.nps_flags.nps2_cap == 0
        && profile.memory_caps.nps_flags.nps4_cap == 0
        && profile.memory_caps.nps_flags.nps8_cap == 0)) {
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
          ret = amdsmi_get_gpu_xcd_counter(processor_handles_[primary_index], &num_xcd);
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
      std::cout << "\t**amdsmi_get_gpu_accelerator_partition_profile(processor_handles_["
                << dv_ind << "], &profile, &partition_id[0]):\n"
                << "\t\t" << smi_amdgpu_get_status_string(ret, false)
                << "\n\t**Current profile.profile_type: "
                << profile_type_str
                << "\n\t**profile.num_partitions: "
                << (profile.num_partitions == kMAX_UINT32
                ? "N/A" : std::to_string(profile.num_partitions))
                << "\n\t**profile.memory_caps: "
                << nps_caps_str
                << "\n\t**profile.profile_index: "
                << (profile.profile_index == kMAX_UINT32
                ? "N/A" : std::to_string(profile.profile_index))
                << "\n\t**profile.num_resources: "
                << profile.num_resources
                << "\n\t**partition_id: "
                << partition_id_str
                << std::endl;
    }
    EXPECT_TRUE(ret == AMDSMI_STATUS_SUCCESS
                || ret == AMDSMI_STATUS_NOT_SUPPORTED);
    amdsmi_accelerator_partition_profile_config_t profile_config = {};
    ret = amdsmi_get_gpu_accelerator_partition_profile_config(processor_handles_[dv_ind],
                                                              &profile_config);
    IF_VERB(STANDARD) {
      std::cout << "\t**amdsmi_get_gpu_accelerator_partition_profile_config(processor_handles_["
                << dv_ind << "], &profile_config):\n"
                << "\t\t" << smi_amdgpu_get_status_string(ret, false)
                << "\n\t**profile_config.num_profiles: "
                << profile_config.num_profiles
                << "\n\t**profile_config.num_resource_profiles: "
                << profile_config.num_resource_profiles
                << std::endl;
    }
    AcceleratorProfileConfig original_profile_config =
      getAvailableProfileConfigs(dv_ind, profile, profile_config, isVerbose);
    orig_dev_config[dv_ind] = original_profile_config;

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
      if ((current_profile.memory_caps.nps_flags.nps1_cap == 0
          && current_profile.memory_caps.nps_flags.nps2_cap == 0
          && current_profile.memory_caps.nps_flags.nps4_cap == 0
          && current_profile.memory_caps.nps_flags.nps8_cap == 0)) {
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
                    << "].resource_type: "
                    << getResourceType(rp.resource_type)
                    << "\n\t\t\tprofile_config.resource_profiles[" << resource_index
                    << "].partition_resource: "
                    << rp.partition_resource
                    << "\n\t\t\tprofile_config.resource_profiles[" << resource_index
                    << "].num_partitions_share_resource: "
                    << rp.num_partitions_share_resource
                    << std::endl;
        }
        resource_index++;
      }
    }
    EXPECT_TRUE(ret == AMDSMI_STATUS_SUCCESS
                || ret == AMDSMI_STATUS_NOT_SUPPORTED);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**" <<  "amdsmi_get_gpu_accelerator_partition_profile_config(): "
                  << "Not supported on this machine" << std::endl;
      }
      continue;
    }
  }

  // Basic check we can reload the driver, regardless of if changing memory partition
  // is supported or not
  // FYI Need to place after saving current compute partitions, since reloading driver will reset
  // all back to SPX/DPX/etc (whatever is default for that NPS mode; see
  // `sudo amd-smi partition -a`).
  IF_VERB(STANDARD) {
    std::cout << "\t**"
              << "======== TEST AMDSMI_GPU_DRIVER_RELOAD() BEFORE"
              << " MEMORY PARTITION CHECKS ===============" << std::endl;
  }
  amdsmi_status_t driver_reload_status = AMDSMI_STATUS_NOT_SUPPORTED;
  std::string preload_message =
    "\t  Reloading the AMD GPU driver before memory partition checks."
    " This may take some time, please wait...";
  ReloadDriverWithMessages(isVerbose, preload_message,
    "amdsmi_gpu_driver_reload() successful.",
    "amdsmi_gpu_driver_reload() failed",
    "amdsmi_gpu_driver_reload() failed with AMDGPU_RESTART_ERR",
    &driver_reload_status);

  // Run memory partition tests
  IF_VERB(STANDARD) {
    std::cout << "\t**=========================================================\n";
    std::cout << "\t**Test: Memory Partition Sets =============================\n";
    std::cout << "\t**=========================================================\n";
  }
  uint32_t current_num_devices = 0;
  smi_amdgpu_get_device_count(&current_num_devices);

  IF_VERB(STANDARD) {
    std::cout << "\t**Total Num Devices: " << current_num_devices << std::endl;
  }
  // Leaving for debug purposes - uncomment to test a specific number of devices
  // uint32_t num_devices_to_test = 1;
  uint32_t num_devices_to_test = current_num_devices;
  for (uint32_t dv_ind = 0; dv_ind < num_devices_to_test; ++dv_ind) {
    bool wasSetSuccess = false;
    if (dv_ind != 0) {
      IF_VERB(STANDARD) {
        std::cout << std::endl;
      }
    }
    PrintDeviceHeader(processor_handles_[dv_ind]);

    // Standard checks to see if API is supported, before running full tests
    ret = amdsmi_get_gpu_memory_partition(
            processor_handles_[dv_ind], orig_memory_partition, k255Len);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
       IF_VERB(STANDARD) {
          std::cout << "\t**" <<  "amdsmi_get_gpu_memory_partition(): "
                    << "Not supported on this machine" << std::endl;
        }
        continue;
    } else {
        CHK_ERR_ASRT(ret)
    }
    IF_VERB(STANDARD) {
      std::cout << std::endl << "\t**Current Memory Partition: "
                << orig_memory_partition << std::endl;
    }

    if (orig_memory_partition[0] == '\0') {
      std::cout << "***System memory partition value is not defined or received"
                  " unexpected data. Skip memory partition test." << std::endl;
      continue;
    }
    ASSERT_TRUE(ret == AMDSMI_STATUS_SUCCESS);

    // Verify api support checking functionality is working
    constexpr uint32_t k2Len = 2;
    char smallBuffer[k2Len];
    err = amdsmi_get_gpu_memory_partition(processor_handles_[dv_ind], smallBuffer, k2Len);
    uint32_t size = static_cast<uint32_t>(sizeof(smallBuffer)/sizeof(*smallBuffer));
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
    err = amdsmi_get_gpu_memory_partition(processor_handles_[dv_ind], nullptr, k255Len);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

    if (err == AMDSMI_STATUS_INVAL) {
      IF_VERB(STANDARD) {
        std::cout << "\t**amdsmi_get_gpu_memory_partition(processor_handles_[" << dv_ind << "], "
                  << "nullptr, 255): "
                  << "Confirmed AMDSMI_STATUS_INVAL was returned."
                  << std::endl;
      }
    }

    err = amdsmi_get_gpu_memory_partition_config(processor_handles_[dv_ind], nullptr);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

    if (err == AMDSMI_STATUS_INVAL) {
      IF_VERB(STANDARD) {
        std::cout << "\t**amdsmi_get_gpu_memory_partition(processor_handles_[" << dv_ind
                  << "], nullptr): Confirmed AMDSMI_STATUS_INVAL was returned."
                  << std::endl;
      }
    }

    // Verify api support checking functionality is working
    err = amdsmi_get_gpu_memory_partition(processor_handles_[dv_ind], orig_memory_partition, k0Len);
    ASSERT_TRUE(err == AMDSMI_STATUS_INVAL);
    if (err == AMDSMI_STATUS_INVAL) {
      IF_VERB(STANDARD) {
        std::cout << "\t**amdsmi_get_gpu_memory_partition(processor_handles_[" << dv_ind << "], "
                  << "orig_memory_partition, 0): "
                  << "Confirmed AMDSMI_STATUS_INVAL was returned."
                  << std::endl;
      }
    }

    amdsmi_memory_partition_config_t* null_memory_partition_config = nullptr;
    err = amdsmi_get_gpu_memory_partition_config(processor_handles_[dv_ind],
                                                  null_memory_partition_config);
    ASSERT_TRUE((err == AMDSMI_STATUS_INVAL) ||
                (err == AMDSMI_STATUS_NOT_SUPPORTED));
    if (err == AMDSMI_STATUS_INVAL) {
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "amdsmi_get_gpu_memory_partition_config(processor_handles_[" << dv_ind << "], "
                  << "nullptr): "
                  << "Confirmed AMDSMI_STATUS_INVAL was returned."
                  << std::endl;
      }
    }

    /****************************************/
    /* amdsmi_set_gpu_memory_partition(...) */
    /****************************************/
    // Verify api support checking functionality is working
    amdsmi_memory_partition_type_t null_memory_partition = {};
    err = amdsmi_set_gpu_memory_partition_mode(processor_handles_[dv_ind], null_memory_partition);
    std::cout << "\t**amdsmi_set_gpu_memory_partition(amdsmi_set_gpu_memory_partition_mode"
              << "(processor_handles_[" << dv_ind << "], nullptr): "
              << smi_amdgpu_get_status_string(err, false) << "\n";
    // Note: new_memory_partition is not set
    ASSERT_TRUE(err == AMDSMI_STATUS_INVAL);
    if (err == AMDSMI_STATUS_INVAL) {
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Confirmed AMDSMI_STATUS_INVAL was returned."
                  << std::endl;
      }
    } else if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
         IF_VERB(STANDARD) {
          std::cout << "\t**" <<  ": "
                    << "amdsmi_set_gpu_memory_partition_mode not supported on this "
                    << "device\n\t    (if amdsmi_get_gpu_memory_partition works, "
                    << "then likely need to set in bios)"
                    << std::endl;
        }
        continue;
    } else {
        DISPLAY_AMDSMI_ERR(err)
    }
    ASSERT_FALSE(err == AMDSMI_STATUS_NO_PERM);

    // Verify api support checking functionality is working
    amdsmi_memory_partition_type_t new_memory_partition = AMDSMI_MEMORY_PARTITION_UNKNOWN;
    err = amdsmi_set_gpu_memory_partition_mode(processor_handles_[dv_ind], new_memory_partition);
    ASSERT_TRUE((err == AMDSMI_STATUS_INVAL) ||
                (err == AMDSMI_STATUS_NOT_SUPPORTED) ||
                (err == AMDSMI_STATUS_NO_PERM));
    if (err == AMDSMI_STATUS_INVAL) {
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Confirmed AMDSMI_STATUS_INVAL was returned."
                  << std::endl;
      } else if (err == AMDSMI_STATUS_NO_PERM) {
        DISPLAY_AMDSMI_ERR(err)
        // tests should not continue if err is a permission issue
        ASSERT_FALSE(err == AMDSMI_STATUS_NO_PERM);
      } else {
        DISPLAY_AMDSMI_ERR(err)
      }
    }

    // Re-run original get, so we can reset to later
    ret = amdsmi_get_gpu_memory_partition(processor_handles_[dv_ind],
                                          orig_memory_partition, k255Len);
    ASSERT_EQ(AMDSMI_STATUS_SUCCESS, ret);

    for (int partition = static_cast<int>(AMDSMI_MEMORY_PARTITION_NPS1);
         partition <= static_cast<int>(AMDSMI_MEMORY_PARTITION_NPS8);
         partition++) {
      ret_set = AMDSMI_STATUS_NOT_SUPPORTED;
      wasSetSuccess = false;
      new_memory_partition = static_cast<amdsmi_memory_partition_type_t>(partition);
      if (new_memory_partition != AMDSMI_MEMORY_PARTITION_NPS1
          && new_memory_partition != AMDSMI_MEMORY_PARTITION_NPS2
          && new_memory_partition != AMDSMI_MEMORY_PARTITION_NPS4
          && new_memory_partition != AMDSMI_MEMORY_PARTITION_NPS8) {
        continue;  // skip unknown partition, this is already tested above ^
      }
      IF_VERB(STANDARD) {
        std::cout << std::endl;
        std::cout << "\t**"
                  << "======== TEST AMDSMI_MEMORY_PARTITION_"
                  << memoryPartitionString(new_memory_partition)
                  << " ===============" << std::endl;
      }
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Attempting to set memory partition to: "
                  << memoryPartitionString(new_memory_partition) << std::endl;
      }

      auto ret_caps = amdsmi_get_gpu_memory_partition_config(processor_handles_[dv_ind],
                                    &current_memory_config);
      std::string memory_caps_str = "N/A";
      if (ret_caps == AMDSMI_STATUS_SUCCESS) {
        memory_caps_str.clear();
        if (current_memory_config.partition_caps.nps_flags.nps1_cap) {
          memory_caps_str += (memory_caps_str.empty() ? "NPS1" : ", NPS1");
        }
        if (current_memory_config.partition_caps.nps_flags.nps2_cap) {
          memory_caps_str += (memory_caps_str.empty() ? "NPS2" : ", NPS2");
        }
        if (current_memory_config.partition_caps.nps_flags.nps4_cap) {
          memory_caps_str += (memory_caps_str.empty() ? "NPS4" : ", NPS4");
        }
        if (current_memory_config.partition_caps.nps_flags.nps8_cap) {
          memory_caps_str += (memory_caps_str.empty() ? "NPS8" : ", NPS8");
        }
      }

      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "amdsmi_get_gpu_memory_partition_config(processor_handles_[" << dv_ind
                  << "], current_memory_config): "
                  << smi_amdgpu_get_status_string(ret_caps, false) << std::endl;
        std::cout << "\t**" << "Available Memory Partition Capabilities: "
                  << memory_caps_str << "\n"
                  << "\t**" << "current_memory_partition_mode: "
                  << memoryPartitionString(current_memory_config.mp_mode) << "\n"
                  << "\t**" << "num_numa_ranges: "
                  << current_memory_config.num_numa_ranges
                  << std::endl;
      }
      ASSERT_TRUE((ret_caps == AMDSMI_STATUS_NOT_SUPPORTED) ||
                  (ret_caps == AMDSMI_STATUS_SUCCESS));

      ret_set = amdsmi_set_gpu_memory_partition_mode(processor_handles_[dv_ind],
                                                      new_memory_partition);
      IF_VERB(STANDARD) {
        std::cout << "\t**" <<  "amdsmi_set_gpu_memory_partition_mode(processor_handles_["
                  << dv_ind << "], " << memoryPartitionString(new_memory_partition) << "): "
                  << smi_amdgpu_get_status_string(ret_set, false) << "\n";
      }
      if (ret_set == AMDSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**" <<  "amdsmi_set_gpu_memory_partition_mode(): "
                    << "Not supported on this machine" << std::endl;
        }
        break;
      } else {
        ASSERT_TRUE((ret_set == AMDSMI_STATUS_SUCCESS)
                  || (ret_set == AMDSMI_STATUS_BUSY)
                  || (ret_set == AMDSMI_STATUS_AMDGPU_RESTART_ERR)
                  || (ret_set == AMDSMI_STATUS_INVAL)
                  || (ret_set == AMDSMI_STATUS_NOT_SUPPORTED));
      }

      amdsmi_status_t driver_reload_status = AMDSMI_STATUS_NOT_SUPPORTED;
      if (ret_set == AMDSMI_STATUS_SUCCESS) {  // do not continue trying to reset
        // Now we require a separate call to reload the driver, since this operation
        // has been removed from the amdsmi_set_gpu_memory_partition_mode and
        // amdsmi_set_gpu_memory_partition().
        // This is to allow the user to select the appropriate time to reload the driver
        // since there can be errors if any device has a workload/process running on it.
        std::string reload_message =
          "\t  Reloading the AMD GPU driver after setting memory partition to "
          + memoryPartitionString(new_memory_partition)
          + ". This may take some time, please wait...";
        std::string driver_reload_success_message =
          "amdsmi_gpu_driver_reload() successful after setting memory partition to "
          + memoryPartitionString(new_memory_partition);
        std::string failure_message =
          "amdsmi_gpu_driver_reload() failed after setting memory partition to "
          + memoryPartitionString(new_memory_partition);
        std::string restart_error_message =
          "amdsmi_gpu_driver_reload() failed with AMDGPU_RESTART_ERR after "
          "setting memory partition to " + memoryPartitionString(new_memory_partition);
        ReloadDriverWithMessages(isVerbose, reload_message,
          driver_reload_success_message,
          failure_message,
          restart_error_message,
          &driver_reload_status);
        if (driver_reload_status == AMDSMI_STATUS_SUCCESS) {
          wasSetSuccess = true;
        }
      }

      ret = amdsmi_get_gpu_memory_partition_config(processor_handles_[dv_ind],
                                                  &current_memory_config);
      if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**" <<  "amdsmi_get_gpu_memory_partition_config(): "
                    << "Not supported on this machine" << std::endl;
        }
        continue;
      }
      CHK_ERR_ASRT(ret)
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Current memory partition: "
                  << memoryPartitionString(current_memory_config.mp_mode)
                  << std::endl;
      }
      if (wasSetSuccess) {
        ASSERT_EQ(AMDSMI_STATUS_SUCCESS, ret_set);
        ASSERT_STREQ(memoryPartitionString(new_memory_partition).c_str(),
                     memoryPartitionString(current_memory_config.mp_mode).c_str());
        CHK_ERR_ASRT(ret_set)
      } else {
        ASSERT_TRUE(ret_set == AMDSMI_STATUS_SUCCESS
                    || ret_set == AMDSMI_STATUS_INVAL
                    || ret_set == AMDSMI_STATUS_NOT_SUPPORTED);
        ASSERT_STRNE(memoryPartitionString(new_memory_partition).c_str(),
                     memoryPartitionString(current_memory_config.mp_mode).c_str());
      }
    }  // END MEMORY PARTITION FOR LOOP

    /* TEST RETURN TO ORIGINAL MEMORY PARTITION SETTING */
    IF_VERB(STANDARD) {
      std::cout << std::endl;
      std::cout << "\t**"
                << "=========== TEST RETURN TO ORIGINAL MEMORY PARTITION "
                << "SETTING (" << orig_memory_partition
                << ") ========" << std::endl;
    }

    ret = amdsmi_get_gpu_memory_partition_config(processor_handles_[dv_ind],
                                                 &current_memory_config);
    ASSERT_TRUE((ret == AMDSMI_STATUS_NOT_SUPPORTED) ||
                  (ret == AMDSMI_STATUS_SUCCESS));
    IF_VERB(STANDARD) {
      std::cout << "\t**"
                << "amdsmi_get_gpu_memory_partition_config(processor_handles_[" << dv_ind
                << "], current_memory_config): "
                << smi_amdgpu_get_status_string(ret, false) << std::endl;
      std::cout << "\t**"
                << "Current memory partition: "
                << memoryPartitionString(current_memory_config.mp_mode)
                << std::endl;
    }
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**" <<  "amdsmi_get_gpu_memory_partition_config(): "
                  << "Not supported on this machine... trying on other devices" << std::endl;
      }
      continue;
    }

    new_memory_partition
      = mapStringToRSMIMemoryPartitionTypes.at(orig_memory_partition);
    IF_VERB(STANDARD) {
      std::cout << "\t**" << "Returning memory partition to: "
                << memoryPartitionString(new_memory_partition) << std::endl;
    }
    ret = amdsmi_set_gpu_memory_partition(processor_handles_[dv_ind], new_memory_partition);
    IF_VERB(STANDARD) {
      std::cout << "\t**"
                << "amdsmi_set_gpu_memory_partition(processor_handles_[" << dv_ind
                << "], " << orig_memory_partition << "): "
                << smi_amdgpu_get_status_string(ret, false) << std::endl;
    }
    CHK_ERR_ASRT(ret)
    if (ret == AMDSMI_STATUS_SUCCESS) {
        // Now we require a separate call to reload the driver, since this operation
        // has been removed from the amdsmi_set_gpu_memory_partition_mode and
        // amdsmi_set_gpu_memory_partition().
        // This is to allow the user to select the appropriate time to reload the driver
        // since there can be errors if any device has a workload/process running on it.
        driver_reload_status = AMDSMI_STATUS_NOT_SUPPORTED;
        std::string reload_message =
          "\t  Reloading the AMD GPU driver after resetting memory partition to "
          + std::string(orig_memory_partition)
          + ". This may take some time, please wait...";
        std::string driver_reload_success_message =
          "amdsmi_gpu_driver_reload() successful after resetting memory partition to "
          + std::string(orig_memory_partition);
        std::string failure_message =
          "amdsmi_gpu_driver_reload() failed after resetting memory partition to "
          + std::string(orig_memory_partition);
        std::string restart_error_message =
          "amdsmi_gpu_driver_reload() failed with AMDGPU_RESTART_ERR after "
          "resetting memory partition to " + std::string(orig_memory_partition);
        ReloadDriverWithMessages(isVerbose, reload_message,
          driver_reload_success_message,
          failure_message,
          restart_error_message,
          &driver_reload_status);
    }
    ret = amdsmi_get_gpu_memory_partition(processor_handles_[dv_ind],
                                          current_memory_partition, k255Len);
    CHK_ERR_ASRT(ret)
    IF_VERB(STANDARD) {
      std::cout << "\t**" << "Attempted to set memory partition: "
                << memoryPartitionString(new_memory_partition) << std::endl
                << "\t**" << "Current memory partition: "
                << current_memory_partition
                << std::endl;
    }
    ASSERT_EQ(AMDSMI_STATUS_SUCCESS, ret);
    ASSERT_STREQ(orig_memory_partition, current_memory_partition);
    IF_VERB(STANDARD) {
      std::cout << "\t**"
                << "Confirmed prior memory partition (" << orig_memory_partition
                << ") is  equal to current memory partition ("
                << current_memory_partition << ")" << std::endl;
    }
  }  // END DEVICE FOR LOOP

  // Restore original compute partition settings (see orig_dev_config ^)
  IF_VERB(STANDARD) {
    std::cout << "\t**=========================================================\n";
    std::cout << "\t**Restore Original Compute Partition Settings =============\n";
    std::cout << "\t**=========================================================\n";
  }
  initial_num_devices = num_monitor_devs();
  for (uint32_t dv_ind = 0; dv_ind < initial_num_devices; ++dv_ind) {
    if (dv_ind != 0) {
      std::cout << "\n";
    }
    PrintDeviceHeader(processor_handles_[dv_ind]);

    AcceleratorProfileConfig original_profile_config = orig_dev_config[dv_ind];

    // Return to original profile
    IF_VERB(STANDARD) {
      std::cout << "\t**Device Index: " << dv_ind << std::endl
                << "\t**======== Return to original AMDSMI_ACCELERATOR_PARTITION_"
                << original_profile_config.original_profile_type_str
                << " (profile_index: "
                << (original_profile_config.original_profile_index == kMAX_UINT32
                ? "N/A" : std::to_string(original_profile_config.original_profile_index))
                << ")"
                << " ===============" << std::endl;
    }
    auto ret_set = amdsmi_set_gpu_accelerator_partition_profile(
                      processor_handles_[dv_ind],
                      original_profile_config.original_profile_index);
    EXPECT_TRUE((ret_set == AMDSMI_STATUS_SETTING_UNAVAILABLE)
                || (ret_set== AMDSMI_STATUS_NO_PERM)
                || (ret_set == AMDSMI_STATUS_SUCCESS)
                || ret_set == AMDSMI_STATUS_BUSY
                || ret_set == AMDSMI_STATUS_NOT_SUPPORTED);
    amdsmi_accelerator_partition_profile_t profile = {};
    uint32_t partition_id[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    auto ret_get = amdsmi_get_gpu_accelerator_partition_profile(processor_handles_[dv_ind],
                                                                &profile, &partition_id[0]);
    if (ret_get == AMDSMI_STATUS_SUCCESS && ret_set == AMDSMI_STATUS_SUCCESS) {
      std::string profile_type_str = partition_types_map.at(profile.profile_type);
      IF_VERB(STANDARD) {
        std::cout << "\t**amdsmi_set_gpu_accelerator_partition_profile(processor_handles_["
                  << dv_ind << "],"
                  << "\n\t\t" << original_profile_config.original_profile_index
                  << " (AMDSMI_ACCELERATOR_PARTITION_"
                  << original_profile_config.original_profile_type_str
                  << "): "
                  << "\n\t\t" << smi_amdgpu_get_status_string(ret_set, false)
                  << "\n\t**amdsmi_get_gpu_accelerator_partition_profile(processor_handles_["
                  << dv_ind << "], &profile, &partition_id[0]):\n"
                  << "\t\t" << smi_amdgpu_get_status_string(ret_get, false)
                  << "\n\t**Current profile.profile_type: "
                  << profile_type_str
                  << "\n\t**profile.num_partitions: "
                  << (profile.num_partitions == kMAX_UINT32
                  ? "N/A" : std::to_string(profile.num_partitions))
                  << "\n\t**profile.profile_index: "
                  << (profile.profile_index == kMAX_UINT32
                  ? "N/A" : std::to_string(profile.profile_index))
                  << std::endl;
      }
      EXPECT_STREQ(partition_types_map.at(profile.profile_type).c_str(),
                  original_profile_config.original_profile_type_str.c_str());
      EXPECT_EQ(profile.profile_type, original_profile_config.original_profile_type);
      EXPECT_EQ(profile.profile_index, original_profile_config.original_profile_index);
    } else {
      IF_VERB(STANDARD) {
        std::cout << "\t**Could not change or read profiles. "
                  << "Skipping return to original profile on this device."
                  << "\n\t**amdsmi_set_gpu_accelerator_partition_profile(): "
                  << smi_amdgpu_get_status_string(ret_set, false)
                  << "\n\t**amdsmi_get_gpu_accelerator_partition_profile(): "
                  << smi_amdgpu_get_status_string(ret_get, false)
                  << std::endl;
      }
    }
  }
}
