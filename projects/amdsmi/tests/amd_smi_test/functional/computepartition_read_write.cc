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
#include <unistd.h>

#include <iostream>
#include <chrono>   // NOLINT [build]
#include <map>
#include <string>
#include <vector>
#include <limits>

#include "gtest/gtest.h"
#include "../test_base.h"
#include "amd_smi/amdsmi.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "computepartition_read_write.h"

TestComputePartitionReadWrite::TestComputePartitionReadWrite() : TestBase() {
  set_title("AMDSMI Compute Partition Read/Write Test");
  set_description("The Compute Partition tests verifies that the compute "
                  "partition can be read and updated properly.");
}

TestComputePartitionReadWrite::~TestComputePartitionReadWrite(void) {
}

void TestComputePartitionReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestComputePartitionReadWrite::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestComputePartitionReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestComputePartitionReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

const uint32_t MAX_UNSUPPORTED_PARTITIONS = 0;
const uint32_t MAX_SPX_PARTITIONS = 1;  // Single GPU node
const uint32_t MAX_DPX_PARTITIONS = 2;
const uint32_t MAX_TPX_PARTITIONS = 3;
const uint32_t MAX_QPX_PARTITIONS = 4;
// const uint32_t MAX_CPX_PARTITIONS = 8;

static const std::string
computePartitionString(amdsmi_compute_partition_type_t computeParitionType) {
/**
 * typedef enum {
 *    AMDSMI_COMPUTE_PARTITION_INVALID = 0,
 *    AMDSMI_COMPUTE_PARTITION_SPX,   //!< Single GPU mode (SPX)- All XCCs work
 *                                    //!< together with shared memory
 *    AMDSMI_COMPUTE_PARTITION_DPX,   //!< Dual GPU mode (DPX)- Half XCCs work
 *                                    //!< together with shared memory
 *    AMDSMI_COMPUTE_PARTITION_TPX,   //!< Triple GPU mode (TPX)- One-third XCCs
 *                                    //!< work together with shared memory
 *    AMDSMI_COMPUTE_PARTITION_QPX,   //!< Quad GPU mode (QPX)- Quarter XCCs
 *                                    //!< work together with shared memory
 *    AMDSMI_COMPUTE_PARTITION_CPX,  //!< Core mode (CPX)- Per-chip XCC with
 *                                   //!< shared memory
 * } amdsmi_compute_partition_type_t;
 * */
  switch (computeParitionType) {
    case AMDSMI_COMPUTE_PARTITION_SPX:
      return "SPX";
    case AMDSMI_COMPUTE_PARTITION_DPX:
      return "DPX";
    case AMDSMI_COMPUTE_PARTITION_TPX:
      return "TPX";
    case AMDSMI_COMPUTE_PARTITION_QPX:
      return "QPX";
    case AMDSMI_COMPUTE_PARTITION_CPX:
      return "CPX";
    default:
      return "N/A";
  }
}

static const std::map<std::string, amdsmi_compute_partition_type_t>
mapStringToSMIComputePartitionTypes {
  {"SPX", AMDSMI_COMPUTE_PARTITION_SPX},
  {"DPX", AMDSMI_COMPUTE_PARTITION_DPX},
  {"TPX", AMDSMI_COMPUTE_PARTITION_TPX},
  {"QPX", AMDSMI_COMPUTE_PARTITION_QPX},
  {"CPX", AMDSMI_COMPUTE_PARTITION_CPX},
  {"UNKNOWN", AMDSMI_COMPUTE_PARTITION_INVALID}
};

static const std::map<amdsmi_accelerator_partition_resource_type_t,
    std::string> resource_types_map = {
  { AMDSMI_ACCELERATOR_XCC, "XCC" },
  { AMDSMI_ACCELERATOR_ENCODER, "ENCODER" },
  { AMDSMI_ACCELERATOR_DECODER, "DECODER" },
  { AMDSMI_ACCELERATOR_DMA, "DMA" },
  { AMDSMI_ACCELERATOR_JPEG, "JPEG" },
  { AMDSMI_ACCELERATOR_MAX, "MAX" },
};

static const std::map<amdsmi_accelerator_partition_type_t, std::string> partition_types_map = {
  { AMDSMI_ACCELERATOR_PARTITION_INVALID, "N/A" },
  { AMDSMI_ACCELERATOR_PARTITION_SPX, "SPX" },
  { AMDSMI_ACCELERATOR_PARTITION_DPX, "DPX" },
  { AMDSMI_ACCELERATOR_PARTITION_TPX, "TPX" },
  { AMDSMI_ACCELERATOR_PARTITION_QPX, "QPX" },
  { AMDSMI_ACCELERATOR_PARTITION_CPX, "CPX" },
  { AMDSMI_ACCELERATOR_PARTITION_MAX, "MAX" },
};

static const std::map<amdsmi_accelerator_partition_type_t, std::string> accelerator_types_map = {
  { AMDSMI_ACCELERATOR_PARTITION_INVALID, "AMDSMI_ACCELERATOR_PARTITION_INVALID" },
  { AMDSMI_ACCELERATOR_PARTITION_SPX, "AMDSMI_ACCELERATOR_PARTITION_SPX" },
  { AMDSMI_ACCELERATOR_PARTITION_DPX, "AMDSMI_ACCELERATOR_PARTITION_DPX" },
  { AMDSMI_ACCELERATOR_PARTITION_TPX, "AMDSMI_ACCELERATOR_PARTITION_TPX" },
  { AMDSMI_ACCELERATOR_PARTITION_QPX, "AMDSMI_ACCELERATOR_PARTITION_QPX" },
  { AMDSMI_ACCELERATOR_PARTITION_CPX, "AMDSMI_ACCELERATOR_PARTITION_CPX" },
  { AMDSMI_ACCELERATOR_PARTITION_MAX, "AMDSMI_ACCELERATOR_PARTITION_MAX" },
};

static void system_wait(int seconds) {
  // Adding a delay - since changing partitions depends on gpus not
  // being in an active state, we'll wait a few seconds before starting
  // full testing
  auto start = std::chrono::high_resolution_clock::now();
  int waitTime = seconds;
  std::cout << "** Waiting for "
            << std::dec << waitTime
            << " seconds, for any GPU"
            << " activity to clear up. **" << std::endl;
  sleep(waitTime);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "** Waiting took " << duration.count() / 1000000
            << " seconds **" << std::endl;
}

static void getProcessorHandles(amdsmi_processor_handle* processor_handles,
                                uint32_t curr_num_devices) {
  if (processor_handles == nullptr) {
    // resize the processor_handles array
    processor_handles = new amdsmi_processor_handle[curr_num_devices];
  }

  for (uint32_t i = 0; i < curr_num_devices; i++) {
    amdsmi_processor_handle p_handle = {};
    smi_amdgpu_get_processor_handle_by_index(i, &p_handle);
    processor_handles[i] = p_handle;

    std::cout << "\t**getProcessorHandles() | processor_handles["
              << std::setfill('0') << std::setw(2)
              << i << "]:\t" << processor_handles[i] << std::endl;
  }
}

static void checkPartitionIdChanges(amdsmi_processor_handle* const processor_handle,
                                    uint32_t dev_id,
                                    const std::string current_partition,
                                    bool isVerbose,
                                    bool reinitialize) {
  uint32_t max_loop = 0;
  uint32_t current_num_devices = 0;
  uint32_t dev = 0;
  uint32_t prev_num_devices = 0;
  smi_amdgpu_get_device_count(&prev_num_devices);

  // re-initialize to ensure new device ordering is followed
  if (reinitialize) {
    if (isVerbose) {
      std::cout << "\t**REINITIALIZING device list due to parition changes.\n";
    }
    amdsmi_shut_down();
    amdsmi_init(AMDSMI_INIT_AMD_GPUS);
  }

  smi_amdgpu_get_device_count(&current_num_devices);

  // std::cout << "\t**Checking Partition ID Changes 3\n";
  if (isVerbose) {
    std::cout << "\t**Device (dev) #: " << dev
      << "; Device dev_id: " << dev_id
      << "; max_loop: " << static_cast<int>(max_loop)
      << "; current_num_devices: " << current_num_devices << "\n";
  }
  // Allocate the memory for the device handlers on the socket
  std::vector<amdsmi_processor_handle> curr_processor_handles(current_num_devices);
  getProcessorHandles(&curr_processor_handles[0], current_num_devices);

  if (current_partition == "SPX" || current_partition == "N/A") {
    max_loop = MAX_SPX_PARTITIONS;
  } else if (current_partition == "DPX") {
    max_loop = MAX_DPX_PARTITIONS;
  } else if (current_partition == "TPX") {
    max_loop = MAX_TPX_PARTITIONS;
  } else if (current_partition == "QPX") {
    max_loop = MAX_QPX_PARTITIONS;
  } else if (current_partition == "CPX") {
    uint16_t num_xcd;

    auto ret = amdsmi_get_gpu_xcd_counter(curr_processor_handles[dev_id], &num_xcd);
    if (ret == AMDSMI_STATUS_SUCCESS) {
      max_loop = static_cast<uint32_t>(num_xcd);
      if (isVerbose) {
        std::cout << "\t**Expecting num_xcd = " << num_xcd << " to equal "
                      "total CPX nodes\n";
      }
    }
  }

  if (dev_id + max_loop > current_num_devices) {
    if (isVerbose) {
      std::cout
      << "\t**[WARNING] Readjusting dev_id (was " << dev_id << ")= " << current_num_devices
      << " - " << max_loop << ": " << (current_num_devices - max_loop) << "\n";
    }
    dev_id = current_num_devices - max_loop;
  }

  for (uint32_t i = dev_id; i < dev_id + max_loop; i++) {
    if (isVerbose) {
      std::cout << "\t**checkPartitionIdChanges DEVICE INFO ===============\n";
      std::cout << "\t**Device (i): " << static_cast<uint32_t>(i) << std::endl;
      std::cout << "\t**dev_id: " << static_cast<uint32_t>(dev_id) << std::endl;
      std::cout << "\t**Device Index: " << static_cast<uint32_t>(dev) << std::endl;

      std::cout << "\t**Processor Handle: " << processor_handle[i] << std::endl;
      std::cout << "\t**Current Processor Handle: " << curr_processor_handles[i] << std::endl;
      std::cout << "\t**Current # of devices: " << static_cast<uint32_t>(current_num_devices)
      << std::endl;
      std::cout << "\t**END checkPartitionIdChanges DEVICE INFO =============\n";
      std::cout << "\t**Device (i) #: " << i
      << "; Device dev_id: " << dev_id
      << "\n\t\t** max_loop: " << static_cast<int>(max_loop)
      << "\n\t\t** current_num_devices: " << current_num_devices << "\n";
    }
    if (i >= current_num_devices) {
      if (isVerbose) {
        std::cout << "\t**[WARNING] Detected max DRM minor limitation "
        "(max of 64).\n\tPlease disable any other drivers taking up PCIe space"
        "\n\t(ex. ast or other drivers -> "
        "\"sudo rmmod amdgpu && sudo rmmod ast && sudo modprobe amdgpu\")."
        "\n\tCPX may not enumerate properly.\n";
      }
      break;
    }
    amdsmi_kfd_info_t kfd_info;
    amdsmi_status_t ret = amdsmi_get_gpu_kfd_info(curr_processor_handles[i], &kfd_info);
    if (isVerbose) {
      std::cout << "\t**Checking Partition ID | Device: " << std::to_string(i)
                << "\n\t\t**Current Partition: " << current_partition
                << "\n\t\t**Max partition IDs to check: " << max_loop
                << "\n\t\t**Current Partition ID: " << std::to_string(kfd_info.current_partition_id)
                << "\n";
    }
    EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS);
    if (ret == AMDSMI_STATUS_SUCCESS && current_partition == "SPX") {
      EXPECT_TRUE(kfd_info.current_partition_id <= max_loop);
      if (isVerbose) {
        std::cout << "\n\t**Confirmed partition_id < " << max_loop
                  << " for SPX"
                  << "\n\t**amdsmi_get_gpu_kfd_info(" + std::to_string(i) +
                         ", &kfd_info); kfd_info.partition_id: "
                  << static_cast<uint32_t>(kfd_info.current_partition_id) << std::endl;
      }
    } else if (ret == AMDSMI_STATUS_SUCCESS && current_partition == "DPX") {
      EXPECT_TRUE(kfd_info.current_partition_id <= max_loop);
      if (isVerbose) {
        std::cout << "\n\t**Confirmed partition_id < " << max_loop
                  << " for DPX"
                  << "\n\t**amdsmi_get_gpu_kfd_info(" + std::to_string(i) +
                         ", &kfd_info); kfd_info.partition_id: "
                  << static_cast<uint32_t>(kfd_info.current_partition_id) << std::endl;
      }
    } else if (ret == AMDSMI_STATUS_SUCCESS && current_partition == "TPX") {
      EXPECT_TRUE(kfd_info.current_partition_id <= max_loop);
      if (isVerbose) {
        std::cout << "\n\t**Confirmed partition_id < "
                  << max_loop << " for TPX"
                  << "\n\t**amdsmi_get_gpu_kfd_info(" + std::to_string(i) +
                         ", &kfd_info); kfd_info.partition_id: "
                  << static_cast<uint32_t>(kfd_info.current_partition_id) << std::endl;
      }
    } else if (ret == AMDSMI_STATUS_SUCCESS && current_partition == "QPX") {
      EXPECT_TRUE(kfd_info.current_partition_id <= max_loop);
      if (isVerbose) {
        std::cout << "\n\t**Confirmed partition_id < "
                  << max_loop << " for QPX"
                  << "\n\t**amdsmi_get_gpu_kfd_info(" + std::to_string(i) +
                         ", &kfd_info); kfd_info.partition_id: "
                  << static_cast<uint32_t>(kfd_info.current_partition_id) << std::endl;
      }
    } else if (ret == AMDSMI_STATUS_SUCCESS && current_partition == "CPX") {
      EXPECT_TRUE(kfd_info.current_partition_id <= max_loop);
      if (isVerbose) {
        std::cout << "\n\t**Confirmed partition_id < "
                  << max_loop << " for CPX"
                 << "\n\t**amdsmi_get_gpu_kfd_info(" + std::to_string(i) +
                         ", &kfd_info); kfd_info.partition_id: "
                  << static_cast<uint32_t>(kfd_info.current_partition_id) << std::endl;
      }
    } else if (ret == AMDSMI_STATUS_SUCCESS && current_partition == "N/A") {
      EXPECT_EQ(kfd_info.current_partition_id, max_loop - 1);
      if (isVerbose) {
        std::cout << "\n\t**Confirmed partition_id = "
                  << (max_loop - 1)
                  << " for current_partition = N/A"
                  << "\n\t**amdsmi_get_gpu_kfd_info(" + std::to_string(i) +
                         ", &kfd_info); kfd_info.partition_id: "
                  << static_cast<uint32_t>(kfd_info.current_partition_id) << std::endl;
      }
    }
  }
}


std::string getResourceType(amdsmi_accelerator_partition_resource_type_t resource_type) {
  std::string resource_type_str = "";
  switch (resource_type) {
    case AMDSMI_ACCELERATOR_XCC:
      resource_type_str = "XCC";
      break;
    case AMDSMI_ACCELERATOR_ENCODER:
      resource_type_str = "ENCODER";
      break;
    case AMDSMI_ACCELERATOR_DECODER:
      resource_type_str = "DECODER";
      break;
    case AMDSMI_ACCELERATOR_DMA:
      resource_type_str = "DMA";
      break;
    case AMDSMI_ACCELERATOR_JPEG:
      resource_type_str = "JPEG";
      break;
    case AMDSMI_ACCELERATOR_MAX:
      resource_type_str = "MAX";
      break;
    default:
      resource_type_str = "N/A";
      break;
  }
  return resource_type_str;
}

void TestComputePartitionReadWrite::Run(void) {
  amdsmi_status_t ret;
  constexpr uint32_t k255Len = 255;
  char orig_char_computePartition[k255Len];
  orig_char_computePartition[0] = '\0';
  char current_char_computePartition[k255Len];
  current_char_computePartition[0] = '\0';
  const uint32_t kMAX_UINT32 = std::numeric_limits<uint32_t>::max();
  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  bool isVerbose = (this->verbosity() &&
        this->verbosity() >= (this->TestBase::VERBOSE_STANDARD)) ? true: false;
  // Confirm system supports compute partition, before executing wait
  ret = amdsmi_get_gpu_compute_partition(0, orig_char_computePartition, k255Len);
  if (ret == AMDSMI_STATUS_SUCCESS) {
    system_wait(15);
  }

  IF_VERB(STANDARD) {
    std::cout << "\t**======================================================================\n";
    std::cout << "\t**Test #1: Get/Set Compute Partition (old functionality) ===============\n";
    std::cout << "\t**======================================================================\n";
  }

  // // TEST 1: Set/Get Compute Partition (old functionality)
  uint32_t initial_num_devices = num_monitor_devs();
  for (uint32_t dv_ind = 0; dv_ind < initial_num_devices; ++dv_ind) {
    if (dv_ind != 0) {
      std::cout << "\n";
    }
    PrintDeviceHeader(processor_handles_[dv_ind]);

    ret = amdsmi_get_gpu_compute_partition(processor_handles_[dv_ind], orig_char_computePartition,
                                            k255Len);
    EXPECT_TRUE(ret == AMDSMI_STATUS_SUCCESS
                || ret == AMDSMI_STATUS_NOT_SUPPORTED);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**" <<  "amdsmi_get_gpu_compute_partition: "
                  << "Not supported on this machine" << std::endl;
      }
      continue;
    }
    for (int partition = static_cast<int>(AMDSMI_COMPUTE_PARTITION_SPX);
         partition <= static_cast<int>(AMDSMI_COMPUTE_PARTITION_CPX);
         partition++) {
      amdsmi_compute_partition_type_t updatePartition
        = static_cast<amdsmi_compute_partition_type_t>(partition);

      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "======== TEST AMDSMI_COMPUTE_PARTITION_"
                  << computePartitionString(updatePartition)
                  << " ===============" << std::endl;
      }

      auto ret_set = amdsmi_set_gpu_compute_partition(processor_handles_[dv_ind], updatePartition);
      IF_VERB(STANDARD) {
          std::cout << "\t**" << "amdsmi_set_gpu_compute_partition(processor_handles_["
                    << dv_ind << "], " << computePartitionString(updatePartition) << "): "
                    << smi_amdgpu_get_status_string(ret_set, false) << "\n"
                    << "\t**Requested Set Partition: "
                    << computePartitionString(updatePartition) << "\n"
                    << "\t**Original Partition: " << orig_char_computePartition
                    << std::endl;
      }
      EXPECT_TRUE(ret_set == AMDSMI_STATUS_SETTING_UNAVAILABLE
        || ret_set== AMDSMI_STATUS_NO_PERM
        || ret_set == AMDSMI_STATUS_SUCCESS
        || ret_set == AMDSMI_STATUS_BUSY
        || ret_set == AMDSMI_STATUS_NOT_SUPPORTED
        || ret_set == AMDSMI_STATUS_INVAL);
      if (ret_set == AMDSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**" <<  "amdsmi_set_gpu_compute_partition: "
                    << "Not supported on this machine" << std::endl;
        }
        continue;
      }
      if (ret_set == AMDSMI_STATUS_INVAL) {
        std::cout << "\t**"
                  << "1st Test: Due to invalid args, skipping rest of test for this device."
                  << "\n\t Device might be in a static partition mode. "
                  << "With inability to change partition modes."
                  << std::endl;
        break;
      }

      ret = amdsmi_get_gpu_compute_partition(processor_handles_[dv_ind],
                                            current_char_computePartition,
                                            k255Len);
      IF_VERB(STANDARD) {
        std::cout << "\t**amdsmi_get_gpu_compute_partition(processor_handles_[" << dv_ind << "], "
                  << current_char_computePartition << "): "
                  << smi_amdgpu_get_status_string(ret, false)
                  << "\n\t**Current Partition (get): "
                  << current_char_computePartition
                  << std::endl;
      }
      if (ret_set == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS);
        EXPECT_EQ(updatePartition, mapStringToSMIComputePartitionTypes.at(
          std::string(current_char_computePartition)));
      } else {
        EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS);
        EXPECT_NE(updatePartition, mapStringToSMIComputePartitionTypes.at(
          std::string(current_char_computePartition)));
      }
    }
    amdsmi_compute_partition_type_t updatePartition =
         static_cast<amdsmi_compute_partition_type_t>(
          mapStringToSMIComputePartitionTypes.at(
            std::string(orig_char_computePartition)));
    auto ret_set = amdsmi_set_gpu_compute_partition(processor_handles_[dv_ind], updatePartition);
    EXPECT_TRUE(ret_set == AMDSMI_STATUS_SETTING_UNAVAILABLE
      || ret_set== AMDSMI_STATUS_NO_PERM
      || ret_set == AMDSMI_STATUS_SUCCESS
      || ret_set == AMDSMI_STATUS_BUSY
      || ret_set == AMDSMI_STATUS_NOT_SUPPORTED
      || ret_set == AMDSMI_STATUS_INVAL);
  }

  IF_VERB(STANDARD) {
    std::cout << "\n";
    std::cout << "\t**======================================================================\n";
    std::cout << "\t**Test #2: Get/Set Compute Partition (new functionality) ===============\n";
    std::cout << "\t**======================================================================\n";
  }

  // TEST 2: Set/Get Compute Partition (new functionality)
  initial_num_devices = num_monitor_devs();
  amdsmi_accelerator_partition_type_t primary_partition_type = AMDSMI_ACCELERATOR_PARTITION_INVALID;
  uint32_t primary_index = 0;
  for (uint32_t dv_ind = 0; dv_ind < initial_num_devices; ++dv_ind) {
    if (dv_ind != 0) {
      std::cout << "\n";
    }
    IF_VERB(STANDARD) {
      std::cout << "\n";
      std::cout << "\t**======================================================================\n";
      std::cout << "\t**Test #2: Get/Set Compute Partition (new functionality) ===============\n";
      std::cout << "\t**DEVICE: #" << std::dec << std::setw(2) << std::setfill('0') << dv_ind
                << " ==========================================================\n";
      std::cout << "\t**======================================================================\n";
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
    AcceleratorProfileConfig original_profile_config = {};
    original_profile_config
      = getAvailableProfileConfigs(dv_ind, profile, profile_config, isVerbose);

    IF_VERB(STANDARD) {
      std::cout << "\t**=========================================================\n";
      std::cout << "\t**Checking invalid profile Set ============================\n";
      std::cout << "\t**=========================================================\n";
    }
    // Test setting invalid profile index
    auto ret_expect_invalid = amdsmi_set_gpu_accelerator_partition_profile(
                                processor_handles_[dv_ind],
                                profile_config.num_profiles);
    IF_VERB(STANDARD) {
      std::cout << "\t**amdsmi_set_gpu_accelerator_partition_profile(processor_handles_["
                << dv_ind << "], " << profile_config.num_profiles << "):"
                << "\n\t\t" << smi_amdgpu_get_status_string(ret_expect_invalid, false)
                << std::endl;
    }
    EXPECT_TRUE(ret_expect_invalid == AMDSMI_STATUS_INVAL
                || ret_expect_invalid == AMDSMI_STATUS_NOT_SUPPORTED);

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
        std::cout << "\t**" <<  "amdsmi_get_gpu_accelerator_partition_profile_config: "
                  << "Not supported on this machine" << std::endl;
      }
      continue;
    }

    for (uint32_t config = 0; config < profile_config.num_profiles; config++) {
      auto new_profile = profile_config.profiles[config];
      std::string new_profile_type_str = "N/A";
      if (new_profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_SPX) {
        new_profile_type_str = "SPX";
      } else if (new_profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_DPX) {
        new_profile_type_str = "DPX";
      } else if (new_profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_TPX) {
        new_profile_type_str = "TPX";
      } else if (new_profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_QPX) {
        new_profile_type_str = "QPX";
      } else if (new_profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_CPX) {
        new_profile_type_str = "CPX";
      }

      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "======== TEST AMDSMI_ACCELERATOR_PARTITION_"
                  << new_profile_type_str << " (profile_index: "
                  << profile_config.profiles[config].profile_index << ")"
                  << " ===============" << std::endl;
      }

      auto ret_set = amdsmi_set_gpu_accelerator_partition_profile(
                        processor_handles_[dv_ind],
                        profile_config.profiles[config].profile_index);
      IF_VERB(STANDARD) {
          std::cout << "\t**" << "amdsmi_set_gpu_accelerator_partition_profile(processor_handles_["
                    << dv_ind << "], " << profile_type_str << "): "
                    << smi_amdgpu_get_status_string(ret_set, false) << "\n"
                    << "\t**Requested Set Partition: "
                    << new_profile_type_str << "\n"
                    << "\t**Original Partition: "
                    << original_profile_config.original_profile_type_str
                    << std::endl;
      }
      EXPECT_TRUE(ret_set == AMDSMI_STATUS_SETTING_UNAVAILABLE
                  || ret_set== AMDSMI_STATUS_NO_PERM
                  || ret_set == AMDSMI_STATUS_SUCCESS
                  || ret_set == AMDSMI_STATUS_BUSY
                  || ret_set == AMDSMI_STATUS_NOT_SUPPORTED
                  || ret_set == AMDSMI_STATUS_INVAL);
      if (ret_set == AMDSMI_STATUS_INVAL) {
        std::cout << "\t**"
                  << "2nd Test: Due to invalid args, skipping rest of test for this device."
                  << "\n\t Device might be in a static partition mode. "
                  << "With inability to change partition modes."
                  << std::endl;
        break;
      }
      if (ret_set == AMDSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**" <<  "amdsmi_set_gpu_accelerator_partition_profile: "
                    << "Not supported on this machine" << std::endl;
        }
        continue;
      }

      auto ret_get = amdsmi_get_gpu_accelerator_partition_profile(processor_handles_[dv_ind],
                        &profile, &partition_id[0]);
      if (ret_get == AMDSMI_STATUS_SUCCESS && ret_set == AMDSMI_STATUS_SUCCESS) {
        profile_type_str = partition_types_map.at(profile.profile_type);
        IF_VERB(STANDARD) {
          std::cout << "\t**amdsmi_set_gpu_accelerator_partition_profile(processor_handles_["
          << dv_ind << "],"
          << "\n\t\t" << profile_config.profiles[config].profile_index
          << " (AMDSMI_ACCELERATOR_PARTITION_" << new_profile_type_str
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
                    new_profile_type_str.c_str());
        EXPECT_EQ(profile.profile_type, new_profile.profile_type);
        EXPECT_EQ(profile.profile_index, new_profile.profile_index);
      }
    }
    IF_VERB(STANDARD) {
      std::cout << "\t**Device Index: " << dv_ind << std::endl
                << "\t**======== Return to original AMDSMI_ACCELERATOR_PARTITION_"
                << original_profile_config.original_profile_type_str << " (profile_index: "
                << (original_profile_config.original_profile_index == kMAX_UINT32 ?
                    "N/A" : std::to_string(original_profile_config.original_profile_index)) << ")"
                << " ===============" << std::endl;
    }
    auto ret_set = amdsmi_set_gpu_accelerator_partition_profile(
                      processor_handles_[dv_ind],
                      original_profile_config.original_profile_index);
    IF_VERB(STANDARD) {
      std::cout << "\t**amdsmi_set_gpu_accelerator_partition_profile(processor_handles_["
                << dv_ind << "],"
                << "\n\t\t" << original_profile_config.original_profile_index
                << " (AMDSMI_ACCELERATOR_PARTITION_"
                << original_profile_config.original_profile_type_str
                << "): "
                << "\n\t\t" << smi_amdgpu_get_status_string(ret_set, false)
                << std::endl;
    }
    EXPECT_TRUE(ret_set == AMDSMI_STATUS_SETTING_UNAVAILABLE
      || ret_set== AMDSMI_STATUS_NO_PERM
      || ret_set == AMDSMI_STATUS_SUCCESS
      || ret_set == AMDSMI_STATUS_BUSY
      || ret_set == AMDSMI_STATUS_NOT_SUPPORTED
      || ret_set == AMDSMI_STATUS_INVAL);
    auto ret_get = amdsmi_get_gpu_accelerator_partition_profile(processor_handles_[dv_ind],
                                                                &profile, &partition_id[0]);
    IF_VERB(STANDARD) {
      std::cout << "\n\t**amdsmi_get_gpu_accelerator_partition_profile(processor_handles_["
                << dv_ind << "], &profile, &partition_id[0]):\n"
                << "\t\t" << smi_amdgpu_get_status_string(ret_get, false)
                << std::endl;
    }

    // older kernels do not support this feature
    if (original_profile_config.original_profile_index == kMAX_UINT32) {
      EXPECT_EQ(ret_get, AMDSMI_STATUS_NOT_SUPPORTED);
      IF_VERB(STANDARD) {
        std::cout << "\t**" << "amdsmi_get_gpu_accelerator_partition_profile: "
                  << "Not supported on this machine, skipping remaining tests." << std::endl;
      }
      break;
    }

    if (ret_get == AMDSMI_STATUS_SUCCESS && ret_set == AMDSMI_STATUS_SUCCESS) {
      profile_type_str = partition_types_map.at(profile.profile_type);
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
    }
  }  // END for (uint32_t dv_ind = 0; dv_ind < initial_num_devices; ++dv_ind)

  IF_VERB(STANDARD) {
    std::cout << "\n";
    std::cout << "\t**======================================================================\n";
    std::cout << "\t**Test #3: Check fluctuating # of devices & partition IDs ==============\n";
    std::cout << "\t**======================================================================\n";
  }

  // ---------------------------------------------------------//
  // TEST 3: Check fluctuating # of devices & partition IDs   //
  // ---------------------------------------------------------//
  initial_num_devices = num_monitor_devs();
  for (uint32_t dv_ind = 0; dv_ind < initial_num_devices; ++dv_ind) {
    if (dv_ind != 0) {
      std::cout << "\n";
    }
    IF_VERB(STANDARD) {
      std::cout << "\n";
      std::cout << "\t**======================================================================\n";
      std::cout << "\t**Test #3: Check fluctuating # of devices & partition IDs ==============\n";
      std::cout << "\t**DEVICE: #" << std::dec << std::setw(2) << std::setfill('0') << dv_ind
                << " ========================================================\n";
      std::cout << "\t**======================================================================\n";
    }
    // Leaving for debug purposes
    uint32_t device_index = 0;
    amdsmi_processor_handle p_handle = {};
    uint32_t current_num_devices = 0;
    smi_amdgpu_get_device_count(&current_num_devices);
    smi_amdgpu_get_processor_handle_by_index(dv_ind, &p_handle);
    smi_amdgpu_get_device_index(p_handle, &device_index);
    IF_VERB(STANDARD) {
      std::cout << "\t=========== START INDEX/p_handle DEVICE INFO 1 ===============\n";
      std::cout << "\t**Dv_ind: " << dv_ind << std::endl;
      std::cout << "\t**Device Index: " << device_index << std::endl;
      std::cout << "\t**Processor Handle (processor_handles_[dv_ind]): "
                << processor_handles_[dv_ind] << std::endl;
      std::cout << "\t**Processor Handle: " << p_handle << std::endl;
      std::cout << "\t**Current # of devices: " << current_num_devices << std::endl;
      std::cout << "\t=========== END INDEX/p_handle DEVICE INFO 1 =============\n";
    }


    PrintDeviceHeader(p_handle);
    ret = amdsmi_get_gpu_compute_partition(p_handle, orig_char_computePartition,
                                            k255Len);
    EXPECT_TRUE(ret == AMDSMI_STATUS_SUCCESS
                || ret == AMDSMI_STATUS_NOT_SUPPORTED);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**" <<  "amdsmi_get_gpu_compute_partition: "
                  << "Not supported on this machine" << std::endl;
      }
      continue;
    }
    for (int partition = static_cast<int>(mapStringToSMIComputePartitionTypes.at(
                            std::string(orig_char_computePartition)));
         partition <= static_cast<int>(AMDSMI_COMPUTE_PARTITION_CPX);
         partition++) {
      uint32_t device_index2 = 0;
      amdsmi_processor_handle p_handle2 = {};
      smi_amdgpu_get_device_count(&current_num_devices);
      smi_amdgpu_get_processor_handle_by_index(dv_ind, &p_handle2);
      smi_amdgpu_get_device_index(p_handle2, &device_index2);
      IF_VERB(STANDARD) {
        std::cout << "\t=========== INDEX/p_handle DEVICE INFO 2 ===============\n";
        std::cout << "\t**Dv_ind: " << dv_ind << std::endl;
        std::cout << "\t**Device Index2: " << device_index2 << std::endl;
        std::cout << "\t**Processor Handle (processor_handles_[dv_ind]): "
                << processor_handles_[dv_ind] << std::endl;
        std::cout << "\t**Processor Handle: " << p_handle << std::endl;
        std::cout << "\t**Processor Handle2: " << p_handle2 << std::endl;
        std::cout << "\t**Current # of devices: " << current_num_devices << std::endl;
        std::cout << "\t=========== END INDEX/p_handle DEVICE INFO 2 =============\n";
      }

      amdsmi_compute_partition_type_t updatePartition
        = static_cast<amdsmi_compute_partition_type_t>(partition);
      auto ret_set = amdsmi_set_gpu_compute_partition(p_handle2, updatePartition);
      IF_VERB(STANDARD) {
          std::cout << "\t**" << "amdsmi_set_gpu_compute_partition(processor_handles_["
                    << dv_ind << "], " << computePartitionString(updatePartition) << "): "
                    << smi_amdgpu_get_status_string(ret_set, false) << "\n"
                    << "\t**Requested Set Partition: "
                    << computePartitionString(updatePartition) << "\n"
                    << "\t**Original Partition: " << orig_char_computePartition
                    << std::endl;
      }
      EXPECT_TRUE(ret_set == AMDSMI_STATUS_SETTING_UNAVAILABLE
        || ret_set== AMDSMI_STATUS_NO_PERM
        || ret_set == AMDSMI_STATUS_SUCCESS
        || ret_set == AMDSMI_STATUS_BUSY
        || ret_set == AMDSMI_STATUS_NOT_SUPPORTED
        || ret_set == AMDSMI_STATUS_INVAL);
      if (ret_set == AMDSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**" <<  "amdsmi_set_gpu_compute_partition: "
                    << "Not supported on this machine" << std::endl;
        }
        continue;
      }
      if (ret_set == AMDSMI_STATUS_INVAL) {
        std::cout << "\t**"
                  << "3rd Test: Due to invalid args, skipping rest of test for this device."
                  << "\n\t Device might be in a static partition mode. "
                  << "With inability to change partition modes."
                  << std::endl;
        break;
      }

      ret = amdsmi_get_gpu_compute_partition(p_handle2,
                                            current_char_computePartition,
                                            k255Len);
      IF_VERB(STANDARD) {
        std::cout << "\t**amdsmi_get_gpu_compute_partition(processor_handles_[" << dv_ind << "], "
                  << current_char_computePartition << "): "
                  << smi_amdgpu_get_status_string(ret, false)
                  << "\n\t**Current Partition (get): "
                  << current_char_computePartition
                  << std::endl;
      }
      if (ret_set == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS);
        EXPECT_EQ(updatePartition, mapStringToSMIComputePartitionTypes.at(
          std::string(current_char_computePartition)));
        checkPartitionIdChanges(processor_handles_, dv_ind,
          std::string(current_char_computePartition),
          isVerbose, true);
      } else {
        EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS);
        EXPECT_NE(updatePartition, mapStringToSMIComputePartitionTypes.at(
          std::string(current_char_computePartition)));
      }
    }

    uint32_t device_index3 = 0;
    amdsmi_processor_handle p_handle3 = {};
    smi_amdgpu_get_processor_handle_by_index(dv_ind, &p_handle3);
    smi_amdgpu_get_device_index(p_handle3, &device_index3);

    amdsmi_compute_partition_type_t updatePartition =
         static_cast<amdsmi_compute_partition_type_t>(
          mapStringToSMIComputePartitionTypes.at(
            std::string(orig_char_computePartition)));
    IF_VERB(STANDARD) {
      std::cout << "\t**ABOUT TO GO BACK TO ORIGINAL PARTITION ("
                << orig_char_computePartition << ")\n";
    }
    auto ret_set = amdsmi_set_gpu_compute_partition(p_handle3, updatePartition);
    checkPartitionIdChanges(processor_handles_, dv_ind, std::string(orig_char_computePartition),
          isVerbose, true);
    if (ret_set == AMDSMI_STATUS_SUCCESS) {
      EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(updatePartition, mapStringToSMIComputePartitionTypes.at(
        std::string(orig_char_computePartition)));
    } else {
      EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS);
      // on guest this means we can't change partitions
      // some partitions will match the original partition
      if (amd::smi::is_vm_guest()) {
        EXPECT_EQ(updatePartition, mapStringToSMIComputePartitionTypes.at(
                  std::string(orig_char_computePartition)));
      } else {
        EXPECT_EQ(updatePartition, mapStringToSMIComputePartitionTypes.at(
                  std::string(orig_char_computePartition)));
      }
    }
    IF_VERB(STANDARD) {
      std::cout << "\t**Get/Set Test #3 (dev_ind: " << std::dec
                << dv_ind << "): Check fluctuating # of devices & partition IDs ===============\n";
    }
  }

  IF_VERB(STANDARD) {
    std::cout << "\n";
    std::cout << "\t**======================================================================\n";
    std::cout << "\t**END Tests ============================================================\n";
    std::cout << "\t**======================================================================\n";
  }
}
