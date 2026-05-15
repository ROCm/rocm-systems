/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017-2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include <stdint.h>
#include <stddef.h>
#include <unistd.h>

#include <iostream>
#include <chrono>   // NOLINT [build]
#include <map>
#include <string>

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi_test/functional/computepartition_read_write.h"
#include "rocm_smi_test/test_common.h"

TestComputePartitionReadWrite::TestComputePartitionReadWrite() : TestBase() {
  set_title("RSMI Compute Partition Read/Write Test");
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
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

const uint32_t MAX_UNSUPPORTED_PARTITIONS = 0;
const uint32_t MAX_SPX_PARTITIONS = 1;
const uint32_t MAX_DPX_PARTITIONS = 2;
const uint32_t MAX_TPX_PARTITIONS = 3;
const uint32_t MAX_QPX_PARTITIONS = 4;
const uint32_t MAX_CPX_PARTITIONS = 8;

static const std::string
computePartitionString(rsmi_compute_partition_type computeParitionType) {
  /**
   * RSMI_COMPUTE_PARTITION_INVALID = 0,
   * RSMI_COMPUTE_PARTITION_CPX,      //!< Core mode (CPX)- Per-chip XCC with
   *                                  //!< shared memory
   * RSMI_COMPUTE_PARTITION_SPX,      //!< Single GPU mode (SPX)- All XCCs work
   *                                  //!< together with shared memory
   * RSMI_COMPUTE_PARTITION_DPX,      //!< Dual GPU mode (DPX)- Half XCCs work
   *                                  //!< together with shared memory
   * RSMI_COMPUTE_PARTITION_TPX,      //!< Triple GPU mode (TPX)- One-third XCCs
   *                                  //!< work together with shared memory
   * RSMI_COMPUTE_PARTITION_QPX,      //!< Quad GPU mode (QPX)- Quarter XCCs
   *                                  //!< work together with shared memory
   */
  switch (computeParitionType) {
    case RSMI_COMPUTE_PARTITION_CPX:
      return "CPX";
    case RSMI_COMPUTE_PARTITION_SPX:
      return "SPX";
    case RSMI_COMPUTE_PARTITION_DPX:
      return "DPX";
    case RSMI_COMPUTE_PARTITION_TPX:
      return "TPX";
    case RSMI_COMPUTE_PARTITION_QPX:
      return "QPX";
    default:
      return "UNKNOWN";
  }
}

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

static const std::map<std::string, rsmi_compute_partition_type_t>
mapStringToRSMIComputePartitionTypes {
  {"DPX", RSMI_COMPUTE_PARTITION_DPX},
  {"TPX", RSMI_COMPUTE_PARTITION_TPX},
  {"QPX", RSMI_COMPUTE_PARTITION_QPX},
  {"CPX", RSMI_COMPUTE_PARTITION_CPX},
  {"SPX", RSMI_COMPUTE_PARTITION_SPX}
};

static void checkPartitionIdChanges(
  uint32_t dev, const std::string current_partition, bool isVerbose,
  bool reinitialize) {
  uint32_t max_loop = MAX_SPX_PARTITIONS;
  uint32_t current_num_devices = 0;

  // re-initialize to ensure new device ordering is followed
  if (reinitialize) {
    if (isVerbose) {
      std::cout << "\t**Reinitializing device list due to parition changes.\n";
    }
    rsmi_shut_down();
    rsmi_init(0);
  }

  if (current_partition == "DPX") {
    max_loop = MAX_DPX_PARTITIONS;
  } else if (current_partition == "TPX") {
    max_loop = MAX_TPX_PARTITIONS;
  } else if (current_partition == "QPX") {
    max_loop = MAX_QPX_PARTITIONS;
  } else if (current_partition == "CPX") {
    max_loop = MAX_CPX_PARTITIONS;
    uint16_t num_xcd;
    rsmi_status_t ret = rsmi_dev_metrics_xcd_counter_get(dev, &num_xcd);
    if (ret == RSMI_STATUS_SUCCESS) {
      max_loop = num_xcd;
      if (isVerbose) {
        std::cout << "\t**Expecting num_xcd = " << num_xcd << " to equal "
                      "total CPX nodes\n";
      }
    }
  }
  rsmi_num_monitor_devices(&current_num_devices);

  for (uint32_t i = dev; i < dev + max_loop; i++) {
    if (dev+max_loop > current_num_devices) {
      std::cout << "\t**Devices: " << dev << "; max_loop: " << max_loop
      << "; current_num_devices: " << current_num_devices << "\n";
      std::cout << "\t**[WARNING] Detected max DRM minor limitation "
      "(max of 64).\n\tPlease disable any other drivers taking up PCIe space"
      "\n\t(ex. ast or other drivers -> "
      "\"sudo rmmod amdgpu && sudo rmmod ast && sudo modprobe amdgpu\")."
      "\n\tCPX may not enumerate properly.\n";
      break;
    }
    uint32_t partition_id;
    rsmi_status_t ret = rsmi_dev_partition_id_get(i, &partition_id);
    std::cout << "\t** Checking Partition ID | Device: " << std::to_string(i)
              << "; Current Partition: " << current_partition
              << " ; Max partition IDs to check: " << max_loop << "\n";
    EXPECT_EQ(ret, RSMI_STATUS_SUCCESS);
    if (ret == RSMI_STATUS_SUCCESS && current_partition == "SPX") {
      EXPECT_LT(partition_id, max_loop);
      if (isVerbose) {
        std::cout << "\n\t**Confirmed partition_id < " << max_loop
                  << " for SPX"
                  << "\n\t**rsmi_dev_partition_id_get(" + std::to_string(i) +
                         ", &partition_id); partition_id = "
                  << static_cast<uint32_t>(partition_id) << std::endl;
      }
    } else if (ret == RSMI_STATUS_SUCCESS && current_partition == "DPX") {
      EXPECT_LT(partition_id, max_loop);
      if (isVerbose) {
        std::cout << "\n\t**Confirmed partition_id < " << max_loop
                  << " for DPX"
                  << "\n\t**rsmi_dev_partition_id_get(" + std::to_string(i) +
                         ", &partition_id); partition_id = "
                  << static_cast<uint32_t>(partition_id) << std::endl;
      }
    } else if (ret == RSMI_STATUS_SUCCESS && current_partition == "TPX") {
      EXPECT_LT(partition_id, max_loop);
      if (isVerbose) {
        std::cout << "\n\t**Confirmed partition_id < "
                  << max_loop << " for TPX"
                  << "\n\t**rsmi_dev_partition_id_get(" + std::to_string(i) +
                         ", &partition_id); partition_id = "
                  << static_cast<uint32_t>(partition_id) << std::endl;
      }
    } else if (ret == RSMI_STATUS_SUCCESS && current_partition == "QPX") {
      EXPECT_LT(partition_id, max_loop);
      if (isVerbose) {
        std::cout << "\n\t**Confirmed partition_id < "
                  << max_loop << " for QPX"
                  << "\n\t**rsmi_dev_partition_id_get(" + std::to_string(i) +
                         ", &partition_id); partition_id = "
                  << static_cast<uint32_t>(partition_id) << std::endl;
      }
    } else if (ret == RSMI_STATUS_SUCCESS && current_partition == "CPX") {
      EXPECT_LT(partition_id, max_loop);
      if (isVerbose) {
        std::cout << "\n\t**Confirmed partition_id < "
                  << max_loop << " for CPX"
                  << "\n\t**rsmi_dev_partition_id_get(" + std::to_string(i) +
                         ", &partition_id); partition_id = "
                  << static_cast<uint32_t>(partition_id) << std::endl;
      }
    } else if (ret == RSMI_STATUS_SUCCESS && current_partition == "UNKNOWN") {
      EXPECT_EQ(partition_id, max_loop - 1);
      if (isVerbose) {
        std::cout << "\n\t**Confirmed partition_id = "
                  << (max_loop - 1)
                  << " for current_partition = UNKNOWN"
                  << "\n\t**rsmi_dev_partition_id_get(" + std::to_string(i) +
                         ", &partition_id); partition_id = "
                  << static_cast<uint32_t>(partition_id) << std::endl;
      }
    }
  }
}

void TestComputePartitionReadWrite::Run(void) {
    rsmi_status_t ret, err;
    char orig_char_computePartition[255];
    orig_char_computePartition[0] = '\0';
    char current_char_computePartition[255];
    current_char_computePartition[0] = '\0';

    TestBase::Run();
    if (setup_failed_) {
      std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
      return;
    }
    bool isVerbose = (this->verbosity() &&
          this->verbosity() >= (this->TestBase::VERBOSE_STANDARD)) ? true: false;

    // Confirm system supports compute partition, before executing wait
    ret = rsmi_dev_compute_partition_get(0, orig_char_computePartition, 255);
    if (ret == RSMI_STATUS_SUCCESS) {
      system_wait(15);
    }

    // initial_num_devices - keep this value static, due to parition changes
    // fluctuating # of devices. We should end up with same # of devices at
    // end of test.
    uint32_t initial_num_devices = num_monitor_devs();
    for (uint32_t dv_ind = 0; dv_ind < initial_num_devices; ++dv_ind) {
      if (dv_ind >= 0) {
        IF_VERB(STANDARD) {
          std::cout << std::endl;
          std::cout << "\t**"
                    << "=========  LOOP THROUGH DEVICES - DEVICE #"
                    << std::to_string(dv_ind) << "  =============="
                    << std::endl;
        }
      }
      PrintDeviceHeader(dv_ind);
      bool devicePartitionUpdated = false;

      ret = rsmi_dev_partition_id_get(dv_ind, nullptr);
      EXPECT_EQ(ret, RSMI_STATUS_INVALID_ARGS);
      IF_VERB(STANDARD) {
        if (ret == RSMI_STATUS_INVALID_ARGS) {
          std::cout << "\t**" << "Confirmed rsmi_dev_partition_id_get(..,nullptr): "
                    << "RSMI_STATUS_INVALID_ARGS" << std::endl;
        }
      }

      std::string partitionStr = "";
      ret = rsmi_dev_compute_partition_get(dv_ind, orig_char_computePartition, 255);
      if (ret == RSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**rsmi_dev_compute_partition_get(): "
                  << "Not supported on this device"
                  << std::endl;
        }
        partitionStr = orig_char_computePartition;
        if (partitionStr.empty()) {
          partitionStr = computePartitionString(
            rsmi_compute_partition_type_t::RSMI_COMPUTE_PARTITION_INVALID);
        }
        // Regardless of partition support - no changes made, so no device
        // refresh needed (ie. rsmi_dev_compute_partition_set(..))
        checkPartitionIdChanges(dv_ind, partitionStr, isVerbose, false);
        continue;
    } else {
        CHK_ERR_ASRT(ret)
        std::string partitionStr = orig_char_computePartition;
        if (partitionStr.empty()) {
          partitionStr = computePartitionString(
            rsmi_compute_partition_type_t::RSMI_COMPUTE_PARTITION_INVALID);
        }
        // Regardless of partition support - no changes made, so no device
        // refresh needed (ie. rsmi_dev_compute_partition_set(..))
        checkPartitionIdChanges(dv_ind, partitionStr, isVerbose, false);
    }
    IF_VERB(STANDARD) {
      std::cout << std::endl << "\t**"
                << "Original compute partition: "
                << orig_char_computePartition << std::endl;
    }

    if ((orig_char_computePartition == NULL) ||
       (orig_char_computePartition[0] == '\0')) {
      std::cout << "***System compute partition value is not defined. "
                  "Skip compute partition test." << std::endl;
      continue;
    }
    EXPECT_EQ(RSMI_STATUS_SUCCESS, ret);

    // Verify api support checking functionality is working
    uint32_t kLength = 2;
    char smallBuffer[kLength];
    err = rsmi_dev_compute_partition_get(dv_ind, smallBuffer, kLength);
    size_t size = sizeof(smallBuffer)/sizeof(*smallBuffer);
    EXPECT_EQ(err, RSMI_STATUS_INSUFFICIENT_SIZE);
    EXPECT_EQ((size_t)kLength, size);
    IF_VERB(STANDARD) {
      if (err == RSMI_STATUS_INSUFFICIENT_SIZE) {
        std::cout << "\t**"
                  << "Confirmed RSMI_STATUS_INSUFFICIENT_SIZE was returned"
                  << "\n\t  and size matches length requested." << std::endl;
      }
    }

    // Verify api support checking functionality is working
    err = rsmi_dev_compute_partition_get(dv_ind, nullptr, 255);
    EXPECT_EQ(err, RSMI_STATUS_INVALID_ARGS);
    IF_VERB(STANDARD) {
      if (err == RSMI_STATUS_INVALID_ARGS) {
        std::cout << "\t**"
                  << "Confirmed RSMI_STATUS_INVALID_ARGS was returned."
                  << std::endl;
      }
    }

    // Verify api support checking functionality is working
    err = rsmi_dev_compute_partition_get(dv_ind, orig_char_computePartition, 0);
    EXPECT_TRUE((err == RSMI_STATUS_INVALID_ARGS) ||
                (err == RSMI_STATUS_NOT_SUPPORTED));
    IF_VERB(STANDARD) {
      if (err == RSMI_STATUS_INVALID_ARGS) {
        std::cout << "\t**"
                  << "Confirmed RSMI_STATUS_INVALID_ARGS was returned."
                  << std::endl;
      }
    }

    // Re-run original get, so we can reset to later
    ret = rsmi_dev_compute_partition_get(dv_ind, orig_char_computePartition,
                                          255);
    EXPECT_EQ(RSMI_STATUS_SUCCESS, ret);
    std::cout << "\t**rsmi_dev_compute_partition_get(" << dv_ind
    << ", " << orig_char_computePartition << ")\n";

    /**
     * RSMI_COMPUTE_PARTITION_INVALID = 0,
     * RSMI_COMPUTE_PARTITION_CPX,      //!< Core mode (CPX)- Per-chip XCC with
     *                                  //!< shared memory
     * RSMI_COMPUTE_PARTITION_SPX,      //!< Single GPU mode (SPX)- All XCCs work
     *                                  //!< together with shared memory
     * RSMI_COMPUTE_PARTITION_DPX,      //!< Dual GPU mode (DPX)- Half XCCs work
     *                                  //!< together with shared memory
     * RSMI_COMPUTE_PARTITION_TPX,      //!< Triple GPU mode (TPX)- One-third XCCs
     *                                  //!< work together with shared memory
     * RSMI_COMPUTE_PARTITION_QPX,      //!< Quad GPU mode (QPX)- Quarter XCCs
     *                                  //!< work together with shared memory
     */

    /**
     * General Loop Logic:
     * [0:SPX, 1:SPX, 2:SPX, 3:SPX]
     * [0:DPX, 1:DPX, 2: SPX, 3:SPX, 4:SPX] <- set 0 to DPX
     * [0:TPX, 1:TPX, 2:TPX, 3:SPX, 4:SPX, 5:SPX] <- set 0 to TPX
     * [0:QPX, 1:QPX, 2:QPX, 3:QPX, 4:SPX, 5:SPX, 6:SPX] <- set 0 to TPX
     * [0:CPX, 1:CPX, 2:CPX, 3:CPX, 4:CPX, 5:SPX, 6:SPX, 7:SPX] <- set 0 to CPX
     * [0:SPX, 1:SPX, 2:SPX, 3:SPX] <- reset(0)
     * +1 index
     * [0:SPX, 1:SPX, 2:SPX, 3:SPX]
     * [0:SPX, 1:DPX, 2: DPX, 3:SPX, 4:SPX] <- set 1 to DPX
     * [0:SPX, 1:TPX, 2:TPX, 3:TPX, 4:SPX, 5:SPX] <- set 1 to TPX
     * [0:SPX, 1:QPX, 2:QPX, 3:QPX, 4:QPX, 5:SPX, 6:SPX] <- set 1 to TPX
     * [0:SPX, 1:CPX, 2:CPX, 3:CPX, 4:CPX, 5:CPX, 6:SPX, 7:SPX] <- set 1 to CPX
     * [0:SPX, 1:SPX, 2:SPX, 3:SPX] <- reset(1)
     * ...
     *
     */
    std::string final_partition_state = "UNKNOWN";

    // loop through modes lower than original, but do not re-initialize this throws
    // off device id numbers in tests
    // (since we started with higher total # of devices)
    for (int partition = static_cast<int>(RSMI_COMPUTE_PARTITION_SPX);
         partition < static_cast<int>(mapStringToRSMIComputePartitionTypes.at(
                                      std::string(orig_char_computePartition)));
         partition++) {
      if (std::string(orig_char_computePartition) == "SPX") {
        break;
      }
      rsmi_compute_partition_type_t updatePartition
        = static_cast<rsmi_compute_partition_type_t>(partition);
      IF_VERB(STANDARD) {
        std::cout << std::endl;
        std::cout << "\t**"
                  << "======== TEST RSMI_COMPUTE_PARTITION_"
                  << computePartitionString(updatePartition)
                  << " ===============" << std::endl;
      }
      ret = rsmi_dev_compute_partition_set(dv_ind, updatePartition);
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "rsmi_dev_compute_partition_set(" << dv_ind
                  << ", updatePartition): "
                  << amd::smi::getRSMIStatusString(ret, false) << "\n"
                  << "\t**New Partition (set): "
                  << computePartitionString(updatePartition) << "\n";
      }
      EXPECT_TRUE(ret == RSMI_STATUS_SETTING_UNAVAILABLE
                  || ret== RSMI_STATUS_PERMISSION
                  || ret == RSMI_STATUS_SUCCESS
                  || ret == RSMI_STATUS_BUSY
                  || ret == RSMI_STATUS_NOT_SUPPORTED
                  || ret == RSMI_STATUS_INVALID_ARGS);

      if (ret == RSMI_STATUS_INVALID_ARGS) {
        std::cout << "\t**"
                  << "1st Test: Due to invalid args, skipping rest of test for this device."
                  << "\n\t Device might be in a static partition mode. "
                  << "With inability to change partition modes."
                  << std::endl;
        break;
      }
    }

    for (int partition = static_cast<int>(mapStringToRSMIComputePartitionTypes.at(
                                      std::string(orig_char_computePartition)));
         partition <= static_cast<int>(RSMI_COMPUTE_PARTITION_CPX);
         partition++) {
      rsmi_compute_partition_type_t updatePartition
        = static_cast<rsmi_compute_partition_type_t>(partition);
      IF_VERB(STANDARD) {
        std::cout << std::endl;
        std::cout << "\t**"
                  << "======== TEST RSMI_COMPUTE_PARTITION_"
                  << computePartitionString(updatePartition)
                  << " ===============" << std::endl;
      }
      ret = rsmi_dev_compute_partition_set(dv_ind, updatePartition);
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "rsmi_dev_compute_partition_set(" << dv_ind
                  << ", updatePartition): "
                  << amd::smi::getRSMIStatusString(ret, false) << "\n"
                  << "\t**New Partition (set): "
                  << computePartitionString(updatePartition) << "\n";
      }
      EXPECT_TRUE(ret == RSMI_STATUS_SETTING_UNAVAILABLE
                  || ret== RSMI_STATUS_PERMISSION
                  || ret == RSMI_STATUS_SUCCESS
                  || ret == RSMI_STATUS_BUSY
                  || ret == RSMI_STATUS_NOT_SUPPORTED
                  || ret == RSMI_STATUS_INVALID_ARGS);

      if (ret == RSMI_STATUS_BUSY) {
        IF_VERB(STANDARD) {
          std::cout << "\t**Device is currently busy.. continue\n";
        }
        system_wait(5);
        continue;
      }

      if (ret == RSMI_STATUS_INVALID_ARGS) {
        std::cout << "\t**"
                  << "2nd test: Due to invalid args, skipping rest of test for this device."
                  << "\n\t Device might be in a static partition mode. "
                  << "With inability to change partition modes."
                  << std::endl;
        break;
      }

      bool isSettingUnavailable = false;
      if (ret == RSMI_STATUS_SETTING_UNAVAILABLE || ret == RSMI_STATUS_INVALID_ARGS) {
        isSettingUnavailable = true;
      }
      rsmi_status_t retGet =
          rsmi_dev_compute_partition_get(dv_ind, current_char_computePartition,
                                         255);
      CHK_ERR_ASRT(retGet)
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Current compute partition: "
                  << current_char_computePartition
                  << std::endl;
      }
      if (isSettingUnavailable) {
        EXPECT_TRUE(ret ==RSMI_STATUS_SETTING_UNAVAILABLE || ret == RSMI_STATUS_INVALID_ARGS);
        EXPECT_STRNE(computePartitionString(updatePartition).c_str(),
                     current_char_computePartition);
        IF_VERB(STANDARD) {
          std::cout << "\t**"
                    << "Confirmed after receiving "
                    << amd::smi::getRSMIStatusString(ret, false)
                    << ",\n\t  current compute "
                    << "partition (" << current_char_computePartition
                    << ") did not update to ("
                    << computePartitionString(updatePartition) << ")"
                    << std::endl;
        }
      } else if (ret == RSMI_STATUS_SUCCESS) {
        if (strcmp(orig_char_computePartition, current_char_computePartition) !=
        0) {
          devicePartitionUpdated = true;
          final_partition_state = current_char_computePartition;
        } else {
          devicePartitionUpdated = false;
        }

        EXPECT_EQ(RSMI_STATUS_SUCCESS, ret);
        EXPECT_STREQ(computePartitionString(updatePartition).c_str(),
                     current_char_computePartition);
        IF_VERB(STANDARD) {
          std::cout << "\t**"
                    << "Confirmed current compute partition ("
                    << current_char_computePartition << ") matches"
                    << "\n\t  requested compute partition ("
                    << computePartitionString(updatePartition) << ")"
                    << std::endl;
        }

        checkPartitionIdChanges(dv_ind, computePartitionString(updatePartition),
                                isVerbose, true);
      } else {
        EXPECT_EQ(RSMI_STATUS_NOT_SUPPORTED, ret);
        IF_VERB(STANDARD) {
          std::cout << "\t**rsmi_dev_compute_partition_set(" << dv_ind
                    << ", " << computePartitionString(updatePartition) << "): "
                    << amd::smi::getRSMIStatusString(ret, false)
                    << "\n\t**Confirmed after receiving "
                    << "RSMI_STATUS_NOT_SUPPORTED,  current compute "
                    << "partition (" << current_char_computePartition
                    << ")\n\tcannot not update to ("
                    << computePartitionString(updatePartition) << ")"
                    << "; if not already the current partition." << std::endl;
        }
        checkPartitionIdChanges(dv_ind, std::string(current_char_computePartition),
                                isVerbose, false);
        // on guest this means we can't change partitions
        // some partitions will match the original partition
        if (amd::smi::is_vm_guest()) {
          continue;
        }
        EXPECT_STRNE(computePartitionString(updatePartition).c_str(),
                     current_char_computePartition);
      }
    }   // END looping through partition changes
    std::cout << "\t**=========== END PARTITION LOOP (dev = "
              << std::to_string(dv_ind) << ") ===========\n";

     /* TEST RETURN TO BOOT COMPUTE PARTITION SETTING */
    IF_VERB(STANDARD) {
      std::cout << std::endl;
      std::cout << "\t**"
                << "=========== TEST RETURN TO BOOT COMPUTE PARTITION SETTING "
                << "========" << std::endl;
    }
    std::string oldPartition = current_char_computePartition;
    rsmi_compute_partition_type_t updatePartition =
         static_cast<rsmi_compute_partition_type_t>(
          mapStringToRSMIComputePartitionTypes.at(
            std::string(orig_char_computePartition)));
    ret = rsmi_dev_compute_partition_set(dv_ind, updatePartition);


    ret = rsmi_dev_compute_partition_get(dv_ind, current_char_computePartition, 255);
    if (strcmp(oldPartition.c_str(), current_char_computePartition) !=
       0) {
        devicePartitionUpdated = true;
        final_partition_state = current_char_computePartition;
    } else {
        devicePartitionUpdated = false;
    }
    CHK_ERR_ASRT(ret)
    IF_VERB(STANDARD) {
      std::cout << "\t**" << "Current compute partition: "
                << current_char_computePartition << "\n"
                << "\t**" << "Old Partition partition (before setting to original): "
                << oldPartition << "\n"
                << "\t**" << "Original compute partition: "
                << orig_char_computePartition << "\n"
                << "\t**" << "Partitions Updated: "
                << (devicePartitionUpdated ? "TRUE" : "FALSE") << "\n";
    }

    if (devicePartitionUpdated) {
      checkPartitionIdChanges(dv_ind, std::string(current_char_computePartition),
                            isVerbose, true);
    } else {
      checkPartitionIdChanges(dv_ind, std::string(current_char_computePartition),
                            isVerbose, false);
    }
    if (devicePartitionUpdated) {
      EXPECT_STRNE(oldPartition.c_str(), current_char_computePartition);
      IF_VERB(STANDARD) {
      std::cout << "\t**"
                << "Confirmed prior partition (" << oldPartition << ") is not "
                << "equal to current\n\t  partition ("
                << current_char_computePartition << ")" << std::endl;
      }
      final_partition_state = std::string(current_char_computePartition);
    } else {
      EXPECT_STREQ(oldPartition.c_str(), current_char_computePartition);
      IF_VERB(STANDARD) {
      std::cout << "\t**"
                << "Confirmed prior partition (" << oldPartition << ") is equal"
                << " to current\n\t  partition ("
                << current_char_computePartition << ")" << std::endl;
      }
    }

    /* TEST RETURN TO ORIGINAL COMPUTE PARTITION SETTING */
    IF_VERB(STANDARD) {
      std::cout << std::endl;
      std::cout << "\t**"
                << "=========== TEST RETURN TO ORIGINAL COMPUTE PARTITION "
                << "SETTING ========" << std::endl;
    }
    rsmi_compute_partition_type_t newPartition
      = mapStringToRSMIComputePartitionTypes.at(
                                      std::string(orig_char_computePartition));
    ret = rsmi_dev_compute_partition_set(dv_ind, newPartition);
    EXPECT_TRUE(ret == RSMI_STATUS_SETTING_UNAVAILABLE
                || ret== RSMI_STATUS_PERMISSION
                || ret == RSMI_STATUS_SUCCESS
                || ret == RSMI_STATUS_BUSY
                || ret == RSMI_STATUS_NOT_SUPPORTED
                || ret == RSMI_STATUS_INVALID_ARGS);
    IF_VERB(STANDARD) {
      std::cout << "\t**"
                << "rsmi_dev_compute_partition_set("
                << std::to_string(dv_ind) << ", "
                << std::string(orig_char_computePartition) << ")" << std::endl;
      std::cout << "\t**"
                << "Attempting to returning compute partition to: "
                << computePartitionString(newPartition) << std::endl;
    }
    ret = rsmi_dev_compute_partition_get(dv_ind, current_char_computePartition,
                                          255);
    CHK_ERR_ASRT(ret)
    IF_VERB(STANDARD) {
      std::cout << "\t**" << "Attempted to set compute partition: "
                << computePartitionString(newPartition) << std::endl
                << "\t**" << "Current compute partition: "
                << current_char_computePartition
                << std::endl;
    }
    EXPECT_EQ(RSMI_STATUS_SUCCESS, ret);
    EXPECT_STREQ(computePartitionString(newPartition).c_str(),
                 current_char_computePartition);

    // only refresh (rsmi_shut_down() -> rsmi_init(0)) device list
    // if there was a partition change
    if (devicePartitionUpdated) {
      checkPartitionIdChanges(dv_ind, computePartitionString(newPartition),
                            isVerbose, true);
    } else {
      checkPartitionIdChanges(dv_ind, computePartitionString(newPartition),
                            isVerbose, false);
    }
    std::cout << "\t**"
                    << "========= END LOOP THROUGH DEVICES - DEVICE #"
                    << std::to_string(dv_ind) << "  =============="
                    << std::endl;
  }  // END looping through devices
  std::cout << "\t**=========== END TEST ===========\n";
}
