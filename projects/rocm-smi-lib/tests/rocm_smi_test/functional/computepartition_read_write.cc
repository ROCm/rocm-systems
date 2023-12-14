/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017-2023, Advanced Micro Devices, Inc.
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
  {"CPX", RSMI_COMPUTE_PARTITION_CPX},
  {"SPX", RSMI_COMPUTE_PARTITION_SPX},
  {"DPX", RSMI_COMPUTE_PARTITION_DPX},
  {"TPX", RSMI_COMPUTE_PARTITION_TPX},
  {"QPX", RSMI_COMPUTE_PARTITION_QPX}
};

void TestComputePartitionReadWrite::Run(void) {
  rsmi_status_t ret, err;
  char orig_char_computePartition[255];
  char current_char_computePartition[255];

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  // Confirm system supports compute partition, before executing wait
  ret = rsmi_dev_compute_partition_get(0, orig_char_computePartition, 255);
  if (ret == RSMI_STATUS_SUCCESS) {
    system_wait(25);
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    if (dv_ind != 0) {
      IF_VERB(STANDARD) {
        std::cout << std::endl;
      }
    }
    PrintDeviceHeader(dv_ind);
    bool devicePartitionUpdated = false;

    // Standard checks to see if API is supported, before running full tests
    ret = rsmi_dev_compute_partition_get(dv_ind, orig_char_computePartition,
                                          255);
    if (ret == RSMI_STATUS_NOT_SUPPORTED) {
       IF_VERB(STANDARD) {
          std::cout << "\t**" <<  ": "
                    << "Not supported on this device" << std::endl;
        }
        continue;
    } else {
        CHK_ERR_ASRT(ret)
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
    ASSERT_EQ(RSMI_STATUS_SUCCESS, ret);

    // Verify api support checking functionality is working
    uint32_t kLength = 2;
    char smallBuffer[kLength];
    err = rsmi_dev_compute_partition_get(dv_ind, smallBuffer, kLength);
    size_t size = sizeof(smallBuffer)/sizeof(*smallBuffer);
    ASSERT_EQ(err, RSMI_STATUS_INSUFFICIENT_SIZE);
    ASSERT_EQ((size_t)kLength, size);
    IF_VERB(STANDARD) {
      if (err == RSMI_STATUS_INSUFFICIENT_SIZE) {
        std::cout << "\t**"
                  << "Confirmed RSMI_STATUS_INSUFFICIENT_SIZE was returned"
                  << "\n\t  and size matches length requested." << std::endl;
      }
    }

    // Verify api support checking functionality is working
    err = rsmi_dev_compute_partition_get(dv_ind, nullptr, 255);
    ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
    IF_VERB(STANDARD) {
      if (err == RSMI_STATUS_INVALID_ARGS) {
        std::cout << "\t**"
                  << "Confirmed RSMI_STATUS_INVALID_ARGS was returned."
                  << std::endl;
      }
    }

    // Verify api support checking functionality is working
    err = rsmi_dev_compute_partition_get(dv_ind, orig_char_computePartition, 0);
    ASSERT_TRUE((err == RSMI_STATUS_INVALID_ARGS) ||
                (err == RSMI_STATUS_NOT_SUPPORTED));
    IF_VERB(STANDARD) {
      if (err == RSMI_STATUS_INVALID_ARGS) {
        std::cout << "\t**"
                  << "Confirmed RSMI_STATUS_INVALID_ARGS was returned."
                  << std::endl;
      }
    }

    // Verify api support checking functionality is working
    rsmi_compute_partition_type_t breakMe;
    err = rsmi_dev_compute_partition_set(dv_ind, breakMe);
    std::cout << "\t**rsmi_dev_compute_partition_set(null ptr): "
              << amd::smi::getRSMIStatusString(err, false) << "\n";
    ASSERT_TRUE((err == RSMI_STATUS_INVALID_ARGS) ||
                (err == RSMI_STATUS_NOT_SUPPORTED) ||
                (err == RSMI_STATUS_PERMISSION));
    IF_VERB(STANDARD) {
      if (err == RSMI_STATUS_INVALID_ARGS) {
        std::cout << "\t**"
                  << "Confirmed RSMI_STATUS_INVALID_ARGS was returned."
                  << std::endl;
      } else if (err == RSMI_STATUS_PERMISSION) {
        DISPLAY_RSMI_ERR(err)
        // tests should not continue if err is a permission issue
        ASSERT_FALSE(err == RSMI_STATUS_PERMISSION);
      } else {
        DISPLAY_RSMI_ERR(err)
      }
    }

    // Re-run original get, so we can reset to later
    ret = rsmi_dev_compute_partition_get(dv_ind, orig_char_computePartition,
                                          255);
    ASSERT_EQ(RSMI_STATUS_SUCCESS, ret);

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

    for (int partition = static_cast<int>(RSMI_COMPUTE_PARTITION_CPX);
         partition <= static_cast<int>(RSMI_COMPUTE_PARTITION_QPX);
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
                  << "rsmi_dev_compute_partition_set(dv_ind, updatePartition): "
                  << amd::smi::getRSMIStatusString(ret, false) << "\n"
                  << "\t**New Partition (set): "
                  << computePartitionString(updatePartition) << "\n";
      }
      ASSERT_TRUE((ret == RSMI_STATUS_SETTING_UNAVAILABLE)
                  || (ret== RSMI_STATUS_PERMISSION)
                  || (ret == RSMI_STATUS_SUCCESS)
                  || ret == RSMI_STATUS_BUSY);

      if (ret == RSMI_STATUS_BUSY) {
        IF_VERB(STANDARD) {
          std::cout << "\t**Device is currently busy.. continue\n";
        }
        system_wait(5);
        continue;
      }

      bool isSettingUnavailable = false;
      if (ret == RSMI_STATUS_SETTING_UNAVAILABLE) {
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
        ASSERT_EQ(RSMI_STATUS_SETTING_UNAVAILABLE, ret);
        ASSERT_STRNE(computePartitionString(updatePartition).c_str(),
                     current_char_computePartition);
        IF_VERB(STANDARD) {
          std::cout << "\t**"
                    << "Confirmed after receiving "
                    << "RSMI_STATUS_SETTING_UNAVAILABLE,\n\t  current compute "
                    << "partition (" << current_char_computePartition
                    << ") did not update to ("
                    << computePartitionString(updatePartition) << ")"
                    << std::endl;
        }
      } else {
        if (strcmp(orig_char_computePartition, current_char_computePartition) !=
        0) {
          devicePartitionUpdated = true;
        } else {
          devicePartitionUpdated = false;
        }

        ASSERT_EQ(RSMI_STATUS_SUCCESS, ret);
        ASSERT_STREQ(computePartitionString(updatePartition).c_str(),
                     current_char_computePartition);
        IF_VERB(STANDARD) {
          std::cout << "\t**"
                    << "Confirmed current compute partition ("
                    << current_char_computePartition << ") matches"
                    << "\n\t  requested compute partition ("
                    << computePartitionString(updatePartition) << ")"
                    << std::endl;
        }
      }
    }   // END looping through partition changes

     /* TEST RETURN TO BOOT COMPUTE PARTITION SETTING */
    IF_VERB(STANDARD) {
      std::cout << std::endl;
      std::cout << "\t**"
                << "=========== TEST RETURN TO BOOT COMPUTE PARTITION SETTING "
                << "========" << std::endl;
    }
    std::string oldPartition = current_char_computePartition;
    bool wasResetSuccess = false;
    ret = rsmi_dev_compute_partition_reset(dv_ind);
    IF_VERB(STANDARD) {
      std::cout << "\t**"
                << "rsmi_dev_compute_partition_reset(dv_ind): "
                << amd::smi::getRSMIStatusString(ret, false) << "\n";
    }
    ASSERT_TRUE((ret == RSMI_STATUS_SUCCESS) ||
                (ret == RSMI_STATUS_NOT_SUPPORTED) ||
                (ret == RSMI_STATUS_BUSY));
    if (ret == RSMI_STATUS_SUCCESS) {
      wasResetSuccess = true;
    }
    ret = rsmi_dev_compute_partition_get(dv_ind, current_char_computePartition,
                                        255);
    CHK_ERR_ASRT(ret)
    IF_VERB(STANDARD) {
      std::cout << "\t**" << "Current compute partition: "
                << current_char_computePartition << "\n"
                << "\t**" << "Original compute partition: "
                << orig_char_computePartition << "\n"
                << "\t**" << "Reset Successful: "
                << (wasResetSuccess ? "TRUE" : "FALSE") << "\n"
                << "\t**" << "Partitions Updated: "
                << (devicePartitionUpdated ? "TRUE" : "FALSE") << "\n";
    }
    if (wasResetSuccess && devicePartitionUpdated) {
      ASSERT_STRNE(oldPartition.c_str(), current_char_computePartition);
      IF_VERB(STANDARD) {
      std::cout << "\t**"
                << "Confirmed prior partition (" << oldPartition << ") is not "
                << "equal to current\n\t  partition ("
                << current_char_computePartition << ")" << std::endl;
      }
    } else {
      ASSERT_STREQ(oldPartition.c_str(), current_char_computePartition);
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
    CHK_ERR_ASRT(ret)
    IF_VERB(STANDARD) {
      std::cout << "\t**" << "Returning compute partition to: "
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
    ASSERT_EQ(RSMI_STATUS_SUCCESS, ret);
    ASSERT_STREQ(computePartitionString(newPartition).c_str(),
                 current_char_computePartition);
  }  // END looping through devices
}
