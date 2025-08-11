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

#include <iostream>
#include <string>
#include <map>

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/memorypartition_read_write.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi_test/test_common.h"

TestMemoryPartitionReadWrite::TestMemoryPartitionReadWrite() : TestBase() {
  set_title("RSMI Memory Partition Read Test");
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
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

static const std::string
memoryPartitionString(rsmi_memory_partition_type memoryPartitionType) {
  switch (memoryPartitionType) {
    case RSMI_MEMORY_PARTITION_NPS1:
      return "NPS1";
    case RSMI_MEMORY_PARTITION_NPS2:
      return "NPS2";
    case RSMI_MEMORY_PARTITION_NPS4:
      return "NPS4";
    case RSMI_MEMORY_PARTITION_NPS8:
      return "NPS8";
    default:
      return "UNKNOWN";
  }
}

static const std::map<std::string, rsmi_memory_partition_type_t>
mapStringToRSMIMemoryPartitionTypes {
  {"NPS1", RSMI_MEMORY_PARTITION_NPS1},
  {"NPS2", RSMI_MEMORY_PARTITION_NPS2},
  {"NPS4", RSMI_MEMORY_PARTITION_NPS4},
  {"NPS8", RSMI_MEMORY_PARTITION_NPS8}
};

static const std::map<std::string, rsmi_compute_partition_type_t>
mapStringToRSMIComputePartitionTypes {
  {"DPX", RSMI_COMPUTE_PARTITION_DPX},
  {"TPX", RSMI_COMPUTE_PARTITION_TPX},
  {"QPX", RSMI_COMPUTE_PARTITION_QPX},
  {"CPX", RSMI_COMPUTE_PARTITION_CPX},
  {"SPX", RSMI_COMPUTE_PARTITION_SPX}
};

void TestMemoryPartitionReadWrite::Run(void) {
  rsmi_status_t ret, err, ret_set;
  char orig_memory_partition[255];
  char current_memory_partition[255];
  char current_memory_capabilities[255];
  orig_memory_partition[0] = '\0';
  current_memory_partition[0] = '\0';
  current_memory_capabilities[0] = '\0';

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  // initial_num_devices - keep this value static, due to parition changes
  // fluctuating # of devices. We should end up with same # of devices at
  // end of test.
  uint32_t initial_num_devices = num_monitor_devs();
  char orig_char_computePartition[initial_num_devices][255];
  char current_char_computePartition[255];

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    // Run and get compute_partition, so we can reset to later
    ret = rsmi_dev_compute_partition_get(dv_ind, orig_char_computePartition[dv_ind],
      255);
    if(ret == RSMI_STATUS_SETTING_UNAVAILABLE
      || ret== RSMI_STATUS_PERMISSION
      || ret == RSMI_STATUS_BUSY
      || ret == RSMI_STATUS_NOT_SUPPORTED
      || ret == RSMI_STATUS_INVALID_ARGS) {
      IF_VERB(STANDARD) {
        std::cout << "\t**rsmi_dev_compute_partition_get(): "
        << "Not supported on this device"
        << std::endl;
      }
      continue;
    }

    std::cout << "\t**rsmi_dev_compute_partition_get(" << dv_ind
      << ", " << orig_char_computePartition[dv_ind] << ")\n";

    ret = rsmi_dev_compute_partition_get(dv_ind, orig_char_computePartition[dv_ind], 255);
    IF_VERB(STANDARD) {
      std::cout << std::endl << "\t**"
        << "Original compute partition: "
        << orig_char_computePartition[dv_ind] << std::endl;
    }
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    bool wasSetSuccess = false;
    if (dv_ind != 0) {
      IF_VERB(STANDARD) {
        std::cout << std::endl;
      }
    }
    PrintDeviceHeader(dv_ind);

    // Standard checks to see if API is supported, before running full tests
    ret = rsmi_dev_memory_partition_get(dv_ind, orig_memory_partition, 255);
    if (ret == RSMI_STATUS_NOT_SUPPORTED) {
       IF_VERB(STANDARD) {
          std::cout << "\t**" <<  ": "
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

    if ((orig_memory_partition == nullptr) ||
       (orig_memory_partition[0] == '\0')) {
      std::cout << "***System memory partition value is not defined or received"
                  " unexpected data. Skip memory partition test." << std::endl;
      continue;
    }
    ASSERT_TRUE(ret == RSMI_STATUS_SUCCESS);

    // Verify api support checking functionality is working
    uint32_t kLen = 2;
    char smallBuffer[kLen];
    err = rsmi_dev_memory_partition_get(dv_ind, smallBuffer, kLen);
    size_t size = sizeof(smallBuffer)/sizeof(*smallBuffer);
    ASSERT_EQ(err, RSMI_STATUS_INSUFFICIENT_SIZE);
    ASSERT_EQ((size_t)kLen, size);
    if (err == RSMI_STATUS_INSUFFICIENT_SIZE) {
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Confirmed RSMI_STATUS_INSUFFICIENT_SIZE was returned "
                  << "and size matches kLen requested." << std::endl;
      }
    }

    // Verify api support checking functionality is working
    err = rsmi_dev_memory_partition_get(dv_ind, nullptr, 255);
    ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);

    if (err == RSMI_STATUS_INVALID_ARGS) {
      IF_VERB(STANDARD) {
        std::cout << "\t**rsmi_dev_memory_partition_get(dv_ind, nullptr, 255): "
                  << "Confirmed RSMI_STATUS_INVALID_ARGS was returned."
                  << std::endl;
      }
    }

    err = rsmi_dev_memory_partition_capabilities_get(dv_ind, nullptr, 255);
    ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);

    if (err == RSMI_STATUS_INVALID_ARGS) {
      IF_VERB(STANDARD) {
        std::cout << "\t**rsmi_dev_memory_partition_capabilities_get(dv_ind, nullptr, 255): "
                  << "Confirmed RSMI_STATUS_INVALID_ARGS was returned."
                  << std::endl;
      }
    }

    // Verify api support checking functionality is working
    err = rsmi_dev_memory_partition_get(dv_ind, orig_memory_partition, 0);
    ASSERT_TRUE((err == RSMI_STATUS_INVALID_ARGS) ||
                (err == RSMI_STATUS_NOT_SUPPORTED));
    if (err == RSMI_STATUS_INVALID_ARGS) {
      IF_VERB(STANDARD) {
        std::cout << "\t**rsmi_dev_memory_partition_get(dv_ind, orig_memory_partition, 0): "
                  << "Confirmed RSMI_STATUS_INVALID_ARGS was returned."
                  << std::endl;
      }
    }

    err = rsmi_dev_memory_partition_capabilities_get(dv_ind, current_memory_capabilities, 0);
    ASSERT_TRUE((err == RSMI_STATUS_INVALID_ARGS) ||
                (err == RSMI_STATUS_NOT_SUPPORTED));
    if (err == RSMI_STATUS_INVALID_ARGS) {
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "rsmi_dev_memory_partition_capabilities_get(dv_ind, "
                  << "current_memory_capabilities, 0): "
                  << "Confirmed RSMI_STATUS_INVALID_ARGS was returned."
                  << std::endl;
      }
    }

    /******************************/
    /* rsmi_dev_memory_partition_set(...) */
    /******************************/
    // Verify api support checking functionality is working
    rsmi_memory_partition_type new_memory_partition = {};
    err = rsmi_dev_memory_partition_set(dv_ind, new_memory_partition);
    std::cout << "\t**rsmi_dev_memory_partition_set(null ptr): "
              << amd::smi::getRSMIStatusString(err, false) << "\n";
    // Note: new_memory_partition is not set
    ASSERT_TRUE((err == RSMI_STATUS_INVALID_ARGS) ||
                (err == RSMI_STATUS_NOT_SUPPORTED));
    if (err == RSMI_STATUS_INVALID_ARGS) {
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Confirmed RSMI_STATUS_INVALID_ARGS was returned."
                  << std::endl;
      }
    } else if (err == RSMI_STATUS_NOT_SUPPORTED) {
         IF_VERB(STANDARD) {
          std::cout << "\t**" <<  ": "
                    << "rsmi_dev_memory_partition_set not supported on this "
                    << "device\n\t    (if rsmi_dev_memory_partition_get works, "
                    << "then likely need to set in bios)"
                    << std::endl;
        }
        continue;
    } else {
        DISPLAY_RSMI_ERR(err)
    }
    ASSERT_FALSE(err == RSMI_STATUS_PERMISSION);

    // Verify api support checking functionality is working
    new_memory_partition =
                      rsmi_memory_partition_type::RSMI_MEMORY_PARTITION_UNKNOWN;
    err = rsmi_dev_memory_partition_set(dv_ind, new_memory_partition);
    ASSERT_TRUE((err == RSMI_STATUS_INVALID_ARGS) ||
                (err == RSMI_STATUS_NOT_SUPPORTED) ||
                (err == RSMI_STATUS_PERMISSION));
    if (err == RSMI_STATUS_INVALID_ARGS) {
      IF_VERB(STANDARD) {
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
    ret = rsmi_dev_memory_partition_get(dv_ind, orig_memory_partition, 255);
    ASSERT_EQ(RSMI_STATUS_SUCCESS, ret);

    for (int partition = RSMI_MEMORY_PARTITION_NPS1;
         partition <= RSMI_MEMORY_PARTITION_NPS8;
         partition++) {
      ret_set = RSMI_STATUS_NOT_SUPPORTED;
      wasSetSuccess = false;
      new_memory_partition = static_cast<rsmi_memory_partition_type>(partition);
      IF_VERB(STANDARD) {
        std::cout << std::endl;
        std::cout << "\t**"
                  << "======== TEST RSMI_MEMORY_PARTITION_"
                  << memoryPartitionString(new_memory_partition)
                  << " ===============" << std::endl;
      }
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Attempting to set memory partition to: "
                  << memoryPartitionString(new_memory_partition) << std::endl;
      }

      rsmi_status_t ret_caps = rsmi_dev_memory_partition_capabilities_get(dv_ind,
                                    current_memory_capabilities, 255);
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "rsmi_dev_memory_partition_capabilities_get(" << dv_ind
                  << ", current_memory_capabilities, 255): "
                  << amd::smi::getRSMIStatusString(ret_caps, false) << std::endl;
        std::cout << "\t**"
                  << "current_memory_capabilities: " << current_memory_capabilities
                  << std::endl;
      }
      ASSERT_TRUE((ret_caps == RSMI_STATUS_NOT_SUPPORTED) ||
                  (ret_caps == RSMI_STATUS_SUCCESS));

      ret_set = rsmi_dev_memory_partition_set(dv_ind, new_memory_partition);
      IF_VERB(STANDARD) {
        std::cout << "\t**" <<  "rsmi_dev_memory_partition_set("
                  << dv_ind << " , " << memoryPartitionString(new_memory_partition) << "): "
                  << amd::smi::getRSMIStatusString(ret_set, false) << "\n";
      }
      if (ret_set == RSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**" <<  ": "
                    << "Not supported on this machine" << std::endl;
        }
        break;
      } else {
        ASSERT_TRUE((ret_set == RSMI_STATUS_SUCCESS)
                  || (ret_set == RSMI_STATUS_BUSY)
                  || (ret_set == RSMI_STATUS_AMDGPU_RESTART_ERR)
                  || (ret_set == RSMI_STATUS_INVALID_ARGS)
                  || (ret_set == RSMI_STATUS_NOT_SUPPORTED));
      }
      IF_VERB(STANDARD) {
        std::cout << "\t**" <<  "rsmi_dev_memory_partition_set("
                  << dv_ind << " , " << memoryPartitionString(new_memory_partition) << "): "
                  << amd::smi::getRSMIStatusString(ret_set, false) << "\n";
      }
      if (ret_set == RSMI_STATUS_SUCCESS) {  // do not continue trying to reset
        wasSetSuccess = true;
      }

      ret = rsmi_dev_memory_partition_get(dv_ind, current_memory_partition,
                                          255);
      CHK_ERR_ASRT(ret)
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Current memory partition: " << current_memory_partition
                  << std::endl;
      }
      if (wasSetSuccess) {
        ASSERT_EQ(RSMI_STATUS_SUCCESS, ret_set);
        ASSERT_STREQ(memoryPartitionString(new_memory_partition).c_str(),
                   current_memory_partition);
        CHK_ERR_ASRT(ret_set)
      } else {
        ASSERT_NE(RSMI_STATUS_SUCCESS, ret_set);
        ASSERT_STRNE(memoryPartitionString(new_memory_partition).c_str(),
                   current_memory_partition);
      }
    }

    /* TEST RETURN TO ORIGINAL MEMORY PARTITION SETTING */
    IF_VERB(STANDARD) {
      std::cout << std::endl;
      std::cout << "\t**"
                << "=========== TEST RETURN TO ORIGINAL MEMORY PARTITION "
                << "SETTING ( " << orig_memory_partition
                << " ) ========" << std::endl;
    }
    std::string oldMode = current_memory_partition;

    ret = rsmi_dev_memory_partition_get(dv_ind, current_memory_partition, 255);
    CHK_ERR_ASRT(ret)
    IF_VERB(STANDARD) {
      std::cout << "\t**"
                << "Current memory partition: " << current_memory_partition
                << std::endl;
    }

    new_memory_partition
      = mapStringToRSMIMemoryPartitionTypes.at(orig_memory_partition);
    IF_VERB(STANDARD) {
      std::cout << "\t**" << "Returning memory partition to: "
                << memoryPartitionString(new_memory_partition) << std::endl;
    }
    ret = rsmi_dev_memory_partition_set(dv_ind, new_memory_partition);
    IF_VERB(STANDARD) {
      std::cout << "\t**"
                << "rsmi_dev_memory_partition_set(" << dv_ind
                << ", " << orig_memory_partition << "): "
                << amd::smi::getRSMIStatusString(ret, false) << std::endl;
    }
    CHK_ERR_ASRT(ret)
    ret = rsmi_dev_memory_partition_get(dv_ind, current_memory_partition, 255);
    CHK_ERR_ASRT(ret)
    IF_VERB(STANDARD) {
      std::cout << "\t**" << "Attempted to set memory partition: "
                << memoryPartitionString(new_memory_partition) << std::endl
                << "\t**" << "Current memory partition: "
                << current_memory_partition
                << std::endl;
    }
    ASSERT_EQ(RSMI_STATUS_SUCCESS, ret);
    ASSERT_STREQ(orig_memory_partition, current_memory_partition);
    IF_VERB(STANDARD) {
      std::cout << "\t**"
                << "Confirmed prior memory partition (" << orig_memory_partition
                << ") is  equal to current memory partition ("
                << current_memory_partition << ")" << std::endl;
    }
  }

  // Return to original compute_partition
  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    ret = rsmi_dev_compute_partition_get(dv_ind, current_char_computePartition, 255);
    if(ret == RSMI_STATUS_SETTING_UNAVAILABLE
      || ret== RSMI_STATUS_PERMISSION
      || ret == RSMI_STATUS_BUSY
      || ret == RSMI_STATUS_NOT_SUPPORTED
      || ret == RSMI_STATUS_INVALID_ARGS)
      continue;

    if (strcmp(orig_char_computePartition[dv_ind], current_char_computePartition) != 0) {
      rsmi_compute_partition_type_t newPartition
        = mapStringToRSMIComputePartitionTypes.at(
                                      std::string(orig_char_computePartition[dv_ind]));
      ret = rsmi_dev_compute_partition_set(dv_ind, newPartition);

      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Returning compute partition to: "
                  << std::string(orig_char_computePartition[dv_ind]) << std::endl;
      }
    }
  }
}
