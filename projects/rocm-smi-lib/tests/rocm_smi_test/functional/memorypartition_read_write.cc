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

void TestMemoryPartitionReadWrite::Run(void) {
  rsmi_status_t ret, err;
  char orig_memory_partition[255];
  char current_memory_partition[255];
  orig_memory_partition[0] = '\0';
  current_memory_partition[0] = '\0';

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
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
        std::cout << "\t**"
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
        std::cout << "\t**"
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
      ret = rsmi_dev_memory_partition_set(dv_ind, new_memory_partition);
      if (ret == RSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**" <<  ": "
                    << "Not supported on this machine" << std::endl;
        }
        break;
      } else {
        CHK_ERR_ASRT(ret)
      }
      if (ret != RSMI_STATUS_SUCCESS) {  // do not continue trying to reset
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
      ASSERT_EQ(RSMI_STATUS_SUCCESS, ret);
      ASSERT_STREQ(memoryPartitionString(new_memory_partition).c_str(),
                   current_memory_partition);
    }

    /* TEST RETURN TO BOOT MEMORY PARTITION SETTING */
    IF_VERB(STANDARD) {
      std::cout << std::endl;
      std::cout << "\t**"
                << "=========== TEST RETURN TO BOOT MEMORY PARTITION "
                << "SETTING ========" << std::endl;
    }
    std::string oldMode = current_memory_partition;
    bool wasResetSuccess = false;
    ret = rsmi_dev_memory_partition_reset(dv_ind);
    ASSERT_TRUE((ret == RSMI_STATUS_SUCCESS) ||
                (ret == RSMI_STATUS_NOT_SUPPORTED));
    if (ret == RSMI_STATUS_SUCCESS) {
      wasResetSuccess = true;
    }
    ret = rsmi_dev_memory_partition_get(dv_ind, current_memory_partition, 255);
    CHK_ERR_ASRT(ret)
    IF_VERB(STANDARD) {
      std::cout << "\t**"
                << "Current memory partition: " << current_memory_partition
                << std::endl;
    }
    if (wasResetSuccess && wasSetSuccess) {
      ASSERT_STRNE(oldMode.c_str(), current_memory_partition);
      IF_VERB(STANDARD) {
      std::cout << "\t**"
                << "Confirmed prior memory partition (" << oldMode << ") is "
                << "not equal to current memory partition ("
                << current_memory_partition << ")" << std::endl;
      }
    } else {
      ASSERT_STREQ(oldMode.c_str(), current_memory_partition);
      IF_VERB(STANDARD) {
      std::cout << "\t**"
                << "Confirmed prior memory partition (" << oldMode << ") is "
                << "equal to current memory partition ("
                << current_memory_partition << ")" << std::endl;
      }
    }

    /* TEST RETURN TO ORIGINAL MEMORY PARTITION SETTING */
    IF_VERB(STANDARD) {
      std::cout << std::endl;
      std::cout << "\t**"
                << "=========== TEST RETURN TO ORIGINAL MEMORY PARTITION "
                << "SETTING ========" << std::endl;
    }
    new_memory_partition
      = mapStringToRSMIMemoryPartitionTypes.at(orig_memory_partition);
    IF_VERB(STANDARD) {
      std::cout << "\t**" << "Returning memory partition to: "
                << memoryPartitionString(new_memory_partition) << std::endl;
    }
    ret = rsmi_dev_memory_partition_set(dv_ind, new_memory_partition);
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
    ASSERT_STREQ(memoryPartitionString(new_memory_partition).c_str(), current_memory_partition);
  }
}
