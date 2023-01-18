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

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/computepartition_read_write.h"
#include "rocm_smi_test/test_common.h"

TestComputePartitionReadWrite::TestComputePartitionReadWrite() : TestBase() {
  set_title("RSMI Compute Partition Read/Write Test");
  set_description("The Compute Parition tests verifies that the compute "
                  "parition can be read and updated properly.");
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
  rsmi_compute_partition_type new_computePartition;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(dv_ind);

    //Standard checks to see if API is supported, before running full tests
    ret = rsmi_dev_compute_partition_get(dv_ind, orig_char_computePartition,
                                          255);
    if (ret == RSMI_STATUS_NOT_SUPPORTED) {
       IF_VERB(STANDARD) {
          std::cout << "\t**" <<  ": "
                    << "Not supported on this machine" << std::endl;
        }
        return;
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
      return;
    }
    EXPECT_EQ(RSMI_STATUS_SUCCESS, ret);

    // Verify api support checking functionality is working
    uint32_t length = 2;
    char smallBuffer[length];
    err = rsmi_dev_compute_partition_get(dv_ind, smallBuffer, length);
    size_t size = sizeof(smallBuffer)/sizeof(*smallBuffer);
    ASSERT_EQ(err, RSMI_STATUS_INSUFFICIENT_SIZE);
    ASSERT_EQ((size_t)length, size);
    IF_VERB(STANDARD) {
      if (err == RSMI_STATUS_INSUFFICIENT_SIZE) {
        std::cout << "\t**"
                  << "Confirmed RSMI_STATUS_INSUFFICIENT_SIZE was returned "
                  << "and size matches length requested." << std::endl;
      }
    }

    // Verify api support checking functionality is working
    err = rsmi_dev_compute_partition_get(dv_ind, nullptr, 255);
    ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);
    IF_VERB(STANDARD) {
      if (err == RSMI_STATUS_NOT_SUPPORTED) {
        std::cout << "\t**"
                  << "Confirmed RSMI_STATUS_NOT_SUPPORTED was returned."
                  << std::endl;
      }
    }

    // Verify api support checking functionality is working
    err = rsmi_dev_compute_partition_get(dv_ind, orig_char_computePartition, 0);
    ASSERT_EQ(err, (RSMI_STATUS_INVALID_ARGS || RSMI_STATUS_NOT_SUPPORTED));
    IF_VERB(STANDARD) {
      if (err == RSMI_STATUS_INVALID_ARGS) {
        std::cout << "\t**"
                  << "Confirmed RSMI_STATUS_INVALID_ARGS was returned."
                  << std::endl;
      }
    }

    // Verify api support checking functionality is working
    err = rsmi_dev_compute_partition_set(dv_ind, new_computePartition);
    // Note: new_computePartition is not set
    // DISPLAY_RSMI_ERR(err)
    EXPECT_TRUE((err == RSMI_STATUS_INVALID_ARGS) ||
                (err == RSMI_STATUS_NOT_SUPPORTED));
    IF_VERB(STANDARD) {
      if (err == RSMI_STATUS_INVALID_ARGS) {
        std::cout << "\t**"
                  << "Confirmed RSMI_STATUS_INVALID_ARGS was returned."
                  << std::endl;
      } else {
        DISPLAY_RSMI_ERR(err)
      }
    }
    ASSERT_FALSE(err == RSMI_STATUS_PERMISSION);

    // Verify api support checking functionality is working
    new_computePartition
      = rsmi_compute_partition_type::RSMI_COMPUTE_PARTITION_INVALID;
    err = rsmi_dev_compute_partition_set(dv_ind, new_computePartition);
    // DISPLAY_RSMI_ERR(err)
    EXPECT_TRUE((err == RSMI_STATUS_INVALID_ARGS) ||
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
    EXPECT_EQ(RSMI_STATUS_SUCCESS, ret);

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

    for (int partition = RSMI_COMPUTE_PARTITION_CPX;
         partition <= RSMI_COMPUTE_PARTITION_QPX;
         partition++) {
      new_computePartition
        = static_cast<rsmi_compute_partition_type>(partition);
      IF_VERB(STANDARD) {
        std::cout << std::endl;
        std::cout << "\t**"
                  << "======== TEST RSMI_COMPUTE_PARTITION_"
                  << computePartitionString(new_computePartition)
                  << " ===============" << std::endl;
      }
      ret = rsmi_dev_compute_partition_set(dv_ind, new_computePartition);
      CHK_ERR_ASRT(ret)
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Attempting to set compute partition to: "
                  << computePartitionString(new_computePartition) << std::endl;
      }
      ret = rsmi_dev_compute_partition_get(dv_ind, current_char_computePartition,
                                          255);
      CHK_ERR_ASRT(ret)
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Current compute partition: "
                  << current_char_computePartition
                  << std::endl;
      }
      EXPECT_EQ(RSMI_STATUS_SUCCESS, ret);
      EXPECT_STREQ(computePartitionString(new_computePartition).c_str(),
                   current_char_computePartition);
    }

    /* TEST RETURN TO ORIGINAL COMPUTE PARTITIONING SETTING */
    IF_VERB(STANDARD) {
      std::cout << std::endl;
      std::cout << "\t**"
                << "=========== TEST RETURN TO ORIGINAL COMPUTE PARTITIONING "
                << "SETTING ========" << std::endl;
    }
    new_computePartition
      = mapStringToRSMIComputePartitionTypes.at(orig_char_computePartition);
    ret = rsmi_dev_compute_partition_set(dv_ind, new_computePartition);
    CHK_ERR_ASRT(ret)
    IF_VERB(STANDARD) {
      std::cout << "\t**" << "Returning compute partition to: "
                << computePartitionString(new_computePartition) << std::endl;
    }
    ret = rsmi_dev_compute_partition_get(dv_ind, current_char_computePartition,
                                          255);
    CHK_ERR_ASRT(ret)
    IF_VERB(STANDARD) {
      std::cout << "\t**" << "Attempted to set compute partition: "
                << computePartitionString(new_computePartition) << std::endl
                << "\t**"
                << "Current compute partition: " << current_char_computePartition
                << std::endl;
    }
    EXPECT_EQ(RSMI_STATUS_SUCCESS, ret);
    EXPECT_STREQ(computePartitionString(new_computePartition).c_str(),
                 current_char_computePartition);
  }
}
