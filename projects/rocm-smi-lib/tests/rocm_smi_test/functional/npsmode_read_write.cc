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
#include "rocm_smi_test/functional/npsmode_read_write.h"
#include "rocm_smi_test/test_common.h"

TestNPSModeReadWrite::TestNPSModeReadWrite() : TestBase() {
  set_title("RSMI NPS Mode Read Test");
  set_description("The NPS Mode tests verifies that the memory "
                  "parition setting can be read and updated properly.");
}

TestNPSModeReadWrite::~TestNPSModeReadWrite(void) {
}

void TestNPSModeReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestNPSModeReadWrite::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestNPSModeReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestNPSModeReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

static const std::string
npsModeString(rsmi_nps_mode_type npsModeType) {
  switch (npsModeType) {
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

static const std::map<std::string, rsmi_nps_mode_type_t>
mapStringToRSMINpsModeTypes {
  {"NPS1", RSMI_MEMORY_PARTITION_NPS1},
  {"NPS2", RSMI_MEMORY_PARTITION_NPS2},
  {"NPS4", RSMI_MEMORY_PARTITION_NPS4},
  {"NPS8", RSMI_MEMORY_PARTITION_NPS8}
};

void TestNPSModeReadWrite::Run(void) {
  rsmi_status_t ret, err;
  char orig_nps_mode[255];
  char current_nps_mode[255];
  orig_nps_mode[0] = '\0';
  current_nps_mode[0] = '\0';

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    if (dv_ind != 0) {
      IF_VERB(STANDARD) {
        std::cout << std::endl;
      }
    }
    PrintDeviceHeader(dv_ind);

    //Standard checks to see if API is supported, before running full tests
    ret = rsmi_dev_nps_mode_get(dv_ind, orig_nps_mode, 255);
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
      std::cout << std::endl << "\t**"
                << "NPS Mode: "
                << orig_nps_mode << std::endl;
    }

    if ((orig_nps_mode == nullptr) ||
       (orig_nps_mode[0] == '\0')) {
      std::cout << "***System nps mode value is not defined or received unexpected data. "
                  "Skip nps mode test." << std::endl;
      continue;
    }
    ASSERT_TRUE(ret == RSMI_STATUS_SUCCESS);

    // Verify api support checking functionality is working
    uint32_t length = 2;
    char smallBuffer[length];
    err = rsmi_dev_nps_mode_get(dv_ind, smallBuffer, length);
    size_t size = sizeof(smallBuffer)/sizeof(*smallBuffer);
    ASSERT_EQ(err, RSMI_STATUS_INSUFFICIENT_SIZE);
    ASSERT_EQ((size_t)length, size);
    if (err == RSMI_STATUS_INSUFFICIENT_SIZE) {
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Confirmed RSMI_STATUS_INSUFFICIENT_SIZE was returned "
                  << "and size matches length requested." << std::endl;
      }
    }

    // Verify api support checking functionality is working
    err = rsmi_dev_nps_mode_get(dv_ind, nullptr, 255);
    ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);

    if (err == RSMI_STATUS_INVALID_ARGS) {
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Confirmed RSMI_STATUS_INVALID_ARGS was returned."
                  << std::endl;
      }
    }

    // Verify api support checking functionality is working
    err = rsmi_dev_nps_mode_get(dv_ind, orig_nps_mode, 0);
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
    /* rsmi_dev_nps_mode_set(...) */
    /******************************/
    // Verify api support checking functionality is working
    rsmi_nps_mode_type new_nps_mode;
    err = rsmi_dev_nps_mode_set(dv_ind, new_nps_mode);
    // Note: new_nps_mode is not set
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
                    << "rsmi_dev_nps_mode_set not supported on this device"
                    << "\n\t    (if rsmi_dev_nps_mode_get works, then likely "
                    << "need to set in bios)"
                    << std::endl;
        }
        continue;
    } else {
        DISPLAY_RSMI_ERR(err)
    }
    ASSERT_FALSE(err == RSMI_STATUS_PERMISSION);

    // Verify api support checking functionality is working
    new_nps_mode = rsmi_nps_mode_type::RSMI_MEMORY_PARTITION_UNKNOWN;
    err = rsmi_dev_nps_mode_set(dv_ind, new_nps_mode);
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
    ret = rsmi_dev_nps_mode_get(dv_ind, orig_nps_mode, 255);
    ASSERT_EQ(RSMI_STATUS_SUCCESS, ret);

    for (int partition = RSMI_MEMORY_PARTITION_NPS1;
         partition <= RSMI_MEMORY_PARTITION_NPS8;
         partition++) {
      new_nps_mode = static_cast<rsmi_nps_mode_type>(partition);
      IF_VERB(STANDARD) {
        std::cout << std::endl;
        std::cout << "\t**"
                  << "======== TEST RSMI_MEMORY_PARTITION_"
                  << npsModeString(new_nps_mode)
                  << " ===============" << std::endl;
      }
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Attempting to set nps mode to: "
                  << npsModeString(new_nps_mode) << std::endl;
      }
      ret = rsmi_dev_nps_mode_set(dv_ind, new_nps_mode);
      CHK_ERR_ASRT(ret)

      ret = rsmi_dev_nps_mode_get(dv_ind, current_nps_mode, 255);
      CHK_ERR_ASRT(ret)
      IF_VERB(STANDARD) {
        std::cout << "\t**"
                  << "Current nps mode: " << current_nps_mode << std::endl;
      }
      ASSERT_EQ(RSMI_STATUS_SUCCESS, ret);
      ASSERT_STREQ(npsModeString(new_nps_mode).c_str(), current_nps_mode);
    }

    /* TEST RETURN TO BOOT NPS MODE SETTING */
    IF_VERB(STANDARD) {
      std::cout << std::endl;
      std::cout << "\t**"
                << "=========== TEST RETURN TO BOOT NPS MODE SETTING "
                << "========" << std::endl;
    }
    std::string oldMode = current_nps_mode;
    bool wasResetSuccess = false;
    ret = rsmi_dev_nps_mode_reset(dv_ind);
    ASSERT_TRUE((ret == RSMI_STATUS_SUCCESS) ||
                (ret == RSMI_STATUS_NOT_SUPPORTED));
    if (ret == RSMI_STATUS_SUCCESS) {
      wasResetSuccess = true;
    }
    ret = rsmi_dev_nps_mode_get(dv_ind, current_nps_mode, 255);
    CHK_ERR_ASRT(ret)
    IF_VERB(STANDARD) {
      std::cout << "\t**"
                << "Current nps mode: " << current_nps_mode << std::endl;
    }
    if (wasResetSuccess) {
      ASSERT_STRNE(oldMode.c_str(), current_nps_mode);
      IF_VERB(STANDARD) {
      std::cout << "\t**"
                << "Confirmed prior nps mode (" << oldMode << ") is not "
                << "equal to current nps mode ("
                << current_nps_mode << ")" << std::endl;
      }
    } else {
      ASSERT_STREQ(oldMode.c_str(), current_nps_mode);
      IF_VERB(STANDARD) {
      std::cout << "\t**"
                << "Confirmed prior nps mode (" << oldMode << ") is equal"
                << " to current nps mode ("
                << current_nps_mode << ")" << std::endl;
      }
    }

    /* TEST RETURN TO ORIGINAL NPS MODE SETTING */
    IF_VERB(STANDARD) {
      std::cout << std::endl;
      std::cout << "\t**"
                << "=========== TEST RETURN TO ORIGINAL NPS MODE "
                << "SETTING ========" << std::endl;
    }
    new_nps_mode
      = mapStringToRSMINpsModeTypes.at(orig_nps_mode);
    IF_VERB(STANDARD) {
      std::cout << "\t**" << "Returning nps mode to: "
                << npsModeString(new_nps_mode) << std::endl;
    }
    ret = rsmi_dev_nps_mode_set(dv_ind, new_nps_mode);
    CHK_ERR_ASRT(ret)
    ret = rsmi_dev_nps_mode_get(dv_ind, current_nps_mode, 255);
    CHK_ERR_ASRT(ret)
    IF_VERB(STANDARD) {
      std::cout << "\t**" << "Attempted to set nps mode: "
                << npsModeString(new_nps_mode) << std::endl
                << "\t**" << "Current nps mode: " << current_nps_mode
                << std::endl;
    }
    ASSERT_EQ(RSMI_STATUS_SUCCESS, ret);
    ASSERT_STREQ(npsModeString(new_nps_mode).c_str(), current_nps_mode);
  }
}
