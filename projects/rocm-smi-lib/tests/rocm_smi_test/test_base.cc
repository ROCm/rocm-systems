/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
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

#include <cassert>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/test_base.h"
#include "rocm_smi_test/test_common.h"
#include "gtest/gtest.h"

static const int kOutputLineLength = 80;
static const char kLabelDelimiter[] = "####";
static const char kDescriptionLabel[] = "TEST DESCRIPTION";
static const char kTitleLabel[] = "TEST NAME";
static const char kRunLabel[] = "TEST EXECUTION";
static const char kCloseLabel[] = "TEST CLEAN UP";
static const char kResultsLabel[] = "TEST RESULTS";

// This one is used outside this file
const char kSetupLabel[] = "TEST SETUP";

TestBase::TestBase() : setup_failed_(false) {
}
TestBase::~TestBase() = default;

void TestBase::MakeHeaderStr(const char *inStr,
                                   std::string *outStr) const {
  assert(outStr != nullptr);
  assert(inStr != nullptr);
  outStr->clear();
  IF_VERB(STANDARD) {
    *outStr = kLabelDelimiter;
    *outStr += " ";
    *outStr += inStr;
    *outStr += " ";
    *outStr += kLabelDelimiter;
  }
}

void TestBase::SetUp(void) {
  SetUp(0);
}

void TestBase::SetUp(uint64_t init_flags) {
  std::string label;
  rsmi_status_t err;

  IF_VERB(STANDARD) {
    MakeHeaderStr(kSetupLabel, &label);
    printf("\n\t%s\n", label.c_str());
  }

  if (init_flags) {
    err = rsmi_init(init_flags);
  } else {
    err = rsmi_init(init_options());
  }

  if (err != RSMI_STATUS_SUCCESS) {
    setup_failed_ = true;
  }
  ASSERT_EQ(err, RSMI_STATUS_SUCCESS);

  err = rsmi_num_monitor_devices(&num_monitor_devs_);
  if (err != RSMI_STATUS_SUCCESS) {
    setup_failed_ = true;
  }
  ASSERT_EQ(err, RSMI_STATUS_SUCCESS);

  if (num_monitor_devs_ == 0) {
    IF_VERB(STANDARD) {
      std::cout << "No monitor devices found on this machine." << std::endl;
      std::cout << "No ROCm SMI tests can be run." << std::endl;
    }
  }
}

void TestBase::PrintDeviceHeader(uint32_t dv_ind) {
  rsmi_status_t err;
  uint16_t val_ui16;

  IF_VERB(STANDARD) {
    std::cout << "\n";
    std::cout << "\t**Device index: " << dv_ind << std::endl;
  }
  err = rsmi_dev_id_get(dv_ind, &val_ui16);
  CHK_ERR_ASRT(err)
  IF_VERB(STANDARD) {
    std::cout << "\t**Device ID: 0x" << std::hex << val_ui16 << std::endl;
  }
  err = rsmi_dev_revision_get(dv_ind, &val_ui16);
  CHK_ERR_ASRT(err)
  IF_VERB(STANDARD) {
    std::cout << "\t**Dev.Rev.ID: 0x" << std::hex << val_ui16 << std::endl;
  }

  char name[128];
  err = rsmi_dev_name_get(dv_ind, name, 128);
  CHK_ERR_ASRT(err)
  IF_VERB(STANDARD) {
    std::cout << "\t**Device name: " << name << std::endl;
  }
  err = rsmi_dev_vendor_id_get(dv_ind, &val_ui16);
  CHK_ERR_ASRT(err)
  IF_VERB(STANDARD) {
    std::cout << "\t**Device Vendor ID: 0x" << std::hex << val_ui16 <<
                                                                     std::endl;
  }
  err = rsmi_dev_subsystem_id_get(dv_ind, &val_ui16);
  CHK_ERR_ASRT(err)
  IF_VERB(STANDARD) {
    std::cout << "\t**Subsystem ID: 0x" << std::hex << val_ui16 << std::endl;
  }
  err = rsmi_dev_subsystem_vendor_id_get(dv_ind, &val_ui16);
  CHK_ERR_ASRT(err)
  IF_VERB(STANDARD) {
    std::cout << "\t**Subsystem Vendor ID: 0x" << std::hex << val_ui16 <<
                                                                     std::endl;
  }
  std::cout << std::setbase(10);
}
void TestBase::Run(void) {
  std::string label;
  IF_VERB(STANDARD) {
    MakeHeaderStr(kRunLabel, &label);
    printf("\n\t%s\n", label.c_str());
  }
  ASSERT_TRUE(!setup_failed_);
}

void TestBase::Close(void) {
  std::string label;
  IF_VERB(STANDARD) {
    MakeHeaderStr(kCloseLabel, &label);
    printf("\n\t%s\n", label.c_str());
  }
  rsmi_status_t err = rsmi_shut_down();
  ASSERT_EQ(err, RSMI_STATUS_SUCCESS);
}

void TestBase::DisplayResults(void) const {
  std::string label;
  IF_VERB(STANDARD) {
    MakeHeaderStr(kResultsLabel, &label);
    printf("\n\t%s\n", label.c_str());
  }
}

void TestBase::DisplayTestInfo(void) {
  IF_VERB(STANDARD) {
    printf("#########################################"
                                  "######################################\n");

    std::string label;
    MakeHeaderStr(kTitleLabel, &label);
    printf("\n\t%s\n%s\n", label.c_str(), title().c_str());

    if (verbosity() >= VERBOSE_STANDARD) {
      MakeHeaderStr(kDescriptionLabel, &label);
      printf("\n\t%s\n%s\n", label.c_str(), description().c_str());
    }
  }
}

void TestBase::set_description(std::string d) {
  int le = kOutputLineLength - 4;

  description_ = d;
  size_t endlptr;

  for (size_t i = le; i < description_.size(); i += le) {
    endlptr = description_.find_last_of(' ', i);
    description_.replace(endlptr, 1, "\n");
    i = endlptr;
  }
}

