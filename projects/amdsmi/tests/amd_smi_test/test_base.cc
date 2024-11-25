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

#include <cassert>

#include "amd_smi/amdsmi.h"
#include "test_base.h"
#include "test_common.h"
#include <gtest/gtest.h>

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
  SetUp(AMDSMI_INIT_AMD_GPUS);
}

void TestBase::SetUp(uint64_t init_flags) {
  std::string label;
  amdsmi_status_t err;

  IF_VERB(STANDARD) {
    MakeHeaderStr(kSetupLabel, &label);
    printf("\n\t%s\n", label.c_str());
  }

  if (init_flags) {
    err = amdsmi_init(init_flags);
  } else {
    err = amdsmi_init(init_options());
  }

  if (err != AMDSMI_STATUS_SUCCESS) {
    setup_failed_ = true;
  }
  ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);


  err = amdsmi_get_socket_handles(&socket_count_, nullptr);
  if (err != AMDSMI_STATUS_SUCCESS) {
    setup_failed_ = true;
  }
  ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);

  // allocate memory
  sockets_.resize(socket_count_);
  err = amdsmi_get_socket_handles(&socket_count_, &sockets_[0]);
  if (err != AMDSMI_STATUS_SUCCESS) {
    setup_failed_ = true;
  }
  ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);

  // collect devices from sockets
  num_monitor_devs_ = 0;
  for (uint32_t i=0; i < socket_count_; i++) {
    // Get all devices of the socket
    uint32_t device_count = 0;
    err = amdsmi_get_processor_handles(sockets_[i],
            &device_count, nullptr);
    if (err != AMDSMI_STATUS_SUCCESS) {
      setup_failed_ = true;
    }
    ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);

    std::vector<amdsmi_processor_handle> processor_handles(device_count);
    err = amdsmi_get_processor_handles(sockets_[i],
            &device_count, &processor_handles[0]);
    if (err != AMDSMI_STATUS_SUCCESS) {
      setup_failed_ = true;
    }
    ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);
    // store the device handles for following tests
    for (uint32_t j=0; j < device_count; j++) {
      if (num_monitor_devs_ >= MAX_MONITOR_DEVICES) {
        setup_failed_ = true;
        ASSERT_EQ(AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS, AMDSMI_STATUS_SUCCESS);
      }
      processor_handles_[num_monitor_devs_] = processor_handles[j];
      num_monitor_devs_++;
    }
  }

  if (num_monitor_devs_ == 0) {
    IF_VERB(STANDARD) {
      std::cout << "No monitor devices found on this machine." << std::endl;
      std::cout << "No AMD SMI tests can be run." << std::endl;
    }
  }
}

void TestBase::PrintDeviceHeader(amdsmi_processor_handle dv_ind) {
  amdsmi_status_t err;
  uint16_t val_ui16;
  amdsmi_asic_info_t info;

  IF_VERB(STANDARD) {
    std::cout << "\t**Device handle: " << dv_ind << std::endl;
  }
  err = amdsmi_get_gpu_id(dv_ind, &val_ui16);
  CHK_ERR_ASRT(err)
  IF_VERB(STANDARD) {
    std::cout << "\t**Device ID: 0x" << std::hex << val_ui16 << std::endl;
  }

  err = amdsmi_get_gpu_revision(dv_ind, &val_ui16);
  CHK_ERR_ASRT(err)
  IF_VERB(STANDARD) {
      std::cout << "\t**Device Revision ID: 0x" << std::hex <<
          val_ui16 << std::endl;
  }

  amdsmi_board_info_t board_info;
  err = amdsmi_get_gpu_board_info(dv_ind, &board_info);
  CHK_ERR_ASRT(err)
  IF_VERB(STANDARD) {
    std::cout << "\t**Device name: " << board_info.product_name  << std::endl;

    err = amdsmi_get_gpu_asic_info(dv_ind, &info);
    CHK_ERR_ASRT(err)
    IF_VERB(STANDARD) {
      std::cout << "\t**Device Vendor ID: 0x" << std::hex <<
          info.vendor_id << std::endl;
    }
  }

  err = amdsmi_get_gpu_subsystem_id(dv_ind, &val_ui16);
  CHK_ERR_ASRT(err)
  IF_VERB(STANDARD) {
    std::cout << "\t**Subsystem ID: 0x" << std::hex << val_ui16 << std::endl;
    std::cout << "\t**Subsystem Vendor ID: 0x" << std::hex
          << info.subvendor_id << std::endl;
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
  amdsmi_status_t err = amdsmi_shut_down();
  ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);
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

