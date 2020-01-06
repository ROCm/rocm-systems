/*
Copyright (c) 2019 - Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <assert.h>

#include "rocm_smi/rocm_smi.h"
#include "rdc_tests/test_base.h"
#include "rdc_tests/test_common.h"
#include "gtest/gtest.h"

static const int kOutputLineLength = 80;
static const char kLabelDelimiter[] = "####";
static const char kDescriptionLabel[] = "TEST DESCRIPTION";
static const char kTitleLabel[] = "TEST NAME";
static const char kSetupLabel[] = "TEST SETUP";
static const char kRunLabel[] = "TEST EXECUTION";
static const char kCloseLabel[] = "TEST CLEAN UP";
static const char kResultsLabel[] = "TEST RESULTS";


TestBase::TestBase() : description_(""), rdc_channel_(0) {
}
TestBase::~TestBase() {
}

static void MakeHeaderStr(const char *inStr, std::string *outStr) {
  assert(outStr != nullptr);
  assert(inStr != nullptr);

  outStr->clear();
  *outStr = kLabelDelimiter;
  *outStr += " ";
  *outStr += inStr;
  *outStr += " ";
  *outStr += kLabelDelimiter;
}

void TestBase::SetUp(void) {
  std::string label;

  MakeHeaderStr(kSetupLabel, &label);
  printf("\n\t%s\n", label.c_str());

  return;
}

void TestBase::PrintDeviceHeader(uint32_t dv_ind) {
  IF_VERB(STANDARD) {
    std::cout << "\t**Device index: " << dv_ind << std::endl;
  }

  // TODO(cfreehil): replace these sections with RDC equiv. as they are
  // implemented
//  err = rsmi_dev_id_get(dv_ind, &val_ui16);
//  CHK_ERR_ASRT(err)
//  IF_VERB(STANDARD) {
//    std::cout << "\t**Device ID: 0x" << std::hex << val_ui16 << std::endl;
//  }
//  char name[128];
//  err = rsmi_dev_name_get(dv_ind, name, 128);
//  CHK_ERR_ASRT(err)
//  IF_VERB(STANDARD) {
//    std::cout << "\t**Device name: " << name << std::endl;
//  }
//  err = rsmi_dev_vendor_id_get(dv_ind, &val_ui16);
//  CHK_ERR_ASRT(err)
//  IF_VERB(STANDARD) {
//    std::cout << "\t**Device Vendor ID: 0x" << std::hex << val_ui16 <<
//                                                                     std::endl;
//  }
//  err = rsmi_dev_subsystem_id_get(dv_ind, &val_ui16);
//  CHK_ERR_ASRT(err)
//  IF_VERB(STANDARD) {
//    std::cout << "\t**Subsystem ID: 0x" << std::hex << val_ui16 << std::endl;
//  }
//  err = rsmi_dev_subsystem_vendor_id_get(dv_ind, &val_ui16);
//  CHK_ERR_ASRT(err)
//  IF_VERB(STANDARD) {
//    std::cout << "\t**Subsystem Vendor ID: 0x" << std::hex << val_ui16 <<
//                                                                     std::endl;
//  }
  std::cout << std::setbase(10);
}

rdc_status_t
TestBase::AllocateRDCChannel(void) {
  rdc_channel_t c;
  rdc_status_t err;

  IF_VERB(STANDARD) {
    std::cout << "\t**Creating channel to host..." << std::endl;
  }

  assert(rdc_channel() == 0);

  err = rdc_channel_create(&c,
      monitor_server_ip() == "" ? nullptr : monitor_server_ip().c_str(),
      monitor_server_port() == "" ? nullptr : monitor_server_port().c_str(),
                                                                        false);

  if (err != RDC_STATUS_SUCCESS) {
    std::cout << "rdc_channel_create() failed" << std::endl;
    return err;
  }

  grpc_connectivity_state grpc_state;

  IF_VERB(STANDARD) {
    std::cout << "\t**Channel created. Getting channel state..." << std::endl;
  }
  err = rdc_channel_state_get(c, true, &grpc_state);
  if (err != RDC_STATUS_SUCCESS) {
    std::cout << "rdc_channel_state_get() failed" << std::endl;
    return err;
  }

  IF_VERB(STANDARD) {
    std::cout << "\t**Channel state: \"" << GetGRPCChanStateStr(grpc_state) <<
                                                             "\"" << std::endl;
    std::cout << "\tVerifying channel connection..." << std::endl;
  }

  err = rdc_channel_connection_verify(c);
  if (err != RDC_STATUS_SUCCESS) {
    std::cout << "rdc_channel_connection_verify() failed" << std::endl;
    return err;
  }

  IF_VERB(STANDARD) {
    std::cout << "\tConnection verified!" << std::endl;
  }
  set_rdc_channel(c);

  err = rdc_num_gpus_get(rdc_channel(), &num_monitor_devs_);
  if (err != RDC_STATUS_SUCCESS) {
    std::cout << "rdc_num_gpus_get() failed" << std::endl;
    return err;
  }

  IF_VERB(STANDARD) {
    std::cout << "\t" << num_monitor_devs() << " devices found on host" <<
                                                                    std::endl;
  }

  return RDC_STATUS_SUCCESS;
}

void TestBase::Run(void) {
  std::string label;

  MakeHeaderStr(kRunLabel, &label);
  printf("\n\t%s\n", label.c_str());
}

void TestBase::Close(void) {
  std::string label;
  rdc_status_t err;

  MakeHeaderStr(kCloseLabel, &label);
  printf("\n\t%s\n", label.c_str());

  err = rdc_channel_destroy(rdc_channel());
  ASSERT_EQ(err, RDC_STATUS_SUCCESS);
  set_rdc_channel(0);
}

void TestBase::DisplayResults(void) const {
  std::string label;
  MakeHeaderStr(kResultsLabel, &label);
  printf("\n\t%s\n", label.c_str());
}

void TestBase::DisplayTestInfo(void) {
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

void TestBase::set_description(std::string d) {
  int le = kOutputLineLength - 4;

  description_ = d;
  size_t endlptr;

  for (size_t i = le; i < description_.size(); i += le) {
    endlptr = description_.find_last_of(" ", i);
    description_.replace(endlptr, 1, "\n");
    i = endlptr;
  }
}

