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
#include "rdc_tests/test_base.h"

#include <assert.h>
#include <gtest/gtest.h>

#include "amd_smi/amdsmi.h"
#include "rdc_tests/test_common.h"

static const int kOutputLineLength = 80;
static const char kLabelDelimiter[] = "####";
static const char kDescriptionLabel[] = "TEST DESCRIPTION";
static const char kTitleLabel[] = "TEST NAME";
static const char kSetupLabel[] = "TEST SETUP";
static const char kRunLabel[] = "TEST EXECUTION";
static const char kCloseLabel[] = "TEST CLEAN UP";
static const char kResultsLabel[] = "TEST RESULTS";
rdc_status_t result;

/*TestBase::TestBase() : description_(""), rdc_channel_(0) {
}*/
TestBase::TestBase() : description_("") {}
TestBase::~TestBase() {}

static void MakeHeaderStr(const char* inStr, std::string* outStr) {
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
  IF_VERB(STANDARD) { std::cout << "\t**Device index: " << dv_ind << std::endl; }

  std::cout << std::setbase(10);
}

rdc_status_t TestBase::AllocateRDCChannel(void) {
  IF_VERB(STANDARD) { std::cout << "\t**Initializing RDC" << std::endl; }
  rdc_status_t result = rdc_init(0);
  if (result != RDC_ST_OK) {
    std::cout << "Error initializing RDC.... " << rdc_status_string(result) << std::endl;
    return result;
  }

  return result;
}

void TestBase::Run(void) {
  std::string label;

  MakeHeaderStr(kRunLabel, &label);
  printf("\n\t%s\n", label.c_str());
}

void TestBase::Close(void) {
  std::string label;

  MakeHeaderStr(kCloseLabel, &label);
  printf("\n\t%s\n", label.c_str());
}

void TestBase::DisplayResults(void) const {
  std::string label;
  MakeHeaderStr(kResultsLabel, &label);
  printf("\n\t%s\n", label.c_str());
}

void TestBase::DisplayTestInfo(void) {
  printf(
      "#########################################"
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
