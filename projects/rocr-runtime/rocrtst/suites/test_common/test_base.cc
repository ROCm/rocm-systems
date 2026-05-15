/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
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

#include <assert.h>

#include "suites/test_common/test_base.h"
#include "suites/test_common/test_common.h"
#include "common/base_rocr_utils.h"
#include "gtest/gtest.h"

static const int kOutputLineLength = 80;
static const char kLabelDelimiter[] = "####";
static const char kDescriptionLabel[] = "TEST DESCRIPTION";
static const char kTitleLabel[] = "TEST NAME";
static const char kSetupLabel[] = "TEST SETUP";
static const char kRunLabel[] = "TEST EXECUTION";
static const char kCloseLabel[] = "TEST CLEAN UP";
static const char kResultsLabel[] = "TEST RESULTS";


TestBase::TestBase() : description_("") {
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

void TestBase::SetupPrint() {
  std::string label;
  MakeHeaderStr(kSetupLabel, &label);
  printf("\n\t%s\n", label.c_str());
}

void TestBase::SetUp(void) {
  hsa_status_t err;
  SetupPrint();
  err = rocrtst::InitAndSetupHSA(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  return;
}

void TestBase::Run(void) {
  std::string label;
  MakeHeaderStr(kRunLabel, &label);
  printf("\n\t%s\n", label.c_str());
}

void TestBase::ClosePrint() {
  std::string label;
  MakeHeaderStr(kCloseLabel, &label);
  printf("\n\t%s\n", label.c_str());
}

void TestBase::Close(void) {
  hsa_status_t err;
  ClosePrint();
  if (monitor_verbosity() > 0) {
    DumpMonitorInfo();
  }

  err = rocrtst::CommonCleanUp(this);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
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

