/*
Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include "rdc_modules/rdc_rocr/TestBase.h"

#include <assert.h>
#include <unistd.h>

#include <algorithm>

#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"
#include "rdc_modules/rdc_rocr/base_rocr_utils.h"

namespace amd {
namespace rdc {

static const int kOutputLineLength = 80;
static const char kLabelDelimiter[] = "####";
static const char kDescriptionLabel[] = "TEST DESCRIPTION";
static const char kTitleLabel[] = "TEST NAME";
static const char kSetupLabel[] = "TEST SETUP";
static const char kRunLabel[] = "TEST EXECUTION";
static const char kCloseLabel[] = "TEST CLEAN UP";
static const char kResultsLabel[] = "TEST RESULTS";

TestBase::TestBase(uint32_t gpu_index) : gpu_index_(gpu_index), description_("") { SetUp(); }
TestBase::~TestBase() { Close(); }

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

hsa_status_t TestBase::SetUp(void) {
  hsa_status_t err = HSA_STATUS_SUCCESS;
  std::string label;
  MakeHeaderStr(kSetupLabel, &label);
  RDC_LOG(RDC_DEBUG, label);

  err = InitAndSetupHSA(this);

  return err;
}

void TestBase::Run(void) {
  std::string label;
  MakeHeaderStr(kRunLabel, &label);
  RDC_LOG(RDC_DEBUG, label);
}

void TestBase::Close(void) {
  hsa_status_t err;
  std::string label;
  MakeHeaderStr(kCloseLabel, &label);
  RDC_LOG(RDC_DEBUG, label);

  err = CommonCleanUp(this);
  throw_if_error(err);
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

hsa_status_t TestBase::get_agent_by_gpu_index(uint32_t gpu_index, hsa_agent_t* agent) {
  hsa_status_t err = HSA_STATUS_SUCCESS;
  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(IterateGPUAgents, &gpus);
  throw_if_error(err, "Fail to iterate agents.");
  if (gpu_index >= gpus.size()) {
    throw_if_error(err, "GPU index is too large.");
  }
  *agent = gpus[gpu_index];
  return err;
}

}  // namespace rdc
}  // namespace amd
