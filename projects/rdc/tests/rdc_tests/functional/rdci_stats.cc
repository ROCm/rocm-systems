/*
Copyright (c) 2020 - Advanced Micro Devices, Inc. All rights reserved.

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

#include "rdc_tests/functional/rdci_stats.h"

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>

#include <iostream>

#include "rdc/rdc.h"
#include "rdc_tests/test_common.h"

TestRdciStats::TestRdciStats() : TestBase() {
  set_title("\tRDC Stats Test");
  set_description(
      "\tThe Stats tests collects and verifies job "
      "statistics running on gpu groups.");
}

TestRdciStats::~TestRdciStats(void) {}

void TestRdciStats::SetUp(void) {
  TestBase::SetUp();
  rdc_status_t result = AllocateRDCChannel();
  ASSERT_EQ(result, RDC_ST_OK);
  return;
}

void TestRdciStats::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestRdciStats::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestRdciStats::Close() {
  TestBase::Close();
  rdc_status_t result;
  if (standalone_) {
    IF_VERB(STANDARD) { std::cout << "\t**Disconnecting from host....\n" << std::endl; }
    result = rdc_disconnect(rdc_handle);
    ASSERT_EQ(result, RDC_ST_OK);
  } else {
    IF_VERB(STANDARD) { std::cout << "\t**Stopping Embedded RDC Engine....\n" << std::endl; }
    result = rdc_stop_embedded(rdc_handle);
    ASSERT_EQ(result, RDC_ST_OK);
  }

  result = rdc_shutdown();
  ASSERT_EQ(result, RDC_ST_OK);
}

void TestRdciStats::Run(void) {
  TestBase::Run();
  rdc_status_t result;
  if (standalone_) {
    IF_VERB(STANDARD) { std::cout << "\t**Connecting to host....\n" << std::endl; }
    char hostIpAddress[] = {"localhost:50051"};
    result = rdc_connect(hostIpAddress, &rdc_handle, nullptr, nullptr, nullptr);
    ASSERT_EQ(result, RDC_ST_OK);
  } else {
    IF_VERB(STANDARD) { std::cout << "\t**Starting embedded RDC engine....\n" << std::endl; }
    result = rdc_start_embedded(RDC_OPERATION_MODE_AUTO, &rdc_handle);
    ASSERT_EQ(result, RDC_ST_OK);
  }

  rdc_gpu_group_t group_id;
  result = rdc_group_gpu_create(rdc_handle, RDC_GROUP_EMPTY, "GRP_NAME", &group_id);
  ASSERT_EQ(result, RDC_ST_OK);

  result = rdc_group_gpu_add(rdc_handle, group_id, 0);
  ASSERT_EQ(result, RDC_ST_OK);

  result = rdc_job_start_stats(rdc_handle, group_id, "0", 1000000);
  if (result == RDC_ST_NOT_SUPPORTED) {
    std::cout << "\t** GPU Metric is not supported"
                 " on this machine"
              << std::endl;
    return;
  }
  ASSERT_EQ(result, RDC_ST_OK);

  result = rdc_job_stop_stats(rdc_handle, "0");
  ASSERT_EQ(result, RDC_ST_OK);

  rdc_job_info_t job_info;
  result = rdc_job_get_stats(rdc_handle, "0", &job_info);
  if (result == RDC_ST_NOT_SUPPORTED) {
    std::cout << "\t** GPU Metric is not supported"
                 " on this machine"
              << std::endl;
    return;
  }
  ASSERT_EQ(result, RDC_ST_OK);

  result = rdc_job_remove(rdc_handle, "0");
  ASSERT_EQ(result, RDC_ST_OK);

  result = rdc_job_remove_all(rdc_handle);
  ASSERT_EQ(result, RDC_ST_OK);
}
