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

#include "rdc_tests/functional/rdci_dmon.h"

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>

#include <iostream>

#include "rdc/rdc.h"
#include "rdc_tests/test_common.h"

TestRdciDmon::TestRdciDmon() : TestBase() {
  set_title("\tRDC Dmon Test");
  set_description("\tThe Dmon tests verifies that the GPUs metrics are being monitored. ");
}

TestRdciDmon::~TestRdciDmon(void) {}

void TestRdciDmon::SetUp(void) {
  TestBase::SetUp();
  rdc_status_t result = AllocateRDCChannel();
  ASSERT_EQ(result, RDC_ST_OK);
  return;
}

void TestRdciDmon::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestRdciDmon::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestRdciDmon::Close() {
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

void TestRdciDmon::Run(void) {
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

  rdc_group_info_t group_info;
  rdc_gpu_group_t group_id;
  rdc_field_grp_t field_group_id;
  result = rdc_group_gpu_create(rdc_handle, RDC_GROUP_EMPTY, "GRP_DMON", &group_id);
  ASSERT_EQ(result, RDC_ST_OK);

  result = rdc_group_gpu_add(rdc_handle, group_id, 0);
  ASSERT_EQ(result, RDC_ST_OK);

  result = rdc_group_gpu_get_info(rdc_handle, group_id, NULL);
  ASSERT_EQ(result, RDC_ST_INVALID_HANDLER);

  result = rdc_group_gpu_get_info(rdc_handle, -1, NULL);
  ASSERT_EQ(result, RDC_ST_INVALID_HANDLER);

  result = rdc_group_gpu_get_info(rdc_handle, group_id, &group_info);
  ASSERT_EQ(result, RDC_ST_OK);
  ASSERT_GT(group_info.count, 0);

  rdc_field_t field_ids[] = {RDC_FI_GPU_TEMP, RDC_FI_POWER_USAGE, RDC_FI_GPU_UTIL};
  uint32_t fsize = sizeof(field_ids) / sizeof(field_ids[0]);
  result = rdc_group_field_create(rdc_handle, fsize, &field_ids[0], "FIELD_GRP", &field_group_id);
  ASSERT_EQ(result, RDC_ST_OK);

  result = rdc_field_watch(rdc_handle, -1, field_group_id, 0, 60, 10);
  ASSERT_EQ(result, RDC_ST_NOT_FOUND);

  result = rdc_field_watch(rdc_handle, group_id, -1, 0, 60, 10);
  ASSERT_EQ(result, RDC_ST_NOT_FOUND);

  result = rdc_field_watch(rdc_handle, group_id, field_group_id, 0, 60, 10);
  ASSERT_EQ(result, RDC_ST_OK);

  result = rdc_field_unwatch(rdc_handle, group_id, field_group_id);
  ASSERT_EQ(result, RDC_ST_OK);

  result = rdc_group_gpu_destroy(rdc_handle, group_id);
  ASSERT_EQ(result, RDC_ST_OK);

  result = rdc_group_field_destroy(rdc_handle, field_group_id);
  ASSERT_EQ(result, RDC_ST_OK);
}
