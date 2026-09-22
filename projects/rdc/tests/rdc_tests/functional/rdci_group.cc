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

#include "rdc_tests/functional/rdci_group.h"

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>

#include <iostream>

#include "rdc/rdc.h"
#include "rdc_tests/test_common.h"

TestRdciGroup::TestRdciGroup() : TestBase() {
  set_title("\tRDC Group Test");
  set_description("\tThe Group tests verifies creation/deletion of GPU groups");
}

TestRdciGroup::~TestRdciGroup(void) {}

void TestRdciGroup::SetUp(void) {
  TestBase::SetUp();
  rdc_status_t result = AllocateRDCChannel();
  ASSERT_EQ(result, RDC_ST_OK);
  return;
}

void TestRdciGroup::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestRdciGroup::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestRdciGroup::Close() {
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

void TestRdciGroup::Run(void) {
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

  rdc_gpu_group_t group_id = 0;
  uint32_t count = 0;
  rdc_group_info_t group_info;
  uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES];
  result = rdc_group_gpu_create(0, RDC_GROUP_EMPTY, "GRP_NAME", &group_id);
  ASSERT_EQ(result, RDC_ST_INVALID_HANDLER);

  result = rdc_group_gpu_create(rdc_handle, RDC_GROUP_EMPTY, NULL, &group_id);
  ASSERT_EQ(result, RDC_ST_BAD_PARAMETER);

  result = rdc_group_gpu_create(rdc_handle, RDC_GROUP_EMPTY, "GRP_NAME", NULL);
  ASSERT_EQ(result, RDC_ST_BAD_PARAMETER);

  result = rdc_group_gpu_create(rdc_handle, RDC_GROUP_EMPTY, "GRP_NAME", &group_id);
  ASSERT_EQ(result, RDC_ST_OK);

  result = rdc_group_gpu_add(rdc_handle, group_id, -1);
  ASSERT_EQ(result, RDC_ST_NOT_FOUND);

  result = rdc_device_get_all(rdc_handle, gpu_index_list, &count);
  ASSERT_EQ(result, RDC_ST_OK);
  for (uint32_t i = 0; i < count; i++) {
    result = rdc_group_gpu_add(rdc_handle, group_id, gpu_index_list[i]);
    ASSERT_EQ(result, RDC_ST_OK);
  }

  rdc_gpu_group_t group_id_list[RDC_MAX_NUM_GROUPS];
  result = rdc_group_get_all_ids(rdc_handle, group_id_list, NULL);
  ASSERT_EQ(result, RDC_ST_INVALID_HANDLER);

  result = rdc_group_get_all_ids(rdc_handle, group_id_list, &count);
  ASSERT_EQ(result, RDC_ST_OK);
  for (uint32_t i = 0; i < count; i++) {
    result = rdc_group_gpu_get_info(rdc_handle, group_id_list[i], NULL);
    ASSERT_EQ(result, RDC_ST_INVALID_HANDLER);
    result = rdc_group_gpu_get_info(rdc_handle, group_id_list[i], &group_info);
    ASSERT_EQ(result, RDC_ST_OK);
    std::cout << "\tGroupID\t"
              << "GroupName\t"
              << "GPUIndex\n";
    std::cout << "\t" << group_id_list[i] << "\t" << group_info.group_name << "\t\t";
    for (uint32_t j = 0; j < group_info.count; j++) {
      std::cout << group_info.entity_ids[j];
      if (j < group_info.count - 1) {
        std::cout << ",";
      }
    }
    std::cout << std::endl;
  }

  result = rdc_group_gpu_destroy(0, group_id);
  ASSERT_EQ(result, RDC_ST_INVALID_HANDLER);

  result = rdc_group_gpu_destroy(rdc_handle, group_id);
  ASSERT_EQ(result, RDC_ST_OK);
}
