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

#include "rdc_tests/functional/rdci_fieldgroup.h"

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>

#include <iostream>

#include "rdc/rdc.h"
#include "rdc_tests/test_common.h"

TestRdciFieldgroup::TestRdciFieldgroup() : TestBase() {
  set_title("\tRDC Fieldgroup Test");
  set_description("\tThe Fieldgroup tests verifies the creation/deletion of fieldgroups.");
}

TestRdciFieldgroup::~TestRdciFieldgroup(void) {}

void TestRdciFieldgroup::SetUp(void) {
  TestBase::SetUp();
  rdc_status_t result = AllocateRDCChannel();
  ASSERT_EQ(result, RDC_ST_OK);
  return;
}

void TestRdciFieldgroup::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestRdciFieldgroup::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestRdciFieldgroup::Close() {
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

void TestRdciFieldgroup::Run(void) {
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

  rdc_field_grp_t field_group_id;
  rdc_field_t field_ids[] = {RDC_FI_GPU_TEMP, RDC_FI_POWER_USAGE};
  rdc_field_t invalid_field_ids[] = {RDC_FI_INVALID, RDC_FI_INVALID};
  uint32_t fsize = sizeof(field_ids) / sizeof(field_ids[0]);
  uint32_t count = 0;
  rdc_field_group_info_t group_info;

  result = rdc_group_field_create(rdc_handle, fsize, &invalid_field_ids[0], "FIELD_GRP",
                                  &field_group_id);
  ASSERT_EQ(result, RDC_ST_NOT_SUPPORTED);

  result = rdc_group_field_create(NULL, fsize, &field_ids[0], "FIELD_GRP", &field_group_id);
  ASSERT_EQ(result, RDC_ST_INVALID_HANDLER);

  result = rdc_group_field_create(rdc_handle, fsize, &field_ids[0], NULL, NULL);
  ASSERT_EQ(result, RDC_ST_INVALID_HANDLER);

  result = rdc_group_field_create(rdc_handle, (RDC_MAX_FIELD_IDS_PER_FIELD_GROUP + 1),
                                  &field_ids[0], "FIELD_GRP", NULL);
  ASSERT_EQ(result, RDC_ST_INVALID_HANDLER);

  result = rdc_group_field_create(rdc_handle, fsize, &field_ids[0], "FIELD_GRP", &field_group_id);
  ASSERT_EQ(result, RDC_ST_OK);

  rdc_field_grp_t group_id_list[RDC_MAX_NUM_FIELD_GROUPS];
  result = rdc_group_field_get_all_ids(rdc_handle, group_id_list, 0);
  ASSERT_EQ(result, RDC_ST_INVALID_HANDLER);

  result = rdc_group_field_get_all_ids(rdc_handle, group_id_list, &count);
  ASSERT_EQ(result, RDC_ST_OK);

  ASSERT_GT(count, 0);

  std::cout << "\tGroupID\t"
            << "GroupName\t"
            << "FieldIds\n";
  for (uint32_t i = 0; i < count; i++) {
    result = rdc_group_field_get_info(rdc_handle, group_id_list[i], 0);
    ASSERT_EQ(result, RDC_ST_BAD_PARAMETER);

    result = rdc_group_field_get_info(rdc_handle, group_id_list[i], &group_info);
    ASSERT_EQ(result, RDC_ST_OK);
  }

  result = rdc_group_field_destroy(NULL, field_group_id);
  ASSERT_EQ(result, RDC_ST_INVALID_HANDLER);

  result = rdc_group_field_destroy(rdc_handle, field_group_id);
  ASSERT_EQ(result, RDC_ST_OK);
}
