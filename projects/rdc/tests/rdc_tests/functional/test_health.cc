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

#include "rdc_tests/functional/test_health.h"

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>

#include <iostream>

#include "rdc/rdc.h"
#include "rdc_tests/test_common.h"

TestRdciHealth::TestRdciHealth() : TestBase() {
  set_title("\tRDC Health Test");
  set_description(
      "\tThe Health tests verify that health watches can be set, read, checked and "
      "cleared, and that the page-retirement fields the memory check depends on are "
      "readable wherever the platform supports them.");
}

TestRdciHealth::~TestRdciHealth(void) {}

void TestRdciHealth::SetUp(void) {
  TestBase::SetUp();
  rdc_status_t result = AllocateRDCChannel();
  ASSERT_EQ(result, RDC_ST_OK);
  return;
}

void TestRdciHealth::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestRdciHealth::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestRdciHealth::Close() {
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

void TestRdciHealth::Run(void) {
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

  const uint32_t gpu_index = 0;
  const unsigned int all_components = RDC_HEALTH_WATCH_PCIE | RDC_HEALTH_WATCH_XGMI |
                                      RDC_HEALTH_WATCH_MEM | RDC_HEALTH_WATCH_EEPROM |
                                      RDC_HEALTH_WATCH_THERMAL | RDC_HEALTH_WATCH_POWER;

  rdc_gpu_group_t group_id;
  result = rdc_group_gpu_create(rdc_handle, RDC_GROUP_EMPTY, "GRP_HEALTH", &group_id);
  ASSERT_EQ(result, RDC_ST_OK);
  result = rdc_group_gpu_add(rdc_handle, group_id, gpu_index);
  ASSERT_EQ(result, RDC_ST_OK);

  // Set every component, read the setting back, run a check, clear.
  IF_VERB(STANDARD) { std::cout << "\t**Setting health watches for all components....\n"; }
  result = rdc_health_set(rdc_handle, group_id, all_components);
  ASSERT_EQ(result, RDC_ST_OK);

  unsigned int components = 0;
  result = rdc_health_get(rdc_handle, group_id, &components);
  ASSERT_EQ(result, RDC_ST_OK);
  ASSERT_EQ(components, all_components);

  // Let the watch collect at least one sample before reading values back.
  result = rdc_field_update_all(rdc_handle, 1);
  ASSERT_EQ(result, RDC_ST_OK);

  rdc_health_response_t response = {};
  result = rdc_health_check(rdc_handle, group_id, &response);
  ASSERT_EQ(result, RDC_ST_OK);
  ASSERT_LE(response.overall_health, RDC_HEALTH_RESULT_FAIL);
  ASSERT_LE(response.incidents_count, static_cast<unsigned int>(HEALTH_MAX_ERROR_ITEMS));

  // RETIRED_PAGE_NUM and PENDING_PAGE_NUM come from the same SMI call, so
  // either both are readable or neither is. Before the zero-bad-pages fix,
  // PENDING_PAGE_NUM failed on every healthy GPU while RETIRED_PAGE_NUM
  // succeeded, which is exactly the inequality asserted here.
  IF_VERB(STANDARD) { std::cout << "\t**Checking page-retirement fields....\n"; }
  rdc_field_value retired = {}, pending = {};
  rdc_status_t retired_result =
      rdc_field_get_latest_value(rdc_handle, gpu_index, RDC_HEALTH_RETIRED_PAGE_NUM, &retired);
  rdc_status_t pending_result =
      rdc_field_get_latest_value(rdc_handle, gpu_index, RDC_HEALTH_PENDING_PAGE_NUM, &pending);
  ASSERT_EQ(pending_result, retired_result);
  if (retired_result == RDC_ST_OK) {
    ASSERT_EQ(pending.status, retired.status);
    if (retired.value.l_int == 0) {
      // No bad pages means no pending pages.
      ASSERT_EQ(pending.value.l_int, 0);
    }
  }

  result = rdc_health_clear(rdc_handle, group_id);
  ASSERT_EQ(result, RDC_ST_OK);
  result = rdc_health_check(rdc_handle, group_id, &response);
  ASSERT_EQ(result, RDC_ST_NOT_FOUND);

  // The EEPROM component alone must watch the field eeprom_check reads.
  // Before that fix it watched only RDC_HEALTH_EEPROM_CONFIG_VALID, which no
  // check consumes, and this read returned RDC_ST_NOT_FOUND.
  IF_VERB(STANDARD) { std::cout << "\t**Setting the EEPROM component alone....\n"; }
  result = rdc_health_set(rdc_handle, group_id, RDC_HEALTH_WATCH_EEPROM);
  ASSERT_EQ(result, RDC_ST_OK);
  result = rdc_field_update_all(rdc_handle, 1);
  ASSERT_EQ(result, RDC_ST_OK);
  rdc_field_value ecc = {};
  result = rdc_field_get_latest_value(rdc_handle, gpu_index, RDC_FI_ECC_UNCORRECT_TOTAL, &ecc);
  ASSERT_EQ(result, RDC_ST_OK);
  result = rdc_health_clear(rdc_handle, group_id);
  ASSERT_EQ(result, RDC_ST_OK);

  // A group with no GPUs has nothing to probe and keeps the pre-probe
  // behaviour: the set succeeds and watches every candidate field.
  IF_VERB(STANDARD) { std::cout << "\t**Setting health watches on an empty group....\n"; }
  rdc_gpu_group_t empty_group_id;
  result = rdc_group_gpu_create(rdc_handle, RDC_GROUP_EMPTY, "GRP_HEALTH_EMPTY", &empty_group_id);
  ASSERT_EQ(result, RDC_ST_OK);
  result = rdc_health_set(rdc_handle, empty_group_id, all_components);
  ASSERT_EQ(result, RDC_ST_OK);
  result = rdc_health_clear(rdc_handle, empty_group_id);
  ASSERT_EQ(result, RDC_ST_OK);
  result = rdc_group_gpu_destroy(rdc_handle, empty_group_id);
  ASSERT_EQ(result, RDC_ST_OK);

  result = rdc_group_gpu_destroy(rdc_handle, group_id);
  ASSERT_EQ(result, RDC_ST_OK);
}
