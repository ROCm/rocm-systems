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

#include "rdc_tests/functional/rdci_discovery.h"

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>

#include <iostream>

#include "rdc/rdc.h"
#include "rdc_tests/test_common.h"

TestRdciDiscovery::TestRdciDiscovery() : TestBase() {
  set_title("\tRDC Discovery Test");
  set_description(
      "\tThe Discovery tests verifies that the GPUs are "
      "discovered and identified .");
}

TestRdciDiscovery::~TestRdciDiscovery(void) {}

void TestRdciDiscovery::SetUp(void) {
  TestBase::SetUp();
  rdc_status_t result = AllocateRDCChannel();
  ASSERT_EQ(result, RDC_ST_OK);
  return;
}

void TestRdciDiscovery::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestRdciDiscovery::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestRdciDiscovery::Close() {
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

void TestRdciDiscovery::Run(void) {
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

  uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES];
  uint32_t count = 0;

  IF_VERB(STANDARD) { std::cout << "\t**Getting the devices in the system\n" << std::endl; }
  result = rdc_device_get_all(0, gpu_index_list, &count);
  ASSERT_EQ(result, RDC_ST_INVALID_HANDLER);

  result = rdc_device_get_all(rdc_handle, gpu_index_list, 0);
  ASSERT_EQ(result, RDC_ST_INVALID_HANDLER);

  result = rdc_device_get_all(rdc_handle, gpu_index_list, &count);
  ASSERT_EQ(result, RDC_ST_OK);

  ASSERT_GT(count, 0);

  IF_VERB(STANDARD) { std::cout << "\t**Fetching attributes of every device\n" << std::endl; }

  rdc_device_attributes_t attribute;
  for (uint32_t i = 0; i < count; i++) {
    result = rdc_device_get_attributes(0, gpu_index_list[i], &attribute);
    ASSERT_EQ(result, RDC_ST_INVALID_HANDLER);

    result = rdc_device_get_attributes(rdc_handle, gpu_index_list[i], 0);
    ASSERT_EQ(result, RDC_ST_INVALID_HANDLER);

    result = rdc_device_get_attributes(rdc_handle, gpu_index_list[i], &attribute);
    ASSERT_EQ(result, RDC_ST_OK);

    std::cout << "\tGPU ID " << i << "  ||  " << attribute.device_name << std::endl;
  }
}
