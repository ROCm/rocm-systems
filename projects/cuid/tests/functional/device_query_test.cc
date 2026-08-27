// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "functional/device_query_test.h"

#include <gtest/gtest.h>

#include <cstdio>

TestDeviceQuery::TestDeviceQuery() {
  SetTitle("Device Query");
  SetDescription(
      "Verify amdcuid_query_device_property returns SUCCESS and the correct "
      "output size when querying AMDCUID_QUERY_DEVICE_TYPE for each handle.");
}

void TestDeviceQuery::Run() {
  if (device_handles_.empty()) {
    GTEST_SKIP() << "No devices found; skipping device query test.";
  }

  for (size_t i = 0; i < device_handles_.size(); ++i) {
    amdcuid_device_type_t device_type;
    uint32_t length = sizeof(device_type);
    amdcuid_status_t status = amdcuid_query_device_property(
        device_handles_[i], AMDCUID_QUERY_DEVICE_TYPE, &device_type, &length);

    CHK_ERR_ASRT(status);
    EXPECT_EQ(length, sizeof(device_type));

    IF_VERB(1) { printf("  Device [%zu] type: %u\n", i, static_cast<unsigned>(device_type)); }
  }
}
