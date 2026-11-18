// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "test_base.h"

#include <gtest/gtest.h>

#include <cstdio>

TestBase::TestBase() = default;

void TestBase::SetUp() {
  uint32_t count = 0;
  amdcuid_status_t status = amdcuid_get_all_handles(nullptr, &count);

  if (status == AMDCUID_STATUS_UNSUPPORTED) {
    device_handles_.clear();
    return;
  }
  if (status != AMDCUID_STATUS_INSUFFICIENT_SIZE && status != AMDCUID_STATUS_SUCCESS) {
    IF_VERB(1) {
      printf("  SetUp: amdcuid_get_all_handles (count query) returned %s\n",
             amdcuid_status_to_string(status));
    }
    setup_failed_ = true;
    return;
  }

  device_handles_.resize(count);
  status = amdcuid_get_all_handles(device_handles_.data(), &count);

  if (status != AMDCUID_STATUS_SUCCESS) {
    IF_VERB(1) {
      printf("  SetUp: amdcuid_get_all_handles returned %s\n", amdcuid_status_to_string(status));
    }
    setup_failed_ = true;
    device_handles_.clear();
    return;
  }

  device_handles_.resize(count);

  IF_VERB(1) { printf("  SetUp: found %u device(s)\n", count); }
}

void TestBase::Close() {}

void TestBase::DisplayTestInfo() {
  IF_VERB(1) {
    printf("\n** Test: %s\n", title_.c_str());
    if (!description_.empty()) {
      printf("   %s\n", description_.c_str());
    }
  }
}

void TestBase::DisplayResults() const {
  IF_VERB(1) { printf("** %s: done\n", title_.c_str()); }
}

void RunGenericTest(TestBase* test) {
  test->DisplayTestInfo();
  test->SetUp();
  if (!test->SetupFailed()) {
    test->Run();
  }
  test->DisplayResults();
  test->Close();
}
