// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "kfd_atfork_read.h"

#include <gtest/gtest.h>
#include <pthread.h>

#include <atomic>
#include <cstdint>
#include <iostream>

#include "rocm_smi/rocm_smi_kfd_data_manager.h"
#include "test_common.h"

namespace {
// pthread_atfork callbacks take no arguments, so the counter is file-scoped.
std::atomic<int> g_atfork_calls{0};
void CountAtfork() { ++g_atfork_calls; }
}  // namespace

TestKfdAtforkRead::TestKfdAtforkRead() : TestBase() {
  set_title("KFD atfork Bypass Test");
  set_description(
      "Verifies the KFD VRAM helper spawns its short-lived child with clone() "
      "rather than fork(), so a caller's pthread_atfork handlers are not "
      "triggered (ROCM-24163). The helper is invoked directly because the "
      "public getter only reaches it as a sysfs fallback.");
}

TestKfdAtforkRead::~TestKfdAtforkRead(void) {}

void TestKfdAtforkRead::SetUp(void) { TestBase::SetUp(); }

void TestKfdAtforkRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestKfdAtforkRead::DisplayResults(void) const { TestBase::DisplayResults(); }

void TestKfdAtforkRead::Close() { TestBase::Close(); }

void TestKfdAtforkRead::Run(void) {
  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  // Register once; atfork handlers persist process-wide.
  static const int reg = pthread_atfork(CountAtfork, CountAtfork, CountAtfork);
  ASSERT_EQ(reg, 0);

  // ExecuteIsolatedQuery always spawns a child, so the gpu_id need not be valid
  // for the spawn (and thus the atfork check) to be exercised.
  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);

    g_atfork_calls = 0;  // measure only this spawn's effect
    uint64_t available = 0;
    (void)amd::smi::kfd::QueryAvailableVram(i, &available);

    EXPECT_EQ(g_atfork_calls.load(), 0)
        << "KFD VRAM helper fired pthread_atfork handlers on gpu=" << i
        << " (expected clone(), got fork())";
  }
}
