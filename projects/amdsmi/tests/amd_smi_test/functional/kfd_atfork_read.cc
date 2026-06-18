/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include "kfd_atfork_read.h"

#include <gtest/gtest.h>
#include <pthread.h>

#include <atomic>
#include <cstdint>
#include <iostream>

#include "../test_common.h"
#include "rocm_smi/rocm_smi_kfd_data_manager.h"

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
