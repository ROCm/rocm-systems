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

#include <pthread.h>

#include <iostream>
#include <thread>  // NOLINT
#include <random>
#include <chrono>  // NOLINT

#include "init_shutdown_refcount.h"
#include <gtest/gtest.h>
#include "amd_smi/amdsmi.h"

extern int32_t
rsmi_test_refcount(uint64_t refcnt_type);

static void rand_sleep_mod(int msec) {
  assert(msec > 10);
  uint64_t seed = static_cast<uint64_t>(time(NULL));
  std::mt19937_64 eng{seed};
  std::uniform_int_distribution<> dist{10, msec};
  std::this_thread::sleep_for(std::chrono::milliseconds{dist(eng)});
}

static void* AMDSMIInitFunction(void* args) {
  amdsmi_status_t status;

  (void)args;
  rand_sleep_mod(100);
  status = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
  EXPECT_EQ(AMDSMI_STATUS_SUCCESS, status);
  pthread_exit(nullptr);
  return nullptr;
}

static void* AMDSMIShutDownFunction(void* args) {
  amdsmi_status_t status;

  (void)args;
  rand_sleep_mod(100);
  status = amdsmi_shut_down();
  EXPECT_EQ(AMDSMI_STATUS_SUCCESS, status);
  pthread_exit(nullptr);
  return nullptr;
}

static void *AMDSMIInitShutDownFunction(void* args) {
  amdsmi_status_t status;

  (void)args;
  rand_sleep_mod(100);
  status = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
  EXPECT_EQ(AMDSMI_STATUS_SUCCESS, status);

  rand_sleep_mod(100);

  status = amdsmi_shut_down();
  EXPECT_EQ(AMDSMI_STATUS_SUCCESS, status);
  pthread_exit(nullptr);
  return nullptr;
}

static const int NumOfThreads = 100;

TestConcurrentInit::TestConcurrentInit(void) : TestBase() {
  set_title("AMDSMI Concurrent Init Test");
  set_description("This test initializes AMDSMI concurrently to verify "
                                         "reference counting functionality.");
}

TestConcurrentInit::~TestConcurrentInit(void) {
}

void TestConcurrentInit::SetUp(void) {
  // TestBase::SetUp();  // Skip usual SetUp to avoid doing the usual amdsmi_init
  return;
}

// Compare required profile for this test case with what we're actually
// running on
void TestConcurrentInit::DisplayTestInfo(void) {
  IF_VERB(STANDARD) {
    TestBase::DisplayTestInfo();
  }
  return;
}

void TestConcurrentInit::DisplayResults(void) const {
  IF_VERB(STANDARD) {
    TestBase::DisplayResults();
  }
  return;
}

void TestConcurrentInit::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

// Compare required profile for this test case with what we're actually
// running on
void TestConcurrentInit::Run(void) {
  if (setup_failed_) {
    IF_VERB(STANDARD) {
      std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    }
    return;
  }

  pthread_t ThreadId[NumOfThreads];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  IF_VERB(STANDARD) {
    std::cout << "Testing concurrent amdsmi_init()..." << std::endl;
  }
  for (int Id = 0; Id < NumOfThreads; ++Id) {
    int ThreadStatus = pthread_create(&ThreadId[Id], &attr,
                                                   AMDSMIInitFunction, nullptr);
    ASSERT_EQ(0, ThreadStatus) << "pthead_create failed.";
  }

  for (int Id = 0; Id < NumOfThreads; ++Id) {
    int err = pthread_join(ThreadId[Id], nullptr);
    ASSERT_EQ(0, err) << "pthread_join failed.";
  }

  // Invoke hsa_shut_down and verify that all the hsa_init's were counted.
  // HSA should be exactly closed after NumOfThreads calls.
  for (int Id = 0; Id < NumOfThreads; ++Id) {
    amdsmi_status_t err = amdsmi_shut_down();
    ASSERT_EQ(AMDSMI_STATUS_SUCCESS, err) << "An amdsmi_init was missed.";
  }

  amdsmi_status_t err = amdsmi_shut_down();
  ASSERT_EQ(AMDSMI_STATUS_INIT_ERROR, err) <<
                "amdsmi_init reference count was too high.";

  int32_t refcnt = rsmi_test_refcount(0);
  ASSERT_EQ(0, refcnt);

  IF_VERB(STANDARD) {
    std::cout << "Concurrent amdsmi_init() test passed." <<
                                                std::endl << std::endl;
    std::cout << "Testing concurrent amdsmi_shut_down()..." << std::endl;
  }
  // Invoke hsa_shut_down and verify that all the hsa_init's were counted.
  // HSA should be exactly closed after NumOfThreads calls.
  for (int Id = 0; Id < NumOfThreads; ++Id) {
    amdsmi_status_t err = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
    ASSERT_EQ(AMDSMI_STATUS_SUCCESS, err);
  }

  for (int Id = 0; Id < NumOfThreads; ++Id) {
    int ThreadStatus =
         pthread_create(&ThreadId[Id], &attr, AMDSMIShutDownFunction, nullptr);
    ASSERT_EQ(0, ThreadStatus) << "pthead_create failed.";
  }

  for (int Id = 0; Id < NumOfThreads; ++Id) {
    int err = pthread_join(ThreadId[Id], nullptr);
    ASSERT_EQ(0, err) << "pthread_join failed.";
  }

  refcnt = rsmi_test_refcount(0);
  ASSERT_EQ(0, refcnt);

  IF_VERB(STANDARD) {
    std::cout << "Concurrent amdsmi_shut_down() passed." << std::endl;
    std::cout <<
      "Testing concurrent amdsmi_init() followed by amdsmi_shut_down()..." <<
                                                                    std::endl;
  }
  for (int Id = 0; Id < NumOfThreads; ++Id) {
    int ThreadStatus =
      pthread_create(&ThreadId[Id], &attr, AMDSMIInitShutDownFunction, nullptr);
    ASSERT_EQ(0, ThreadStatus) << "pthead_create failed.";
  }

  for (int Id = 0; Id < NumOfThreads; ++Id) {
    int err = pthread_join(ThreadId[Id], nullptr);
    ASSERT_EQ(0, err) << "pthread_join failed.";
  }

  refcnt = rsmi_test_refcount(0);
  ASSERT_EQ(0, refcnt);

  IF_VERB(STANDARD) {
    std::cout <<
      "Concurrent amdsmi_init() followed by amdsmi_shut_down() passed." <<
                                                                    std::endl;
  }
}
