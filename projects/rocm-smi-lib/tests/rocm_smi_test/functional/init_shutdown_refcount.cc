/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include <pthread.h>

#include <algorithm>
#include <iostream>
#include <thread>  // NOLINT
#include <random>
#include <chrono>  // NOLINT

#include "rocm_smi_test/functional/init_shutdown_refcount.h"
#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/test_common.h"

extern int32_t
rsmi_test_refcount(uint64_t refcnt_type);

static void rand_sleep_mod(int msec) {
  assert(msec > 10);
  unsigned int seed = time(NULL);
  std::mt19937_64 eng{seed};
  std::uniform_int_distribution<> dist{10, msec};
  std::this_thread::sleep_for(std::chrono::milliseconds{dist(eng)});
}

static void* RSMIInitFunction(void* args) {
  rsmi_status_t status;

  (void)args;
  rand_sleep_mod(100);
  status = rsmi_init(0);
  EXPECT_EQ(RSMI_STATUS_SUCCESS, status);
  pthread_exit(nullptr);
  return nullptr;
}

static void* RSMIShutDownFunction(void* args) {
  rsmi_status_t status;

  (void)args;
  rand_sleep_mod(100);
  status = rsmi_shut_down();
  EXPECT_EQ(RSMI_STATUS_SUCCESS, status);
  pthread_exit(nullptr);
  return nullptr;
}

static void *RSMIInitShutDownFunction(void* args) {
  rsmi_status_t status;

  (void)args;
  rand_sleep_mod(100);
  status = rsmi_init(0);
  EXPECT_EQ(RSMI_STATUS_SUCCESS, status);

  rand_sleep_mod(100);

  status = rsmi_shut_down();
  EXPECT_EQ(RSMI_STATUS_SUCCESS, status);
  pthread_exit(nullptr);
  return nullptr;
}

static const int NumOfThreads = 100;

TestConcurrentInit::TestConcurrentInit(void) : TestBase() {
  set_title("RSMI Concurrent Init Test");
  set_description("This test initializes RSMI concurrently to verify "
                                         "reference counting functionality.");
}

TestConcurrentInit::~TestConcurrentInit(void) {
}

void TestConcurrentInit::SetUp(void) {
  // TestBase::SetUp();  // Skip usual SetUp to avoid doing the usual rsmi_init
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
  // rsmi_shut_down(), so it should be done after other hsa cleanup
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
    std::cout << "Testing concurrent rsmi_init()..." << std::endl;
  }
  for (int Id = 0; Id < NumOfThreads; ++Id) {
    int ThreadStatus = pthread_create(&ThreadId[Id], &attr,
                                                   RSMIInitFunction, nullptr);
    ASSERT_EQ(0, ThreadStatus) << "pthead_create failed.";
  }

  for (int Id = 0; Id < NumOfThreads; ++Id) {
    int err = pthread_join(ThreadId[Id], nullptr);
    ASSERT_EQ(0, err) << "pthread_join failed.";
  }

  // Invoke hsa_shut_down and verify that all the hsa_init's were counted.
  // HSA should be exactly closed after NumOfThreads calls.
  for (int Id = 0; Id < NumOfThreads; ++Id) {
    rsmi_status_t err = rsmi_shut_down();
    ASSERT_EQ(RSMI_STATUS_SUCCESS, err) << "An rsmi_init was missed.";
  }

  rsmi_status_t err = rsmi_shut_down();
  ASSERT_EQ(RSMI_INITIALIZATION_ERROR, err) <<
                "rsmi_init reference count was too high.";

  int32_t refcnt = rsmi_test_refcount(0);
  ASSERT_EQ(0, refcnt);

  IF_VERB(STANDARD) {
    std::cout << "Concurrent rsmi_init() test passed." <<
                                                std::endl << std::endl;
    std::cout << "Testing concurrent rsmi_shut_down()..." << std::endl;
  }
  // Invoke hsa_shut_down and verify that all the hsa_init's were counted.
  // HSA should be exactly closed after NumOfThreads calls.
  for (int Id = 0; Id < NumOfThreads; ++Id) {
    rsmi_status_t err = rsmi_init(0);
    ASSERT_EQ(RSMI_STATUS_SUCCESS, err);
  }

  for (int Id = 0; Id < NumOfThreads; ++Id) {
    int ThreadStatus =
         pthread_create(&ThreadId[Id], &attr, RSMIShutDownFunction, nullptr);
    ASSERT_EQ(0, ThreadStatus) << "pthead_create failed.";
  }

  for (int Id = 0; Id < NumOfThreads; ++Id) {
    int err = pthread_join(ThreadId[Id], nullptr);
    ASSERT_EQ(0, err) << "pthread_join failed.";
  }

  refcnt = rsmi_test_refcount(0);
  ASSERT_EQ(0, refcnt);

  IF_VERB(STANDARD) {
    std::cout << "Concurrent rsmi_shut_down() passed." << std::endl;
    std::cout <<
      "Testing concurrent rsmi_init() followed by rsmi_shut_down()..." <<
                                                                    std::endl;
  }
  for (int Id = 0; Id < NumOfThreads; ++Id) {
    int ThreadStatus =
      pthread_create(&ThreadId[Id], &attr, RSMIInitShutDownFunction, nullptr);
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
      "Concurrent rsmi_init() followed by rsmi_shut_down() passed." <<
                                                                    std::endl;
  }
}
