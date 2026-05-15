/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2018, Advanced Micro Devices, Inc.
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

#include <algorithm>
#include <iostream>


#include "suites/functional/concurrent_shutdown.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

void* TestHSAShutdownFunction(void* args) {
  // This function called for each thread
  // This will shutdown the HSA runtime concurrently.
  hsa_status_t status;

  // Shutdown the hsa runtime concurrently
  status = hsa_shut_down();
  if (status != HSA_STATUS_SUCCESS) {
    std::cout << "Failed" << std::endl;
  }
  pthread_exit(NULL);
}

static const int NumOfThreads = 1000;  // Number of thread to be created
static const int NumTimesInitalize = 1000;  // Number of time the hsa runtime will be initialized

#define RET_IF_HSA_ERR(err) { \
  if ((err) != HSA_STATUS_SUCCESS) { \
    const char* msg = 0; \
    hsa_status_string(err, &msg); \
    std::cout << "hsa api call failure at line " << __LINE__ << ", file: " << \
                          __FILE__ << ". Call returned " << err << std::endl; \
    std::cout << msg << std::endl; \
    return (err); \
  } \
}

ConcurrentShutdownTest::ConcurrentShutdownTest(void) : TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.
  set_title("RocR Concurrent Shutdown Test");
  set_description("This test initializes HSA runtime sequentially, shutdown concurrently");
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
ConcurrentShutdownTest::~ConcurrentShutdownTest(void) {
}

void ConcurrentShutdownTest::SetUp(void) {
  hsa_status_t status;
  // Initialize the hsa runtime sequentially, NumTimesInitalize
  for (int Counter = 0; Counter < NumTimesInitalize; ++Counter) {
  // Initialize hsa runtime NumTimesInitalize times.
    status = hsa_init();
    if (status != HSA_STATUS_SUCCESS) {
      std::cout << "Failed" << std::endl;
    }
  }
  return;  // hsa runtime initalized pthread callback function
}

void ConcurrentShutdownTest::Run(void) {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
  TestBase::Run();
}

void ConcurrentShutdownTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void ConcurrentShutdownTest::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void ConcurrentShutdownTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  // all the reference count decremented in main function, ConcurrentShutdownTest::SequentiallyInitializeRuntime()
}

void ConcurrentShutdownTest::TestConcurrentShutdown(void) {
  pthread_t ThreadId[NumOfThreads];
  pthread_attr_t attr;
  pthread_attr_init(&attr);

  // Setting the attribute to PTHREAD_CREATE_JOINABLE
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  for (int Id = 0; Id < NumOfThreads; ++Id) {  // This is to create threads concurrently
                                               // HSA runtime will be shutdown concurrently from each thread
    int ThreadStatus = pthread_create(ThreadId + Id,
                                      &attr, TestHSAShutdownFunction, &Id);
    // Check if the thread is created successfully
    if (ThreadStatus < 0) {
      std::cout << Id << "Thread creation failed " << std::endl;
    }
  }
}
#undef RET_IF_HSA_ERR
