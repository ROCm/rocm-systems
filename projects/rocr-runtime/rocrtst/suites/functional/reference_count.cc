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

/* Test Name: reference_count
 *
 * Purpose: Verifies that the hsa_init and hsa_shutdown APIs properly increment
 * and decrement reference counting.
 *
 * Test Description:
 * 1) Initialize the ROC runtime with hsa_init by calling that API N times, (N
 * should be large).
 * 2) Verify that the runtime is operational by querying the agent list.
 * 3) Call hsa_shutdown N-1 times.
 * 4) Again, verify the runtime is operational by querying the agent list.
 *
 * Expected Results: The runtime should remain operational when the reference
 * count is positive. Repeated calls to hsa_init should not cause undefined behavior.
 *
 */
#include <algorithm>
#include <iostream>


#include "suites/functional/reference_count.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

static const int NumOfTimes = 1000;  // No of times the hsa runtime will be initialized
static const double MaxRefCount = 2147483649;  // Setting to max value to test to INIT_MAX+2 as defined in hsa runtime

#define RET_IF_HSA_ERR(err) { \
  if ((err) != HSA_STATUS_SUCCESS) { \
    const char* msg = 0; \
    hsa_status_string(err, &msg); \
    std::cout << "hsa api call failure at line " << __LINE__ << ", file: " << \
                          __FILE__ << ". Call returned " << err << std::endl; \
    std::cout << msg << std::endl; \
  } \
}

ReferenceCountTest::ReferenceCountTest(bool referenceCount_, bool maxReferenceCount_) : TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.
  if (referenceCount_) {
    set_title("RocR Reference Count Test");
    set_description("Initializes HSA runtime N times and shutdown N-1 times, again call shutdown");
  } else if (maxReferenceCount_) {
    set_title("RocR Max Reference Count Test");
    set_description("This test initializes HSA runtime to maximum allowed reference count");
  }
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
ReferenceCountTest::~ReferenceCountTest(void) {
}

// Compare required profile for this test case with what we're actually
// running on
void ReferenceCountTest::SetUp(void) {
  return;  // hsa runtime initalized in ReferenceCountTest::TestReferenceCount()
}


// Compare required profile for this test case with what we're actually
// running on
void ReferenceCountTest::Run(void) {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

// Compare required profile for this test case with what we're actually
// running on
void ReferenceCountTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void ReferenceCountTest::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void ReferenceCountTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  // all the reference count decremented in main function, ReferenceCountTest::TestReferenceCount(void)
}

void ReferenceCountTest::TestReferenceCount(void) {
  hsa_status_t status;
  // Initialize hsa runtime N times
  for (int i = 0; i < NumOfTimes; ++i) {
    status = hsa_init();
    RET_IF_HSA_ERR(status);
  }

  // Shutdown hsa runtime N - 1 times
  for (int i = 0; i < NumOfTimes-1; ++i) {
    status = hsa_shut_down();
    RET_IF_HSA_ERR(status);
  }

  status = hsa_shut_down();
  RET_IF_HSA_ERR(status);
}

void ReferenceCountTest::TestMaxReferenceCount(void) {
  hsa_status_t status;
  // Initialize hsa runtime to maximum allowed  times
  for (int i = 0; i < MaxRefCount; ++i) {
    status = hsa_init();
    if (status != HSA_STATUS_SUCCESS && status == HSA_STATUS_ERROR_REFCOUNT_OVERFLOW) {
      std::cout << "Max allowed reference count is = " << i << std::endl;
      // Gracefull exit after reaching the INIT_MAX as defined in hsa rutnime.
      break;
    }
  }
  for (int i = 0; i < MaxRefCount-2; ++i) {
    status = hsa_shut_down();
    RET_IF_HSA_ERR(status);
  }
}
#undef RET_IF_HSA_ERR
