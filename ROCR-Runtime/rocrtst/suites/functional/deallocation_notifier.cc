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

/* Test Name: deallocation_notifier
 *
 * Purpose: Verifies that deallocation callbacks are invoked prior to destruction,
 * are not retained between successive allocations, may be registered on non-base
 * addresses, are invoked exactly once, run concurrently with other APIs, and other
 * callbacks.
 *
 * Test Description:
 * Various interleavings of allocate, register callback, deregister callback, and deallocate.
 *
 * Expected Results: Callbacks should run before free returns.  Callbacks should trigger when
 * their allocation is released.  Free shoud deregister invoked callbacks.  Callbacks should not
 * be able to double free the allocation they monitor.  Callbacks should be able to execute
 * ROCr APIs including hsa_amd_memory_pool_allocate and hsa_amd_memory_pool_free, possibly
 * triggering other callbacks.
 *
 */
#include "suites/functional/deallocation_notifier.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

struct callback_status {
  int callback_status = 0;
  void* released_ptr = nullptr;
};

static callback_status notifiers[2];
static hsa_amd_memory_pool_t pool;

#define REGISTER(ptr, callback, i)                                                                 \
  do {                                                                                             \
    notifiers[i].callback_status = 0;                                                              \
    notifiers[i].released_ptr = ptr;                                                               \
    status = hsa_amd_register_deallocation_callback(ptr, callback, (void*)i);                      \
    ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Register deallocation callback error.";              \
  } while (false)

static void call(void* ptr, void* user) {
  size_t index = reinterpret_cast<size_t>(user);
  ASSERT_EQ(ptr, notifiers[index].released_ptr) << "Bad deallocation callback address";
  notifiers[index].callback_status = 1;
}

static void doublefree(void* ptr, void* user) {
  call(ptr, user);

  hsa_status_t status = hsa_amd_memory_pool_free(ptr);
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ALLOCATION, status) << "Double free did not return an error.";
}

static void recursive(void* ptr, void* user) {
  ASSERT_EQ(0, user) << "Wrong index.";
  call(ptr, user);

  hsa_status_t status = hsa_amd_memory_pool_allocate(pool, 4096, 0, &ptr);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory allocation failure.";
  REGISTER(ptr, call, 1);
  hsa_amd_memory_pool_free(ptr);
  ASSERT_EQ(1, notifiers[1].callback_status) << "Callback not executed.";
}

DeallocationNotifierTest::DeallocationNotifierTest() : TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.
  set_title("RocR Deallocation Notifier Test");
  set_description("Tests deallocation notification callbacks");
}

DeallocationNotifierTest::~DeallocationNotifierTest(void) {}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void DeallocationNotifierTest::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  pool = device_pool();

  return;
}

void DeallocationNotifierTest::Run(void) {
// Compare required profile for this test case with what we're actually
// running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
  TestDeallocationNotifier();
}

void DeallocationNotifierTest::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void DeallocationNotifierTest::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void DeallocationNotifierTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void DeallocationNotifierTest::TestDeallocationNotifier(void) {
  hsa_status_t status;

  // Attempt register on null address.  Should fail.
  void* ptr = nullptr;
  status = hsa_amd_register_deallocation_callback(ptr, call, (void*)0xDEADBEEF);
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status) << "Register deallocation callback error.";

  // Attempt register on bad address (ie one not known to ROCr).  Should fail.
  ptr = malloc(4096);
  status = hsa_amd_register_deallocation_callback(ptr, call, (void*)0xDEADBEEF);
  free(ptr);
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ALLOCATION, status) << "Register deallocation callback error.";

  // Allocate, register and free.  Callback should complete before free returns.
  status = hsa_amd_memory_pool_allocate(pool, 4096, 0, &ptr);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory allocation failure.";
  REGISTER(ptr, call, 0);
  status = hsa_amd_memory_pool_free(ptr);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory free failure.";
  ASSERT_EQ(1, notifiers[0].callback_status) << "Callback not executed.";

  // Re-allocate, free.  No callback should be invoked.
  notifiers[0].callback_status = 0;
  status = hsa_amd_memory_pool_allocate(pool, 4096, 0, &ptr);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory allocation failure.";
  status = hsa_amd_memory_pool_free(ptr);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory free failure.";
  ASSERT_EQ(0, notifiers[0].callback_status) << "Callback reused.";

  // Allocate, register with non-base address, free.
  status = hsa_amd_memory_pool_allocate(pool, 4096, 0, &ptr);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory allocation failure.";
  REGISTER((char*)ptr + 1024, call, 0);
  status = hsa_amd_memory_pool_free(ptr);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory free failure.";
  ASSERT_EQ(1, notifiers[0].callback_status) << "Callback not executed.";

  // Allocate, Register, Deregister, Free.  No callback should be invoked.
  status = hsa_amd_memory_pool_allocate(pool, 4096, 0, &ptr);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory allocation failure.";
  REGISTER((char*)ptr + 1024, call, 0);
  status = hsa_amd_deregister_deallocation_callback((char*)ptr + 1024, call);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Deregister deallocation callback error.";
  status = hsa_amd_memory_pool_free(ptr);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory free failure.";
  ASSERT_EQ(0, notifiers[0].callback_status) << "Callback reused.";

  // Allocate, register, register another and free.  Callbacks should complete before free returns.
  status = hsa_amd_memory_pool_allocate(pool, 4096, 0, &ptr);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory allocation failure.";
  REGISTER(ptr, call, 0);
  REGISTER((char*)ptr + 1024, call, 1);
  status = hsa_amd_memory_pool_free(ptr);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory free failure.";
  ASSERT_EQ(1, notifiers[0].callback_status) << "Callback not executed.";
  ASSERT_EQ(1, notifiers[1].callback_status) << "Callback not executed.";

  // Repeat deregister.  Should error.
  status = hsa_amd_deregister_deallocation_callback((char*)ptr + 1024, call);
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status) << "Deregister deallocation callback error.";

  // Deregister from null.  Should error.
  status = hsa_amd_deregister_deallocation_callback(nullptr, call);
  ASSERT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status) << "Deregister deallocation callback error.";

  // Allocate fragment (second <2MB vram allocation), register, free.
  void* ptr0;
  status = hsa_amd_memory_pool_allocate(pool, 4096, 0, &ptr0);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory allocation failure.";
  status = hsa_amd_memory_pool_allocate(pool, 4096, 0, &ptr);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory allocation failure.";
  REGISTER(ptr, call, 0);
  status = hsa_amd_memory_pool_free(ptr);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory free failure.";
  ASSERT_EQ(1, notifiers[0].callback_status) << "Callback not executed.";

  // Allocate multiple fragments, register, free.  Free order should be respected by callbacks.
  // Reuse fragment ptr0 from above.
  status = hsa_amd_memory_pool_allocate(pool, 4096, 0, &ptr);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory allocation failure.";
  REGISTER(ptr, call, 0);
  REGISTER(ptr0, call, 1);
  status = hsa_amd_memory_pool_free(ptr0);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory free failure.";
  ASSERT_EQ(1, notifiers[1].callback_status) << "Callback not executed.";
  ASSERT_EQ(0, notifiers[0].callback_status) << "Callback executed improperly.";
  status = hsa_amd_memory_pool_free(ptr);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory free failure.";
  ASSERT_EQ(1, notifiers[0].callback_status) << "Callback not executed.";

  // Allocate, register, free, with double free in callback.  Callbacks should not be able to free
  // the triggering address again.
  status = hsa_amd_memory_pool_allocate(pool, 4096, 0, &ptr);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory allocation failure.";
  REGISTER(ptr, doublefree, 0);
  status = hsa_amd_memory_pool_free(ptr);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory free failure.";
  ASSERT_EQ(1, notifiers[0].callback_status) << "Callback not executed.";

  // Allocate, register, free, with allocate, register, free in callback.  Callbacks should nest and
  // have access to HSA APIs.
  status = hsa_amd_memory_pool_allocate(pool, 4096, 0, &ptr);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory allocation failure.";
  REGISTER(ptr, recursive, 0);
  status = hsa_amd_memory_pool_free(ptr);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Memory free failure.";
  ASSERT_EQ(1, notifiers[0].callback_status) << "Callback not executed.";
}
