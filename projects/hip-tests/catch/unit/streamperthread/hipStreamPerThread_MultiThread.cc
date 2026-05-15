/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <vector>
#include <thread>

static void Copy_to_device() {
  unsigned int ele_size = (32 * 1024);  // 32KB
  int* A_h = nullptr;
  int* A_d = nullptr;

  HIP_CHECK(hipHostMalloc(&A_h, ele_size * sizeof(int)));
  HIP_CHECK(hipMalloc(&A_d, ele_size * sizeof(int)));

  for (unsigned int i = 0; i < ele_size; ++i) {
    A_h[i] = 123;
  }
  HIP_CHECK(
      hipMemcpyAsync(A_d, A_h, ele_size * sizeof(int), hipMemcpyHostToDevice, hipStreamPerThread));
  // Clean up
  HIP_CHECK(hipHostFree(A_h));
  HIP_CHECK(hipFree(A_d));
}

/*
hipStreamPerThread is an implicit stream which gets destroyed once thread is completed.
Scenario : App pushes Async task(s) into hipStreamPerThread and did not wait for it to complete.
Watch out : Incomplete task in hipStreamPerThread should not cause any crash due to thread exit.
 */
HIP_TEST_CASE(Unit_hipStreamPerThread_MultiThread) {
  constexpr unsigned int MAX_THREAD_CNT = 10;
  std::vector<std::thread> threads(MAX_THREAD_CNT);

  for (auto& th : threads) {
    th = std::thread(Copy_to_device);
  }

  for (auto& th : threads) {
    th.detach();
  }
}
