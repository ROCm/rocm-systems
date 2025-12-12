/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <vector>
#include <thread>

/*
 hipDeviceReset deletes all active streams including hipStreamPerThread.
 Scenario: App calls hipDeviceReset while in other thread some Async operation is in
          progress on hipStreamPerThread.
 Watch out: hipDeviceRest should be successfull without any crash
 */
static void Copy_to_device() {
  unsigned int ele_size = (32 * 1024);  // 32KB
  int* A_h = nullptr;
  int* A_d = nullptr;

  hipError_t status = hipHostMalloc(&A_h, ele_size * sizeof(int));
  if (status != hipSuccess) return;

  status = hipMalloc(&A_d, ele_size * sizeof(int));
  if (status != hipSuccess) return;

  for (unsigned int i = 0; i < ele_size; ++i) {
    A_h[i] = 123;
  }
  HIP_CHECK(
      hipMemcpyAsync(A_d, A_h, ele_size * sizeof(int), hipMemcpyHostToDevice, hipStreamPerThread));
  // Clean up
  HIP_CHECK(hipHostFree(A_h));
  HIP_CHECK(hipFree(A_d));
}

HIP_TEST_CASE(Unit_hipStreamPerThread_DeviceReset_1) {
  constexpr unsigned int MAX_THREAD_CNT = 10;
  std::vector<std::thread> threads(MAX_THREAD_CNT);

  for (auto& th : threads) {
    th = std::thread(Copy_to_device);
  }
  for (auto& th : threads) {
    th.join();
  }

  HIP_CHECK(hipDeviceReset());
}

/*
 hipDeviceReset deletes all active streams including hipStreamPerThread.
 Scenario: i) Launch Async task on hipStreamPerThread and waits for it to complete.
          ii) Call hipDeviceReset to delete all active stream
         iii) Again try to launch Async task on hipStreamPerThread
 Watch out: Since hipStreamPerThread is an implicit stream hence even after device reset
            it should available to use.
 */
HIP_TEST_CASE(Unit_hipStreamPerThread_DeviceReset_2) {
  unsigned int ele_size = (32 * 1024);  // 32KB
  int* A_h = nullptr;
  int* A_d = nullptr;

  hipError_t status = hipHostMalloc(&A_h, ele_size * sizeof(int));
  if (status != hipSuccess) return;
  status = hipMalloc(&A_d, ele_size * sizeof(int));
  if (status != hipSuccess) return;

  for (unsigned int i = 0; i < ele_size; ++i) {
    A_h[i] = 123;
  }
  status =
      hipMemcpyAsync(A_d, A_h, ele_size * sizeof(int), hipMemcpyHostToDevice, hipStreamPerThread);
  if (status != hipSuccess) return;
  HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

  // Host Memory is not destroyed with hipDeviceReset, need to free it
  // explicitly to avoid memory leaks
  HIP_CHECK(hipHostFree(A_h));
  HIP_CHECK(hipDeviceReset());

  // After reset all memory objects will be destroyed hence allocating them again
  // Intention is to use hipStreamPerThread successfully after reset hence not validating
  // values after copy
  status = hipHostMalloc(&A_h, ele_size * sizeof(int));
  if (status != hipSuccess) return;
  status = hipMalloc(&A_d, ele_size * sizeof(int));
  if (status != hipSuccess) return;

  status =
      hipMemcpyAsync(A_d, A_h, ele_size * sizeof(int), hipMemcpyHostToDevice, hipStreamPerThread);
  if (status != hipSuccess) return;
  HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

  // Clean up
  HIP_CHECK(hipHostFree(A_h));
  HIP_CHECK(hipFree(A_d));
}
