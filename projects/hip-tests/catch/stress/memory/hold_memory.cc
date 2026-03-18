/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <iostream>
#include <chrono>
#include <thread>

#define HIP_CHECK(call)                                                                            \
  {                                                                                                \
    auto res_ = (call);                                                                            \
    if (res_ != hipSuccess) {                                                                      \
      std::cout << "Failed in: " << #call << std::endl;                                            \
      return -1;                                                                                   \
    }                                                                                              \
  }

int main() {
  size_t freeMem = 0, totalMem = 0;
  HIP_CHECK(hipMemGetInfo(&freeMem, &totalMem));

  void* ptr;
  HIP_CHECK(hipMalloc(&ptr, 0.4 * totalMem));  // hold 40% of total gpu memory
  std::cout << "Sleeping..." << std::endl;
  std::this_thread::sleep_for(
      std::chrono::seconds(4));  //  sleep for few seconds till test complete
  std::cout << "Waking up..." << std::endl;
  HIP_CHECK(hipFree(ptr));
}