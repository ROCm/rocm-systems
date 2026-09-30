/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>

#define ASSERT_EQUAL(lhs, rhs) HIP_ASSERT(lhs == rhs)
#define ASSERT_LE(lhs, rhs) HIPASSERT(lhs <= rhs)
#define ASSERT_GE(lhs, rhs) HIPASSERT(lhs >= rhs)

constexpr int MaxGPUs = 8;

template <typename T>
void printResults(T* ptr, int size) {
  for (int i = 0; i < size; i++) {
    std::cout << ptr[i] << " ";
  }
  std::cout << '\n';
}

template <typename T>
void compareResults(T* cpu, T* gpu, int size) {
  for (unsigned int i = 0; i < size / sizeof(T); i++) {
    if (cpu[i] != gpu[i]) {
      INFO("Results do not match at index " << i);
      REQUIRE(cpu[i] == gpu[i]);
    }
  }
}


// Search if the sum exists in the expected results array
template <typename T>
void verifyResults(T* hPtr, T* dPtr, int size) {
  int i = 0, j = 0;
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      if (hPtr[i] == dPtr[j]) {
        break;
      }
    }
    if (j == size) {
      INFO("Result verification failed!");
      REQUIRE(j != size);
    }
  }
}