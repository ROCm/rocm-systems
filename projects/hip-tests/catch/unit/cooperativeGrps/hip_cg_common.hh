/*
Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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