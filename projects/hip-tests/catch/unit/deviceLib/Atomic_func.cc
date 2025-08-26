/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>

// Test case to validate atomicInc and atomicDec functions.
// if TestToRun=1, then atomicInc function will be tested and validated
// if TestToRun=2, then atomicDec function will be tested and validated.


// kernel function for atomicInc
static __global__ void AtomicCheckInc(int* g_ptr) {
  atomicInc(reinterpret_cast<unsigned int*>(&g_ptr[0]), 17);
}

// kernel function for atomicDec
static __global__ void AtomicCheckDec(int* g_ptr) {
  atomicDec(reinterpret_cast<unsigned int*>(&g_ptr[0]), 25);
}

// verify results for atomicInc
static int verifyResultInc(int value) {
  int limit = 17;
  value = (value >= limit) ? 0 : value + 1;
  return value;
}

// verify results for atomicDec
static int verifyResultDec(int value) {
  int limit = 25;
  value = ((value == 0) || (value > limit)) ? limit : value - 1;
  return value;
}

// common fuction to launch atomic functions kernel.
static void launchAtomicFunction(int* Hptr, int val, int TestToRun) {
  unsigned int memSize = sizeof(int) * 1;
  int* dptr{nullptr};
  // allocate device memory
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dptr), memSize));
  // copy host memory to device
  HIP_CHECK(hipMemcpy(dptr, Hptr, memSize, hipMemcpyHostToDevice));
  // launch kernel function
  if (TestToRun == 1) {
    AtomicCheckInc<<<1, 1>>>(dptr);
  } else if (TestToRun == 2) {
    AtomicCheckDec<<<1, 1>>>(dptr);
  }
  // copy back from device to host
  HIP_CHECK(hipMemcpy(Hptr, dptr, memSize, hipMemcpyDeviceToHost));
  // verify the results.
  if (TestToRun == 1) {
    int result = verifyResultInc(val);
    REQUIRE(result == Hptr[0]);
  } else if (TestToRun == 2) {
    int result = verifyResultDec(val);
    REQUIRE(result == Hptr[0]);
  }
  // Cleanup memory
  HIP_CHECK(hipFree(dptr));
}

TEST_CASE("Unit_AtomicFunctions_Inc") {
  int* Hptr{nullptr};
  int val;
  // Allocate Host memory
  Hptr = reinterpret_cast<int*>(malloc(sizeof(int)));
  SECTION("Test case when value is lesser than limit") {
    val = Hptr[0] = 10;
    launchAtomicFunction(Hptr, val, 1);
  }
  SECTION("Test case when value is greater than limit") {
    val = Hptr[0] = 20;
    launchAtomicFunction(Hptr, val, 1);
  }
  SECTION("Test case when value is equal to the limit") {
    val = Hptr[0] = 17;
    launchAtomicFunction(Hptr, val, 1);
  }
  free(Hptr);
}

TEST_CASE("Unit_AtomicFunctions_Dec") {
  int* Hptr{nullptr};
  int val;
  // Allocate Host memory
  Hptr = reinterpret_cast<int*>(malloc(sizeof(int)));
  SECTION("Test case when value is less than limit") {
    val = Hptr[0] = 4;
    launchAtomicFunction(Hptr, val, 2);
  }
  SECTION("Test case when value is greater than limit") {
    val = Hptr[0] = 31;
    launchAtomicFunction(Hptr, val, 2);
  }
  SECTION("Test case when value is equal to the limit") {
    val = Hptr[0] = 25;
    launchAtomicFunction(Hptr, val, 2);
  }
  free(Hptr);
}
