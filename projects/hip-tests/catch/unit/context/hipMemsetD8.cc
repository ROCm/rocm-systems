/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#define N 1024
constexpr char memsetval = 'b';

HIP_TEST_CASE(Unit_hipMemsetD8_Functional) {
  size_t Nbytes = N * sizeof(char);
  char* A_h = new char[Nbytes];
  ;

  hipDeviceptr_t A_d;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_d), Nbytes));

  HIP_CHECK(hipMemsetD8(A_d, memsetval, Nbytes));

  HIP_CHECK(hipMemcpy(A_h, reinterpret_cast<void*>(A_d), Nbytes, hipMemcpyDeviceToHost));

  for (int i = 0; i < N; i++) {
    REQUIRE(A_h[i] == memsetval);
  }

  HIP_CHECK(hipFree(reinterpret_cast<void*>(A_d)));
  delete[] A_h;
}
