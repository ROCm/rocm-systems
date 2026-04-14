/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#define LEN 1024
#define SIZE (LEN << 2)

TEST_CASE(Unit_hipDrvMemcpy_Functional) {
  int *A, *B;
  hipDeviceptr_t Ad, Bd;
  A = new int[LEN];
  B = new int[LEN];

  for (int i = 0; i < LEN; i++) {
    A[i] = i;
  }

  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));

  HIP_CHECK(hipMemcpyHtoD(Ad, A, SIZE));
  HIP_CHECK(hipMemcpyDtoD(Bd, Ad, SIZE));
  HIP_CHECK(hipMemcpyDtoH(B, Bd, SIZE));

  for (int i = 0; i < 16; i++) {
    REQUIRE(A[i] == B[i]);
  }

  int *Ah, *Bh;
  HIP_CHECK(hipHostMalloc(&Ah, SIZE, 0));
  HIP_CHECK(hipHostMalloc(&Bh, SIZE, 0));
  memcpy(Ah, A, SIZE);
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMemcpyHtoDAsync(Ad, Ah, SIZE, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpyDtoDAsync(Bd, Ad, SIZE, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpyDtoHAsync(Bh, Bd, SIZE, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(Ad)));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(Bd)));

  REQUIRE(Ah[10] == Bh[10]);
  HIP_CHECK(hipFreeHost(Ah));
  HIP_CHECK(hipFreeHost(Bh));
  delete[] A;
  delete[] B;
}
