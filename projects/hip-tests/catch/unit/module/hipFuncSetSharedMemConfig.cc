/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

__global__ void ReverseSeq(int* A, int* B, int N) {
  extern __shared__ int SMem[];
  int offset = threadIdx.x;
  int MirrorVal = N - offset - 1;
  SMem[offset] = A[offset];
  __syncthreads();
  B[offset] = SMem[MirrorVal];
}
/**
 * @addtogroup hipFuncSetSharedMemConfig
 * @{
 * @ingroup ModuleTest
 * `hipError_t hipFuncSetSharedMemConfig(const void* func, hipSharedMemConfig config)` -
 * Sets shared memory configuation for a specific function
 */

/**
 * Test Description
 * ------------------------
 * - Test case to set shared memory configuations for a specific function for different flags.

 * Test source
 * ------------------------
 * - catch/unit/module/hipFuncSetSharedMemConfig.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
*/
TEST_CASE(Unit_hipFuncSetSharedMemConfig_functional) {
  int *Ah = NULL, *RAh = NULL, NELMTS = 128;
  int *Ad = NULL, *RAd = NULL;
  Ah = reinterpret_cast<int*>(malloc(NELMTS * sizeof(int)));
  RAh = reinterpret_cast<int*>(malloc(NELMTS * sizeof(int)));
  HIP_CHECK(hipMalloc(&Ad, NELMTS * sizeof(int)));
  HIP_CHECK(hipMalloc(&RAd, NELMTS * sizeof(int)));
  for (int i = 0; i < NELMTS; ++i) {
    Ah[i] = i;
    RAh[i] = NELMTS - i - 1;
  }
  HIP_CHECK(hipMemcpy(Ad, Ah, NELMTS * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(RAd, 0, NELMTS * sizeof(int)));

  // Testing hipFuncSetSharedMemConfig() with hipSharedMemBankSizeDefault flag
  SECTION("Flag: hipSharedMemBankSizeDefault") {
    HIP_CHECK(hipFuncSetSharedMemConfig(reinterpret_cast<const void*>(&ReverseSeq),
                                        hipSharedMemBankSizeDefault));
    // Kernel Launch with shared mem size of = NELMTS * sizeof(int)
    ReverseSeq<<<1, NELMTS, NELMTS * sizeof(int)>>>(Ad, RAd, NELMTS);
    memset(Ah, 0, NELMTS * sizeof(int));
    // Verifying the results
    HIP_CHECK(hipMemcpy(Ah, RAd, NELMTS * sizeof(int), hipMemcpyDeviceToHost));
    for (int i = 0; i < NELMTS; ++i) {
      REQUIRE(Ah[i] == RAh[i]);
    }
  }

  // Testing hipFuncSetSharedMemConfig() with hipSharedMemBankSizeFourBytes flag
  SECTION("Flag: hipSharedMemBankSizeFourBytes") {
    HIP_CHECK(hipFuncSetSharedMemConfig(reinterpret_cast<const void*>(&ReverseSeq),
                                        hipSharedMemBankSizeFourByte));
    HIP_CHECK(hipMemset(RAd, 0, NELMTS * sizeof(int)));
    // Kernel Launch with shared mem size of = NELMTS * sizeof(int)
    ReverseSeq<<<1, NELMTS, NELMTS * sizeof(int)>>>(Ad, RAd, NELMTS);
    memset(Ah, 0, NELMTS * sizeof(int));
    // Verifying the results
    HIP_CHECK(hipMemcpy(Ah, RAd, NELMTS * sizeof(int), hipMemcpyDeviceToHost));
    for (int i = 0; i < NELMTS; ++i) {
      REQUIRE(Ah[i] == RAh[i]);
    }
  }
  // Testing hipFuncSetSharedMemConfig() with hipSharedMemBankSizeEightBytes flg
  SECTION("Flag: hipSharedMemBankSizeEightByte") {
    HIP_CHECK(hipFuncSetSharedMemConfig(reinterpret_cast<const void*>(&ReverseSeq),
                                        hipSharedMemBankSizeEightByte));
    HIP_CHECK(hipMemset(RAd, 0, NELMTS * sizeof(int)));
    // Kernel Launch with shared mem size of = NELMTS * sizeof(int)
    ReverseSeq<<<1, NELMTS, NELMTS * sizeof(int)>>>(Ad, RAd, NELMTS);
    memset(Ah, 0, NELMTS * sizeof(int));
    // Verifying the results
    HIP_CHECK(hipMemcpy(Ah, RAd, NELMTS * sizeof(int), hipMemcpyDeviceToHost));
    for (int i = 0; i < NELMTS; ++i) {
      REQUIRE(Ah[i] == RAh[i]);
    }
  }

  free(Ah);
  free(RAh);
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(RAd));
}
