/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
unsafeAtomicAdd on CoarseGrainMemory
1. The following test scenario verifies
unsafeAtomicAdd on CoarseGrain memory without any unsafeatomics flag
This testcase works only on gfx90a, gfx942, gfx950.
*/

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_features.hh>

#define INC_VAL 10
#define INITIAL_VAL 5
template <typename T> static __global__ void AtomicCheck(T* Ad, T* result) {
  T inc_val = 10;
  *result = unsafeAtomicAdd(Ad, inc_val);
}

/*unsafeAtomicAdd API for the coarse grained memory variable
  without any flag
Input: Ad{5}, INC_VAL{10}
Output: unsafeAtomicAdd API would work and the 0/P is INITIAL_VAL + INC_VAL
        Generate the assembly file and check whether
        global_atomic_cmpswap instruction is generated
        or not */

HIP_TEMPLATE_TEST_CASE(Unit_unsafeAtomicAdd_NonCoherentwithoutflag, float, double) {
  hipDeviceProp_t prop;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  std::string gfxName(prop.gcnArchName);
  if (CheckIfFeatSupported(CTFeatures::CT_FEATURE_FINEGRAIN_HWSUPPORT, gfxName)) {
    if (prop.canMapHostMemory != 1) {
      SUCCEED("Does not support HostPinned Memory");
    } else {
      TestType *A_h{nullptr}, *result{nullptr};
      TestType *A_d{nullptr}, *result_d{nullptr};
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), sizeof(TestType),
                              hipHostMallocNonCoherent));
      A_h[0] = INITIAL_VAL;
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&result), sizeof(TestType),
                              hipHostMallocNonCoherent));
      result[0] = INITIAL_VAL;
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d), A_h, 0));
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&result_d), result, 0));
      hipLaunchKernelGGL(AtomicCheck<TestType>, dim3(1), dim3(1), 0, 0, A_d, result_d);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      bool testResult;
      REQUIRE(A_h[0] == INITIAL_VAL + INC_VAL);
      REQUIRE(result[0] == INITIAL_VAL);
      if ((std::is_same<TestType, float>::value)) {
        testResult = HipTest::assemblyFile_Verification<TestType>(
            "unsafeAtomicAdd_NonCoherent_withoutflag-hip-amdgcn(.*)\\.s", "global_atomic_add_f32");
        REQUIRE(testResult == true);
      } else {
        testResult = HipTest::assemblyFile_Verification<TestType>(
            "unsafeAtomicAdd_NonCoherent_withoutflag-hip-amdgcn(.*)\\.s", "global_atomic_add_f64");
        REQUIRE(testResult == true);
      }
      HIP_CHECK(hipHostFree(A_h));
      HIP_CHECK(hipHostFree(result));
    }
  } else {
    SUCCEED(
        "Memory model feature is only supported for gfx90a, gfx942, gfx950,"
        "Hence skipping the testcase for this GPU "
        << device);
  }
}
