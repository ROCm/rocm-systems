/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <memcpy1d_tests_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

// hipMemcpyDtoH
HIP_TEST_CASE(Unit_hipMemcpyDtoH_Positive_Basic) {
  MemcpyDeviceToHostShell<false>([](void* dst, void* src, size_t count) {
    return hipMemcpyDtoH(dst, reinterpret_cast<hipDeviceptr_t>(src), count);
  });
}

HIP_TEST_CASE(Unit_hipMemcpyDtoH_Positive_Synchronization_Behavior) {
  const auto f = [](void* dst, void* src, size_t count) {
    return hipMemcpyDtoH(dst, reinterpret_cast<hipDeviceptr_t>(src), count);
  };
  MemcpyDtoHPageableSyncBehavior(f, true);
  MemcpyDtoHPinnedSyncBehavior(f, true);
}

HIP_TEST_CASE(Unit_hipMemcpyDtoH_Negative_Parameters) {
  using namespace std::placeholders;
  LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoH(dst, reinterpret_cast<hipDeviceptr_t>(src), count);
      },
      host_alloc.ptr(), device_alloc.ptr(), kPageSize);
}

// hipMemcpyHtoD
HIP_TEST_CASE(Unit_hipMemcpyHtoD_Positive_Basic) {
  MemcpyHostToDeviceShell<false>([](void* dst, void* src, size_t count) {
    return hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(dst), src, count);
  });
}

HIP_TEST_CASE(Unit_hipMemcpyHtoD_Positive_Synchronization_Behavior) {
  MemcpyHPageabletoDSyncBehavior(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(dst), src, count);
      },
      true);
}

HIP_TEST_CASE(Unit_hipMemcpyHtoD_Negative_Parameters) {
  using namespace std::placeholders;
  LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(dst), src, count);
      },
      device_alloc.ptr(), host_alloc.ptr(), kPageSize);
}

// hipMemcpyDtoD
HIP_TEST_CASE(Unit_hipMemcpyDtoD_Positive_Basic) {
  const auto f = [](void* dst, void* src, size_t count) {
    return hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(dst),
                         reinterpret_cast<hipDeviceptr_t>(src), count);
  };
  SECTION("Peer access enabled") { MemcpyDeviceToDeviceShell<false, true>(f); }
  SECTION("Peer access disabled") { MemcpyDeviceToDeviceShell<false, false>(f); }
}

HIP_TEST_CASE(Unit_hipMemcpyDtoD_Positive_Synchronization_Behavior) {
  // This behavior differs on NVIDIA and AMD, on AMD the hipMemcpy calls is synchronous with
  // respect to the host
#if HT_AMD
  HipTest::HIP_SKIP_TEST(
      "EXSWCPHIPT-127 - Memcpy from device to device memory behavior differs on AMD and Nvidia");
  return;
#endif
  MemcpyDtoDSyncBehavior(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(dst),
                             reinterpret_cast<hipDeviceptr_t>(src), count);
      },
      false);
}

HIP_TEST_CASE(Unit_hipMemcpyDtoD_Negative_Parameters) {
  using namespace std::placeholders;
  LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(dst),
                             reinterpret_cast<hipDeviceptr_t>(src), count);
      },
      dst_alloc.ptr(), src_alloc.ptr(), kPageSize);
}
