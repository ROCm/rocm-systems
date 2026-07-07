/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
constexpr size_t kRangeBytes = sizeof(int);

bool ManagedMemorySupported() {
  void* ptr = nullptr;
  const hipError_t status = hipMallocManaged(&ptr, kRangeBytes, hipMemAttachGlobal);
  if (status == hipSuccess) {
    HIP_CHECK(hipFree(ptr));
    return true;
  }
  if (status == hipErrorNotSupported || status == hipErrorOutOfMemory) {
    return false;
  }
  HIP_CHECK(status);
  return false;
}

void SkipIfManagedMemoryUnsupported() {
  if (!ManagedMemorySupported()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kManagedMemoryUnsupported);
  }
}

// Applies advice and skips (rather than fails) when the runtime does not support
// the advice path at all. Returns true when advice was accepted.
bool ApplyAdviseOrSkip(const void* dev_ptr, size_t count, hipMemoryAdvise advice, int device) {
  const hipError_t status = hipMemAdvise(dev_ptr, count, advice, device);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}

// Queries a single range attribute and skips (rather than fails) when the query
// path is unsupported. Returns true when the query succeeded.
bool QueryRangeAttributeOrSkip(void* data, size_t data_size, hipMemRangeAttribute attribute,
                               const void* dev_ptr, size_t count) {
  const hipError_t status =
      hipMemRangeGetAttribute(data, data_size, attribute, dev_ptr, count);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}
}  // namespace

HIP_TEST_CASE(Contract_MemAdvise_SetReadMostly_RangeAttributeReflectsAdvice) {
  SkipIfManagedMemoryUnsupported();
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  int* data = nullptr;
  HIP_CHECK(hipMallocManaged(&data, kRangeBytes, hipMemAttachGlobal));

  if (!ApplyAdviseOrSkip(data, kRangeBytes, hipMemAdviseSetReadMostly, device)) {
    HIP_CHECK(hipFree(data));
    HIP_SKIP_TEST("hipMemAdviseSetReadMostly is not supported by this runtime path.");
  }

  int read_mostly = 0;
  if (!QueryRangeAttributeOrSkip(&read_mostly, sizeof(read_mostly),
                                 hipMemRangeAttributeReadMostly, data, kRangeBytes)) {
    HIP_CHECK(hipFree(data));
    HIP_SKIP_TEST("hipMemRangeAttributeReadMostly query is not supported by this runtime path.");
  }

  if (read_mostly == 0) {
    HIP_CHECK(hipFree(data));
    HIP_SKIP_TEST(
        "hipMemAdviseSetReadMostly succeeded but the range attribute did not report the advice as "
        "observable on this device/runtime path.");
  }

  REQUIRE(read_mostly != 0);

  // Undo the advice and confirm the observable state clears when supported.
  if (ApplyAdviseOrSkip(data, kRangeBytes, hipMemAdviseUnsetReadMostly, device)) {
    int cleared = 1;
    if (QueryRangeAttributeOrSkip(&cleared, sizeof(cleared), hipMemRangeAttributeReadMostly, data,
                                  kRangeBytes)) {
      REQUIRE(cleared == 0);
    }
  }

  HIP_CHECK(hipFree(data));
}

HIP_TEST_CASE(Contract_MemAdvise_SetPreferredLocation_RangeAttributeReturnsDevice) {
  SkipIfManagedMemoryUnsupported();
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  int* data = nullptr;
  HIP_CHECK(hipMallocManaged(&data, kRangeBytes, hipMemAttachGlobal));

  if (!ApplyAdviseOrSkip(data, kRangeBytes, hipMemAdviseSetPreferredLocation, device)) {
    HIP_CHECK(hipFree(data));
    HIP_SKIP_TEST("hipMemAdviseSetPreferredLocation is not supported by this runtime path.");
  }

  int preferred = -1;
  if (!QueryRangeAttributeOrSkip(&preferred, sizeof(preferred),
                                 hipMemRangeAttributePreferredLocation, data, kRangeBytes)) {
    HIP_CHECK(hipFree(data));
    HIP_SKIP_TEST(
        "hipMemRangeAttributePreferredLocation query is not supported by this runtime path.");
  }

  if (preferred != device) {
    HIP_CHECK(hipFree(data));
    HIP_SKIP_TEST(
        "hipMemAdviseSetPreferredLocation succeeded but the range attribute did not report the "
        "current device as preferred location on this device/runtime path.");
  }

  REQUIRE(preferred == device);

  // Undo the advice and confirm the preferred location is no longer the current
  // device when supported. Avoid pinning a specific unset sentinel value.
  if (ApplyAdviseOrSkip(data, kRangeBytes, hipMemAdviseUnsetPreferredLocation, device)) {
    int unset = device;
    if (QueryRangeAttributeOrSkip(&unset, sizeof(unset), hipMemRangeAttributePreferredLocation,
                                  data, kRangeBytes)) {
      REQUIRE(unset != device);
    }
  }

  HIP_CHECK(hipFree(data));
}

HIP_TEST_CASE(Contract_MemAdvise_SetAccessedBy_RangeAttributeReflectsDevice) {
  SkipIfManagedMemoryUnsupported();
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  int* data = nullptr;
  HIP_CHECK(hipMallocManaged(&data, kRangeBytes, hipMemAttachGlobal));

  if (!ApplyAdviseOrSkip(data, kRangeBytes, hipMemAdviseSetAccessedBy, device)) {
    HIP_CHECK(hipFree(data));
    HIP_SKIP_TEST("hipMemAdviseSetAccessedBy is not supported by this runtime path.");
  }

  int accessed_by[1] = {-1};
  if (!QueryRangeAttributeOrSkip(accessed_by, sizeof(accessed_by),
                                 hipMemRangeAttributeAccessedBy, data, kRangeBytes)) {
    HIP_CHECK(hipFree(data));
    HIP_SKIP_TEST("hipMemRangeAttributeAccessedBy query is not supported by this runtime path.");
  }

  if (accessed_by[0] != device) {
    HIP_CHECK(hipFree(data));
    HIP_SKIP_TEST(
        "hipMemAdviseSetAccessedBy succeeded but the range attribute did not list the current "
        "device on this device/runtime path.");
  }

  REQUIRE(accessed_by[0] == device);

  HIP_CHECK(hipFree(data));
}

HIP_TEST_CASE(Contract_MemAdvise_RangeGetAttributes_MultipleAttributes_Succeed) {
  SkipIfManagedMemoryUnsupported();
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  int* data = nullptr;
  HIP_CHECK(hipMallocManaged(&data, kRangeBytes, hipMemAttachGlobal));

  const bool read_mostly_set =
      ApplyAdviseOrSkip(data, kRangeBytes, hipMemAdviseSetReadMostly, device);
  const bool preferred_set =
      ApplyAdviseOrSkip(data, kRangeBytes, hipMemAdviseSetPreferredLocation, device);
  if (!read_mostly_set && !preferred_set) {
    HIP_CHECK(hipFree(data));
    HIP_SKIP_TEST("hipMemAdvise is not supported by this runtime path.");
  }

  int read_mostly = 0;
  int preferred = -1;
  void* results[2] = {&read_mostly, &preferred};
  size_t result_sizes[2] = {sizeof(read_mostly), sizeof(preferred)};
  hipMemRangeAttribute attributes[2] = {hipMemRangeAttributeReadMostly,
                                        hipMemRangeAttributePreferredLocation};

  const hipError_t status =
      hipMemRangeGetAttributes(results, result_sizes, attributes, 2, data, kRangeBytes);
  if (status == hipErrorNotSupported) {
    HIP_CHECK(hipFree(data));
    HIP_SKIP_TEST("hipMemRangeGetAttributes is not supported by this runtime path.");
  }
  HIP_CHECK(status);

  bool observable = false;
  if (read_mostly_set && read_mostly != 0) {
    REQUIRE(read_mostly != 0);
    observable = true;
  }
  if (preferred_set && preferred == device) {
    REQUIRE(preferred == device);
    observable = true;
  }

  if (!observable) {
    HIP_CHECK(hipFree(data));
    HIP_SKIP_TEST(
        "hipMemRangeGetAttributes succeeded but neither advice was reported as observable on this "
        "device/runtime path.");
  }

  HIP_CHECK(hipFree(data));
}

HIP_TEST_CASE(Contract_MemAdvise_NullPointer_IsRejected) {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  const hipError_t status =
      hipMemAdvise(nullptr, sizeof(int), hipMemAdviseSetReadMostly, device);

  REQUIRE(status != hipSuccess);
}

HIP_TEST_CASE(Contract_MemAdvise_RangeGetAttribute_NullData_IsRejected) {
  SkipIfManagedMemoryUnsupported();

  int* data = nullptr;
  HIP_CHECK(hipMallocManaged(&data, kRangeBytes, hipMemAttachGlobal));

  const hipError_t status = hipMemRangeGetAttribute(
      nullptr, sizeof(int), hipMemRangeAttributeReadMostly, data, sizeof(int));

  REQUIRE(status != hipSuccess);

  HIP_CHECK(hipFree(data));
}
