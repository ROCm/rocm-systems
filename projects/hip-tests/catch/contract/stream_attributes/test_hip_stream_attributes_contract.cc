/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr size_t kWindowBytes = 256;

// Skips the test when no device is visible so that the stream attribute
// contracts are only exercised against a provisioned runtime.
void RequireDevice() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
  }
}
}  // namespace

HIP_TEST_CASE(Contract_StreamAttributes_Priority_RoundTrips) {
  RequireDevice();
  hip::contract::ContractCleanup cleanup;

  int least_priority = 0;
  int greatest_priority = 0;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));

  // Stream priority is fixed at creation time; the attribute API exposes it as a
  // read-only value. Create the stream with the greatest (highest) priority,
  // which is a valid value within the reported inclusive range, then read it
  // back through hipStreamGetAttribute.
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamDefault, greatest_priority));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  REQUIRE(stream != nullptr);

  hipStreamAttrValue get_value{};
  const hipError_t get_status =
      hipStreamGetAttribute(stream, hipStreamAttributePriority, &get_value);
  if (get_status == hipErrorNotSupported) {
    // Reading the priority attribute is an optional capability on some
    // backends; an unsupported report is a contract-compliant outcome.
    HIP_SKIP_TEST("hipStreamGetAttribute(priority) is not supported on this device");
  }
  HIP_CHECK(get_status);

  // HIP orders priorities so that greatest (highest) is numerically smaller or
  // equal to least (lowest). The reported priority must be clamped within that
  // inclusive range regardless of any backend-specific coercion.
  REQUIRE(greatest_priority <= get_value.priority);
  REQUIRE(get_value.priority <= least_priority);
}

HIP_TEST_CASE(Contract_StreamAttributes_SyncPolicy_RoundTrips) {
  RequireDevice();
  hip::contract::ContractCleanup cleanup;

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  hipStreamAttrValue set_value{};
  set_value.syncPolicy = hipSyncPolicyBlockingSync;

  const hipError_t set_status =
      hipStreamSetAttribute(stream, hipStreamAttributeSynchronizationPolicy, &set_value);
  if (set_status == hipErrorNotSupported) {
    // Setting the synchronization policy attribute is an optional capability on
    // some backends; an unsupported report is a contract-compliant outcome.
    HIP_SKIP_TEST("hipStreamSetAttribute(syncPolicy) is not supported on this device");
  }
  HIP_CHECK(set_status);

  hipStreamAttrValue get_value{};
  HIP_CHECK(
      hipStreamGetAttribute(stream, hipStreamAttributeSynchronizationPolicy, &get_value));

  // The synchronization policy must round-trip as the exact value that was set;
  // it is a discrete enum that the runtime records verbatim.
  REQUIRE(get_value.syncPolicy == hipSyncPolicyBlockingSync);
}

HIP_TEST_CASE(Contract_StreamAttributes_AccessPolicyWindow_RoundTrips) {
  RequireDevice();
  hip::contract::ContractCleanup cleanup;

  int current_device = -1;
  HIP_CHECK(hipGetDevice(&current_device));

  int max_window_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&max_window_size,
                                  hipDeviceAttributeAccessPolicyMaxWindowSize, current_device));
  if (max_window_size <= 0) {
    // Access policy windows are a device capability; a zero maximum window size
    // means the device does not support this attribute.
    HIP_SKIP_TEST("device does not support access policy windows (max window size is 0)");
  }

  hipStream_t stream = nullptr;
  void* device_ptr = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipMalloc(&device_ptr, kWindowBytes));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });

  hipStreamAttrValue set_value{};
  set_value.accessPolicyWindow.base_ptr = device_ptr;
  set_value.accessPolicyWindow.num_bytes = kWindowBytes;
  set_value.accessPolicyWindow.hitRatio = 0.5f;
  set_value.accessPolicyWindow.hitProp = hipAccessPropertyPersisting;
  set_value.accessPolicyWindow.missProp = hipAccessPropertyNormal;

  const hipError_t set_status =
      hipStreamSetAttribute(stream, hipStreamAttributeAccessPolicyWindow, &set_value);
  if (set_status == hipErrorNotSupported || set_status == hipErrorInvalidValue) {
    // Setting the access policy window attribute is an optional capability; some
    // backends do not accept this attribute id on hipStreamSetAttribute even
    // when a nonzero device window size is reported, and report the attribute as
    // unsupported (hipErrorNotSupported) or reject it as an invalid attribute
    // (hipErrorInvalidValue). Both are contract-compliant unsupported outcomes.
    HIP_SKIP_TEST("hipStreamSetAttribute(accessPolicyWindow) is not supported on this device");
  }
  HIP_CHECK(set_status);

  hipStreamAttrValue get_value{};
  HIP_CHECK(hipStreamGetAttribute(stream, hipStreamAttributeAccessPolicyWindow, &get_value));

  // The access policy window fields must round-trip as they were set.
  REQUIRE(get_value.accessPolicyWindow.base_ptr == device_ptr);
  REQUIRE(get_value.accessPolicyWindow.num_bytes == kWindowBytes);
  REQUIRE(get_value.accessPolicyWindow.hitRatio == 0.5f);
  REQUIRE(get_value.accessPolicyWindow.hitProp == hipAccessPropertyPersisting);
  REQUIRE(get_value.accessPolicyWindow.missProp == hipAccessPropertyNormal);
}

HIP_TEST_CASE(Contract_StreamAttributes_CopyAttributes_PropagatesToDestination) {
  RequireDevice();
  hip::contract::ContractCleanup cleanup;

  hipStream_t src_stream = nullptr;
  hipStream_t dst_stream = nullptr;
  HIP_CHECK(hipStreamCreate(&src_stream));
  cleanup.Add([src_stream] { (void)hipStreamDestroy(src_stream); });
  HIP_CHECK(hipStreamCreate(&dst_stream));
  cleanup.Add([dst_stream] { (void)hipStreamDestroy(dst_stream); });

  hipStreamAttrValue set_value{};
  set_value.syncPolicy = hipSyncPolicyBlockingSync;

  const hipError_t set_status =
      hipStreamSetAttribute(src_stream, hipStreamAttributeSynchronizationPolicy, &set_value);
  if (set_status == hipErrorNotSupported) {
    // Setting the synchronization policy attribute is an optional capability on
    // some backends; an unsupported report is a contract-compliant outcome.
    HIP_SKIP_TEST("hipStreamSetAttribute(syncPolicy) is not supported on this device");
  }
  HIP_CHECK(set_status);

  const hipError_t copy_status = hipStreamCopyAttributes(dst_stream, src_stream);
  if (copy_status == hipErrorNotSupported) {
    // Copying stream attributes is an optional capability on some backends; an
    // unsupported report is a contract-compliant outcome.
    HIP_SKIP_TEST("hipStreamCopyAttributes is not supported on this device");
  }
  HIP_CHECK(copy_status);

  hipStreamAttrValue get_value{};
  HIP_CHECK(hipStreamGetAttribute(dst_stream, hipStreamAttributeSynchronizationPolicy,
                                  &get_value));

  // Copying attributes must propagate the source stream's synchronization
  // policy to the destination stream verbatim.
  REQUIRE(get_value.syncPolicy == hipSyncPolicyBlockingSync);
}

HIP_TEST_CASE(Contract_StreamAttributes_GetAttribute_RejectsInvalidInputs) {
  RequireDevice();

  // BACKEND-DIFF: The invalid-input rejection contracts for hipStreamGetAttribute
  // are only exercised on AMD. On NVIDIA hipStreamGetAttribute maps to
  // cudaStreamGetAttribute, which does not validate the value-out pointer and
  // dereferences it - a null pointer faults instead of returning a defined
  // error - so the rejection contract cannot be evaluated safely there. The
  // unknown-attribute sub-check additionally uses hipLaunchAttributeMax, a
  // sentinel defined only on AMD. Parity would require matching null-argument
  // validation and a portable unknown-attribute sentinel.
#ifdef __HIP_PLATFORM_AMD__
  hip::contract::ContractCleanup cleanup;

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  // A null value-out pointer is a caller error and must be rejected with
  // hipErrorInvalidValue rather than silently succeeding.
  const hipError_t null_out_status =
      hipStreamGetAttribute(stream, hipStreamAttributePriority, nullptr);
  REQUIRE(null_out_status == hipErrorInvalidValue);

  // An attribute id that is not a valid stream attribute must be rejected with
  // hipErrorInvalidValue. hipLaunchAttributeMax is the one-past-the-end sentinel
  // of the launch attribute enum, so it is guaranteed to be an unknown attribute
  // id.
  hipStreamAttrValue get_value{};
  const hipStreamAttrID invalid_attr = static_cast<hipStreamAttrID>(hipLaunchAttributeMax);
  const hipError_t invalid_attr_status =
      hipStreamGetAttribute(stream, invalid_attr, &get_value);
  REQUIRE(invalid_attr_status == hipErrorInvalidValue);
#else
  HIP_SKIP_TEST("hipStreamGetAttribute does not validate the value-out pointer on the NVIDIA "
                "backend; the invalid-input rejection contract cannot be exercised safely.");
#endif  // __HIP_PLATFORM_AMD__
}
