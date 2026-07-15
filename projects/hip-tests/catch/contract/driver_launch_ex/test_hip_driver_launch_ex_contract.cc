/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstring>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

// Driver-style extended launch and SM-resource group-split contracts.
//
// hipLaunchKernelExC launches a kernel through a driver-style hipLaunchConfig_t
// and is exercised as a functional round-trip: a single-thread kernel writes a
// value through a device pointer and the value is read back after the launch.
//
// hipDevSmResourceSplit partitions a device's SM resource into caller-sized groups
// (as opposed to the count-based hipDevSmResourceSplitByCount covered by the green
// context domain); the produced group must contain a positive SM count bounded by
// the whole-device SM count.
//
// Both are gated to discrete GPUs. On integrated devices these device-resource and
// driver-launch paths are not meaningfully exercised, so the tests skip.

namespace {
int CurrentDevice() {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  return device;
}

void SkipIfIntegratedDevice() {
  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, CurrentDevice()));
  if (props.integrated != 0) {
    HIP_SKIP_TEST("driver-style extended launch and SM resource split are only "
                  "exercised on discrete GPUs.");
  }
}

// Owns a device allocation and frees it on destruction, so a failing REQUIRE
// (which throws and unwinds) does not leak the buffer into sibling tests.
class ScopedDeviceInt {
 public:
  explicit ScopedDeviceInt(int initial_value) {
    HIP_CHECK(hipMalloc(&ptr_, sizeof(int)));
    HIP_CHECK(hipMemset(ptr_, initial_value, sizeof(int)));
  }
  ~ScopedDeviceInt() {
    if (ptr_ != nullptr) {
      (void)hipFree(ptr_);
    }
  }
  ScopedDeviceInt(const ScopedDeviceInt&) = delete;
  ScopedDeviceInt& operator=(const ScopedDeviceInt&) = delete;

  int* get() { return ptr_; }
  int** address() { return &ptr_; }

 private:
  int* ptr_ = nullptr;
};

__global__ void WriteValueKernel(int* out, int value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = value;
  }
}
}  // namespace

// hipLaunchKernelExC submits a kernel via a driver-style launch configuration.
// The launched kernel must publish its argument value, observable after a device
// synchronize.
HIP_TEST_CASE(Contract_DriverLaunchEx_LaunchKernelExC_WritesExpectedValue) {
  SkipIfIntegratedDevice();

  ScopedDeviceInt device_value(0);

  constexpr int kExpected = 0x5A5A;
  int value = kExpected;
  void* args[] = {device_value.address(), &value};

  hipLaunchConfig_t config{};
  config.gridDim = dim3(1);
  config.blockDim = dim3(1);
  config.dynamicSmemBytes = 0;
  config.stream = nullptr;
  config.attrs = nullptr;
  config.numAttrs = 0;

  const hipError_t status =
      hipLaunchKernelExC(&config, reinterpret_cast<const void*>(WriteValueKernel), args);
  if (status == hipErrorNotSupported) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipLaunchKernelExC is not supported by this runtime path.");
  }
  HIP_CHECK(status);
  HIP_CHECK(hipDeviceSynchronize());

  int observed = -1;
  HIP_CHECK(hipMemcpy(&observed, device_value.get(), sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(observed == kExpected);
}

// hipDevSmResourceSplit partitions the device SM resource into caller-specified
// groups. Requesting one group must yield a group whose SM count is within
// (0, device SM count]; the split cannot invent SMs the device lacks.
HIP_TEST_CASE(Contract_DriverLaunchEx_DevSmResourceSplit_ProducesBoundedGroup) {
  SkipIfIntegratedDevice();

  hipDevResource device_resource{};
  const hipError_t query_status =
      hipDeviceGetDevResource(CurrentDevice(), &device_resource, hipDevResourceTypeSm);
  if (query_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Device resource queries are not supported by this runtime path.");
  }
  HIP_CHECK(query_status);
  REQUIRE(device_resource.type == hipDevResourceTypeSm);
  REQUIRE(device_resource.sm.smCount > 0);

  hipDevResource group{};
  hipDevResource remainder{};
  hipDevSmResourceGroupParams params{};
  params.smCount = 1;

  const hipError_t status =
      hipDevSmResourceSplit(&group, 1, &device_resource, &remainder, 0, &params);
  // The group-parameter split is a configuration some backends do not accept:
  // RDNA GPUs support the count-based split (hipDevSmResourceSplitByCount) but
  // reject the group-params variant with hipErrorInvalidResourceConfiguration,
  // while all CDNA arches accept it. Treat both "not supported" and "invalid
  // resource configuration" as a capability skip rather than a contract
  // violation; the split is only asserted where the backend accepts the config.
  if (status == hipErrorNotSupported || status == hipErrorInvalidResourceConfiguration) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("SM resource group splitting is not supported by this runtime path.");
  }
  HIP_CHECK(status);

  REQUIRE(group.sm.smCount > 0);
  REQUIRE(group.sm.smCount <= device_resource.sm.smCount);
}
