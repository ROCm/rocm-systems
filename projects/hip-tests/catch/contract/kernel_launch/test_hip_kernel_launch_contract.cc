/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

#if HT_AMD
#include <hip/hip_ext.h>
#endif

// The symbol contracts resolve device globals through the public symbol query
// APIs. Those APIs look the symbols up by their registered external name, so the
// device globals (and the kernels that keep them live in the device image) must
// have external linkage and therefore live at file scope rather than inside an
// anonymous namespace.
constexpr int kContractSymbolArraySize = 8;
__device__ int g_contract_symbol_scalar = 0;
__device__ int g_contract_symbol_array[kContractSymbolArraySize] = {};

// Tiny kernels that reference the device globals so the compiler keeps the
// symbols live in the device image, making them resolvable through the public
// symbol query APIs.
__global__ void TouchSymbolScalarKernel() {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    g_contract_symbol_scalar = g_contract_symbol_scalar + 1;
  }
}

__global__ void TouchSymbolArrayKernel() {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    g_contract_symbol_array[0] = g_contract_symbol_array[0] + 1;
  }
}

namespace {
constexpr int kExpectedValue = 0x1234;

// A tiny in-source kernel that publishes a value through a device pointer so the
// cooperative, extended, and AMD extension launch contracts share one launchable
// symbol without any external per-arch code-object fixture.
__global__ void WriteValueKernel(int* output, int value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *output = value;
  }
}

int ReadDeviceInt(int* device_ptr) {
  int value = 0;
  HIP_CHECK(hipMemcpy(&value, device_ptr, sizeof(value), hipMemcpyDeviceToHost));
  return value;
}

// Reports whether the current device advertises cooperative launch support so the
// cooperative-launch contracts can skip cleanly on paths that lack it.
bool CooperativeLaunchSupported() {
  int current_device = 0;
  HIP_CHECK(hipGetDevice(&current_device));
  int cooperative_launch = 0;
  HIP_CHECK(hipDeviceGetAttribute(&cooperative_launch, hipDeviceAttributeCooperativeLaunch,
                                  current_device));
  return cooperative_launch != 0;
}

// Portable symbol-reference idiom. The HIP C++ hipGetSymbolAddress/Size
// overloads take the device global by reference (const T&) and apply the address
// operator internally, so the correct argument on both backends is the bare
// global via HIP_SYMBOL; taking its address here would double-address it and fail
// symbol lookup.
#define CONTRACT_SYMBOL(expr) HIP_SYMBOL(expr)
}  // namespace

HIP_TEST_CASE(Contract_KernelLaunch_CooperativeKernel_WritesExpectedValue) {
  if (!CooperativeLaunchSupported()) {
    HIP_SKIP_TEST("This device does not support cooperative kernel launch.");
  }

  int* device_value = nullptr;
  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  HIP_CHECK(hipMemset(device_value, 0, sizeof(*device_value)));

  // A cooperative launch of the host-function pointer with a single-thread grid
  // must execute and publish the expected value deterministically.
  int value = kExpectedValue;
  void* kernel_args[] = {&device_value, &value};
  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<const void*>(WriteValueKernel), dim3(1),
                                       dim3(1), kernel_args, 0, nullptr));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(ReadDeviceInt(device_value) == kExpectedValue);

  HIP_CHECK(hipFree(device_value));
}

HIP_TEST_CASE(Contract_KernelLaunch_CooperativeKernel_NullFunction_IsRejected) {
  if (!CooperativeLaunchSupported()) {
    HIP_SKIP_TEST("This device does not support cooperative kernel launch.");
  }

  // Launching a cooperative kernel with a null function pointer must not silently
  // succeed. The exact error code is backend-specific, so only a non-success
  // status is required. The null is typed as const void* so the plain (non
  // template) launch overload is selected.
  const void* null_function = nullptr;
  const hipError_t status =
      hipLaunchCooperativeKernel(null_function, dim3(1), dim3(1), nullptr, 0, nullptr);
  REQUIRE(status != hipSuccess);
}

HIP_TEST_CASE(Contract_KernelLaunch_GetSymbolAddress_ReturnsUsableDevicePointer) {
  // Launch a kernel that references the device global so the symbol is emitted
  // and resolvable in the device image on every runtime path.
  hipLaunchKernelGGL(TouchSymbolScalarKernel, dim3(1), dim3(1), 0, 0);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // Resolving a device global through the public symbol API must yield a non-null
  // device pointer that behaves like any other device allocation for copies.
  int* symbol_ptr = nullptr;
  HIP_CHECK(hipGetSymbolAddress(reinterpret_cast<void**>(&symbol_ptr),
                                CONTRACT_SYMBOL(g_contract_symbol_scalar)));
  REQUIRE(symbol_ptr != nullptr);

  const int written = kExpectedValue;
  HIP_CHECK(hipMemcpy(symbol_ptr, &written, sizeof(written), hipMemcpyHostToDevice));

  REQUIRE(ReadDeviceInt(symbol_ptr) == kExpectedValue);
}

HIP_TEST_CASE(Contract_KernelLaunch_GetSymbolSize_MatchesDeclaredSize) {
  // Launch a kernel that references the device global array so the symbol is
  // emitted and resolvable in the device image on every runtime path.
  hipLaunchKernelGGL(TouchSymbolArrayKernel, dim3(1), dim3(1), 0, 0);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // The reported size of a device global array must match its declared byte size.
  size_t symbol_size = 0;
  HIP_CHECK(hipGetSymbolSize(&symbol_size, CONTRACT_SYMBOL(g_contract_symbol_array)));
  REQUIRE(symbol_size == sizeof(g_contract_symbol_array));
}

HIP_TEST_CASE(Contract_KernelLaunch_GetSymbolAddress_NullSymbol_IsRejected) {
  // Resolving a null symbol must not silently succeed. The exact error code is
  // backend-specific, so only a non-success status is required.
  void* symbol_ptr = nullptr;
  const hipError_t status = hipGetSymbolAddress(&symbol_ptr, nullptr);
  REQUIRE(status != hipSuccess);
}

HIP_TEST_CASE(Contract_KernelLaunch_LaunchKernelEx_WritesExpectedValue) {
  int* device_value = nullptr;
  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  HIP_CHECK(hipMemset(device_value, 0, sizeof(*device_value)));

  // A minimal extended-launch configuration (single-thread grid, no dynamic
  // shared memory, default stream, no attributes) must execute the kernel and
  // publish the expected value deterministically.
  hipLaunchConfig_t config = {};
  config.gridDim = dim3(1);
  config.blockDim = dim3(1);
  config.dynamicSmemBytes = 0;
  config.stream = nullptr;
  config.attrs = nullptr;
  config.numAttrs = 0;

  HIP_CHECK(hipLaunchKernelEx(&config, WriteValueKernel, device_value, kExpectedValue));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(ReadDeviceInt(device_value) == kExpectedValue);

  HIP_CHECK(hipFree(device_value));
}

#if HT_AMD
HIP_TEST_CASE(Contract_KernelLaunch_ExtLaunchKernel_WritesExpectedValue) {
  int* device_value = nullptr;
  int value = kExpectedValue;
  void* kernel_args[] = {&device_value, &value};

  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  HIP_CHECK(hipMemset(device_value, 0, sizeof(*device_value)));

  // The AMD extended launch entry point (no start/stop events, no flags) must
  // execute the kernel and publish the expected value deterministically.
  HIP_CHECK(hipExtLaunchKernel(reinterpret_cast<const void*>(WriteValueKernel), dim3(1), dim3(1),
                               kernel_args, 0, nullptr, nullptr, nullptr, 0));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(ReadDeviceInt(device_value) == kExpectedValue);

  HIP_CHECK(hipFree(device_value));
}
#endif  // HT_AMD
