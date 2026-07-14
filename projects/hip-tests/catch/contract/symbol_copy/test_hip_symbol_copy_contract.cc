/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <cstring>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

// The symbol copy contracts resolve device globals through the public symbol
// copy APIs. Those APIs look the symbols up by their registered external name,
// so the device globals (and the kernels that keep them live in the device
// image) must have external linkage and therefore live at file scope rather
// than inside an anonymous namespace.
constexpr int kContractSymbolArraySize = 8;
__device__ int g_contract_symbol_copy_scalar = 0;
__device__ int g_contract_symbol_copy_array[kContractSymbolArraySize] = {};

// Tiny kernels that reference the device globals so the compiler keeps the
// symbols live in the device image, making them resolvable through the public
// symbol copy APIs.
__global__ void TouchSymbolCopyScalarKernel() {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    g_contract_symbol_copy_scalar = g_contract_symbol_copy_scalar + 1;
  }
}

__global__ void TouchSymbolCopyArrayKernel() {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    g_contract_symbol_copy_array[0] = g_contract_symbol_copy_array[0] + 1;
  }
}

namespace {
constexpr int kSentinel = 0x1234;

// Skips the test when no device is visible so that the symbol copy contracts are
// only exercised against a provisioned runtime.
void RequireDevice() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
  }
}

// Launches the touch kernels and synchronizes so the device globals are emitted
// and resolvable in the device image on every runtime path before any symbol
// copy operation runs against them.
void TouchAndSyncSymbols() {
  hipLaunchKernelGGL(TouchSymbolCopyScalarKernel, dim3(1), dim3(1), 0, 0);
  HIP_CHECK(hipGetLastError());
  hipLaunchKernelGGL(TouchSymbolCopyArrayKernel, dim3(1), dim3(1), 0, 0);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
}
}  // namespace

HIP_TEST_CASE(Contract_SymbolCopy_ToSymbol_ThenFromSymbol_RoundTripsScalar) {
  RequireDevice();
  TouchAndSyncSymbols();

  // Writing a sentinel to a scalar device global through hipMemcpyToSymbol and
  // reading it back through hipMemcpyFromSymbol must round-trip the value.
  const int written = kSentinel;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_contract_symbol_copy_scalar), &written,
                              sizeof(written), 0, hipMemcpyHostToDevice));

  int read_back = 0;
  HIP_CHECK(hipMemcpyFromSymbol(&read_back, HIP_SYMBOL(g_contract_symbol_copy_scalar),
                                sizeof(read_back), 0, hipMemcpyDeviceToHost));

  REQUIRE(read_back == kSentinel);
}

HIP_TEST_CASE(Contract_SymbolCopy_ToSymbolWithOffset_WritesAtElementOffset) {
  RequireDevice();
  TouchAndSyncSymbols();

  // Zero the whole array so the offset write is the only change.
  int zeros[kContractSymbolArraySize] = {};
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_contract_symbol_copy_array), zeros, sizeof(zeros), 0,
                              hipMemcpyHostToDevice));

  // Writing a single element at a byte offset of 2*sizeof(int) must land at
  // element index 2 and leave every other element untouched.
  const int written = kSentinel;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_contract_symbol_copy_array), &written, sizeof(written),
                              2 * sizeof(int), hipMemcpyHostToDevice));

  int read_back[kContractSymbolArraySize] = {};
  HIP_CHECK(hipMemcpyFromSymbol(read_back, HIP_SYMBOL(g_contract_symbol_copy_array),
                                sizeof(read_back), 0, hipMemcpyDeviceToHost));

  for (int i = 0; i < kContractSymbolArraySize; ++i) {
    if (i == 2) {
      REQUIRE(read_back[i] == kSentinel);
    } else {
      REQUIRE(read_back[i] == 0);
    }
  }
}

HIP_TEST_CASE(Contract_SymbolCopy_FromSymbol_DefaultDirection_ReadsBytes) {
  RequireDevice();
  TouchAndSyncSymbols();

  // Seed the scalar global directly through its resolved device address so the
  // read path is exercised independently of hipMemcpyToSymbol.
  int* symbol_ptr = nullptr;
  HIP_CHECK(hipGetSymbolAddress(reinterpret_cast<void**>(&symbol_ptr),
                                HIP_SYMBOL(g_contract_symbol_copy_scalar)));
  REQUIRE(symbol_ptr != nullptr);

  const int written = kSentinel;
  HIP_CHECK(hipMemcpy(symbol_ptr, &written, sizeof(written), hipMemcpyHostToDevice));

  // Reading through hipMemcpyFromSymbol with the default direction must return
  // the seeded bytes.
  int read_back = 0;
  HIP_CHECK(hipMemcpyFromSymbol(&read_back, HIP_SYMBOL(g_contract_symbol_copy_scalar),
                                sizeof(read_back)));

  REQUIRE(read_back == kSentinel);
}

HIP_TEST_CASE(Contract_SymbolCopy_ToSymbolAsync_FromSymbolAsync_RoundTripsInStreamOrder) {
  RequireDevice();
  TouchAndSyncSymbols();
  hip::contract::ContractCleanup cleanup;

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });

  // A stream-ordered write followed by a stream-ordered read on the same stream
  // must round-trip the value once the stream is synchronized.
  const int written = kSentinel;
  HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(g_contract_symbol_copy_scalar), &written,
                                   sizeof(written), 0, hipMemcpyHostToDevice, stream));

  int read_back = 0;
  HIP_CHECK(hipMemcpyFromSymbolAsync(&read_back, HIP_SYMBOL(g_contract_symbol_copy_scalar),
                                     sizeof(read_back), 0, hipMemcpyDeviceToHost, stream));

  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(read_back == kSentinel);
}

HIP_TEST_CASE(Contract_SymbolCopy_NullSymbol_IsRejected) {
  RequireDevice();

  // Copying to or from a null symbol must not silently succeed. The exact error
  // code is backend-specific, so only a non-success status is required. The null
  // is typed as const void* so the non-template C API overload is selected;
  // a bare nullptr would bind to the C++ template (T = std::nullptr_t), which
  // passes the address of a non-null temporary and would not exercise the
  // null-symbol contract.
  const void* null_symbol = nullptr;
  int value = kSentinel;

  const hipError_t to_status =
      hipMemcpyToSymbol(null_symbol, &value, sizeof(value), 0, hipMemcpyHostToDevice);
  REQUIRE(to_status != hipSuccess);

  const hipError_t from_status =
      hipMemcpyFromSymbol(&value, null_symbol, sizeof(value), 0, hipMemcpyDeviceToHost);
  REQUIRE(from_status != hipSuccess);
}

HIP_TEST_CASE(Contract_SymbolCopy_OutOfBoundsSizePlusOffset_IsRejected) {
  RequireDevice();
  TouchAndSyncSymbols();

  // A copy whose offset plus size exceeds the declared symbol size must not
  // silently succeed. The exact error code is backend-specific, so only a
  // non-success status is required.
  const size_t symbol_bytes = sizeof(g_contract_symbol_copy_array);
  int value = kSentinel;
  const hipError_t status =
      hipMemcpyToSymbol(HIP_SYMBOL(g_contract_symbol_copy_array), &value, sizeof(value),
                        symbol_bytes, hipMemcpyHostToDevice);
  REQUIRE(status != hipSuccess);
}
