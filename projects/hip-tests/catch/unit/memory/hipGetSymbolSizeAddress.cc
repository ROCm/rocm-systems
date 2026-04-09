/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <tuple>

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

namespace {
constexpr size_t kArraySize = 5;
}  // anonymous namespace

#define HIP_GET_SYMBOL_SIZE_ADDRESS_DEFINE_GLOBALS(type)                                           \
  __device__ type type##_var = 0;                                                                  \
  __device__ type type##_arr[kArraySize] = {};                                                     \
  __global__ void type##_var_address_validation_kernel(void* ptr, bool* out) {                     \
    *out = static_cast<void*>(&type##_var) == ptr;                                                 \
  }                                                                                                \
  __global__ void type##_arr_address_validation_kernel(void* ptr, bool* out) {                     \
    *out = static_cast<void*>(type##_arr) == ptr;                                                  \
  }

HIP_GET_SYMBOL_SIZE_ADDRESS_DEFINE_GLOBALS(int)
HIP_GET_SYMBOL_SIZE_ADDRESS_DEFINE_GLOBALS(float)
HIP_GET_SYMBOL_SIZE_ADDRESS_DEFINE_GLOBALS(char)
HIP_GET_SYMBOL_SIZE_ADDRESS_DEFINE_GLOBALS(double)

template <typename T, size_t N, void (*validation_kernel)(void*, bool*)>
static void HipGetSymbolSizeAddressTest(const void* symbol) {
  constexpr auto size = N * sizeof(T);

  T* symbol_ptr = nullptr;
  size_t symbol_size = 0;
  HIP_CHECK(hipGetSymbolAddress(reinterpret_cast<void**>(&symbol_ptr), symbol));
  HIP_CHECK(hipGetSymbolSize(&symbol_size, symbol));
  REQUIRE(symbol_size == size);
  REQUIRE(symbol_ptr != nullptr);

  LinearAllocGuard<bool> equal_addresses(LinearAllocs::hipMalloc, sizeof(bool));
  HIP_CHECK(hipMemset(equal_addresses.ptr(), false, sizeof(*equal_addresses.ptr())))
  validation_kernel<<<1, 1>>>(symbol_ptr, equal_addresses.ptr());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(nullptr));
  bool ok = false;
  HIP_CHECK(hipMemcpy(&ok, equal_addresses.ptr(), sizeof(ok), hipMemcpyDeviceToHost));
  REQUIRE(ok);

  constexpr T expected_value = 42;
  std::array<T, N> fill_buffer;
  std::fill_n(fill_buffer.begin(), N, expected_value);
  HIP_CHECK(hipMemcpy(symbol_ptr, fill_buffer.data(), symbol_size, hipMemcpyHostToDevice));


  std::array<T, N> read_buffer;
  HIP_CHECK(hipMemcpy(read_buffer.data(), symbol_ptr, symbol_size, hipMemcpyDeviceToHost));
  ArrayFindIfNot(read_buffer.data(), expected_value, read_buffer.size());
}

#if HT_AMD
#define SYMBOL(expr) &HIP_SYMBOL(expr)
#else
#define SYMBOL(expr) HIP_SYMBOL(expr)
#endif

#define HIP_GET_SYMBOL_SIZE_ADDRESS_TEST(type)                                                     \
  HipGetSymbolSizeAddressTest<type, 1, type##_var_address_validation_kernel>(SYMBOL(type##_var));  \
  HipGetSymbolSizeAddressTest<type, kArraySize, type##_arr_address_validation_kernel>(             \
      SYMBOL(type##_arr));

HIP_TEST_CASE(Unit_hipGetSymbolSizeAddress_Positive_Basic) {
  SECTION("int") { HIP_GET_SYMBOL_SIZE_ADDRESS_TEST(int); }
  SECTION("float") { HIP_GET_SYMBOL_SIZE_ADDRESS_TEST(float); }
  SECTION("char") { HIP_GET_SYMBOL_SIZE_ADDRESS_TEST(char); }
  SECTION("double") { HIP_GET_SYMBOL_SIZE_ADDRESS_TEST(double); }
}

HIP_TEST_CASE(Unit_hipGetSymbolAddress_Negative_Parameters) {
// Causes a segfault in CUDA
#if HT_AMD
  SECTION("devPtr == nullptr") {
    HIP_CHECK_ERROR(hipGetSymbolAddress(nullptr, SYMBOL(int_var)), hipErrorInvalidValue);
  }
#endif

  SECTION("symbolName == nullptr") {
    void* ptr = nullptr;
    HIP_CHECK_ERROR(hipGetSymbolAddress(&ptr, nullptr), hipErrorInvalidSymbol);
  }
}

HIP_TEST_CASE(Unit_hipGetSymbolSize_Negative_Parameters) {
// Causes a segfault in CUDA
#if HT_AMD
  SECTION("size == nullptr") {
    HIP_CHECK_ERROR(hipGetSymbolSize(nullptr, SYMBOL(int_var)), hipErrorInvalidValue);
  }
#endif

  SECTION("symbolName == nullptr") {
    size_t size = 0;
    HIP_CHECK_ERROR(hipGetSymbolSize(&size, nullptr), hipErrorInvalidSymbol);
  }
}
