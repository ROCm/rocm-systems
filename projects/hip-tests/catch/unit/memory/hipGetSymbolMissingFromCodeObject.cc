/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup hipGetSymbolAddress hipGetSymbolAddress
 * @{
 * @ingroup MemoryTest
 */

// The tests use a symbol that only the host compilation pass has, so they exist only there
#if !defined(__HIP_DEVICE_COMPILE__)
// Registered by the application, but missing from the code object loaded on the device
__device__ int host_registered_var;

/**
 * Test Description
 * ------------------------
 *  - A __device__ variable that the application registers but the loaded code object doesn't
 *    contain must be reported as hipErrorInvalidSymbol, not abort the process.
 * Test source
 * ------------------------
 *  - unit/memory/hipGetSymbolMissingFromCodeObject.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.14
 */
HIP_TEST_CASE(Unit_hipGetSymbolAddress_Negative_MissingFromCodeObject) {
  // An unregistered symbol gets the same error; a runtime that looks the variable up in the
  // code object and doesn't handle the miss aborts here instead
  void* ptr = nullptr;
  HIP_CHECK_ERROR(hipGetSymbolAddress(&ptr, HIP_SYMBOL(host_registered_var)),
                  hipErrorInvalidSymbol);
  size_t size = 0;
  HIP_CHECK_ERROR(hipGetSymbolSize(&size, HIP_SYMBOL(host_registered_var)),
                  hipErrorInvalidSymbol);
}
#endif

/**
 * End doxygen group MemoryTest.
 * @}
 */
