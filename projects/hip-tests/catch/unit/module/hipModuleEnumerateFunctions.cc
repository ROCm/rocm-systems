/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip_module_common.hh"
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <vector>

/**
 * @addtogroup hipModuleEnumerateFunctions hipModuleEnumerateFunctions
 * @{
 * @ingroup ModuleTest
 * `hipError_t hipModuleEnumerateFunctions(hipFunction_t* functions, unsigned int numFunctions,
 *                                        hipModule_t mod)`
 * - Returns the function handles within a module
 */

HIP_TEST_CASE(Unit_hipModuleEnumerateFunctions_ZeroMax) {
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, "vcpy_kernel.code"));

  hipFunction_t function = nullptr;
  HIP_CHECK(hipModuleEnumerateFunctions(&function, 0, module));

  HIP_CHECK(hipModuleUnload(module));
}

HIP_TEST_CASE(Unit_hipModuleEnumerateFunctions_PartialFill) {
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, "vcpy_kernel.code"));

  unsigned int count = 0;
  HIP_CHECK(hipModuleGetFunctionCount(&count, module));
  REQUIRE(count >= 1);

  std::vector<hipFunction_t> functions(count + 1, reinterpret_cast<hipFunction_t>(0xDEADBEEF));
  HIP_CHECK(hipModuleEnumerateFunctions(functions.data(), count, module));
  REQUIRE(functions[0] != nullptr);
  REQUIRE(functions[0] != reinterpret_cast<hipFunction_t>(0xDEADBEEF));
  REQUIRE(functions[count] == reinterpret_cast<hipFunction_t>(0xDEADBEEF));

  HIP_CHECK(hipModuleUnload(module));
}

HIP_TEST_CASE(Unit_hipModuleEnumerateFunctions_HandlesUsable) {
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, "vcpy_kernel.code"));

  unsigned int count = 0;
  HIP_CHECK(hipModuleGetFunctionCount(&count, module));
  REQUIRE(count >= 1);

  std::vector<hipFunction_t> functions(count);
  HIP_CHECK(hipModuleEnumerateFunctions(functions.data(), count, module));

  hipFunction_t byName = nullptr;
  HIP_CHECK(hipModuleGetFunction(&byName, module, "hello_world"));
  REQUIRE(byName != nullptr);

  bool found = false;
  for (unsigned int i = 0; i < count; ++i) {
    if (functions[i] == byName) {
      found = true;
      break;
    }
  }
  REQUIRE(found);

  int maxThreads = 0;
  HIP_CHECK(hipFuncGetAttribute(&maxThreads, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                functions[0]));
  REQUIRE(maxThreads > 0);

  HIP_CHECK(hipModuleUnload(module));
}

HIP_TEST_CASE(Unit_hipModuleEnumerateFunctions_Negative) {
  unsigned int count = 1;
  hipFunction_t function = nullptr;
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, "vcpy_kernel.code"));

  SECTION("Null module") {
    HIP_CHECK_ERROR(hipModuleEnumerateFunctions(&function, count, nullptr),
                    hipErrorInvalidValue);
  }
  SECTION("Null functions buffer") {
    HIP_CHECK_ERROR(hipModuleEnumerateFunctions(nullptr, count, module), hipErrorInvalidValue);
  }

  HIP_CHECK(hipModuleUnload(module));
}

/**
 * End doxygen group ModuleTest.
 * @}
 */
