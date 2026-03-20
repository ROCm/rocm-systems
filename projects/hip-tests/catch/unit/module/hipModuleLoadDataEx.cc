/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip_module_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>


HIP_TEST_CASE(Unit_hipModuleLoadDataEx_Positive_Basic) {
  HIP_CHECK(hipFree(nullptr));
  hipModule_t module = nullptr;

  SECTION("Load compiled module from file") {
    const auto loaded_module = LoadModuleIntoBuffer("empty_module.code");
    HIP_CHECK(hipModuleLoadDataEx(&module, loaded_module.data(), 0, nullptr, nullptr));
    REQUIRE(module != nullptr);
    HIP_CHECK(hipModuleUnload(module));
  }

  SECTION("Load RTCd module") {
    const auto rtc = CreateRTCCharArray(R"(extern "C" __global__ void kernel() {})");
    HIP_CHECK(hipModuleLoadDataEx(&module, rtc.data(), 0, nullptr, nullptr));
    REQUIRE(module != nullptr);
    HIP_CHECK(hipModuleUnload(module));
  }
}

HIP_TEST_CASE(Unit_hipModuleLoadDataEx_Negative_Parameters) {
  HIP_CHECK(hipFree(nullptr));
  hipModule_t module = nullptr;

  SECTION("module == nullptr") {
    const auto loaded_module = LoadModuleIntoBuffer("empty_module.code");
    HIP_CHECK_ERROR(hipModuleLoadDataEx(nullptr, loaded_module.data(), 0, nullptr, nullptr),
                    hipErrorInvalidValue);
    LoadModuleIntoBuffer("empty_module.code");
  }

  SECTION("image == nullptr") {
    HIP_CHECK_ERROR(hipModuleLoadDataEx(&module, nullptr, 0, nullptr, nullptr),
                    hipErrorInvalidValue);
  }
}
