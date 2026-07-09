/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
constexpr char kMissingModuleFile[] = "hip-contract-test-missing-module.code";
constexpr uint32_t kNotAFatbin = 0x12345678;
}  // namespace

HIP_TEST_CASE(Contract_ModuleLoadFile_LoadNullModule_IsRejected) {
  REQUIRE(hipModuleLoad(nullptr, kMissingModuleFile) != hipSuccess);
}

HIP_TEST_CASE(Contract_ModuleLoadFile_LoadNullFilename_IsRejected) {
  hipModule_t module = nullptr;

  REQUIRE(hipModuleLoad(&module, nullptr) != hipSuccess);
}

HIP_TEST_CASE(Contract_ModuleLoadFile_LoadEmptyFilename_IsRejected) {
  hipModule_t module = nullptr;

  REQUIRE(hipModuleLoad(&module, "") != hipSuccess);
}

HIP_TEST_CASE(Contract_ModuleLoadFile_LoadMissingFile_IsRejected) {
  hipModule_t module = nullptr;

  REQUIRE(hipModuleLoad(&module, kMissingModuleFile) != hipSuccess);
}

HIP_TEST_CASE(Contract_ModuleLoadFile_LoadFatBinaryNullFatbin_IsRejected) {
  hipModule_t module = nullptr;

  REQUIRE(hipModuleLoadFatBinary(&module, nullptr) != hipSuccess);
}

HIP_TEST_CASE(Contract_ModuleLoadFile_LoadFatBinaryNullModule_IsRejected) {
  // Use a non-null dummy pointer so this exercises the null module out-parameter.
  REQUIRE(hipModuleLoadFatBinary(nullptr, &kNotAFatbin) != hipSuccess);
}
