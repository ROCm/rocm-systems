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

// @asserts: hipModuleLoad - rejects a null module out-parameter with a non-success error
HIP_TEST_CASE(Contract_ModuleLoadFile_LoadNullModule_IsRejected) {
  REQUIRE(hipModuleLoad(nullptr, kMissingModuleFile) != hipSuccess);
}

// @asserts: hipModuleLoad - rejects a null filename with a non-success error
HIP_TEST_CASE(Contract_ModuleLoadFile_LoadNullFilename_IsRejected) {
  hipModule_t module = nullptr;

  REQUIRE(hipModuleLoad(&module, nullptr) != hipSuccess);
}

// @asserts: hipModuleLoad - rejects an empty filename string with a non-success error
HIP_TEST_CASE(Contract_ModuleLoadFile_LoadEmptyFilename_IsRejected) {
  hipModule_t module = nullptr;

  REQUIRE(hipModuleLoad(&module, "") != hipSuccess);
}

// @asserts: hipModuleLoad - rejects a filename that does not exist on disk with a non-success error
HIP_TEST_CASE(Contract_ModuleLoadFile_LoadMissingFile_IsRejected) {
  hipModule_t module = nullptr;

  REQUIRE(hipModuleLoad(&module, kMissingModuleFile) != hipSuccess);
}

// @asserts: hipModuleLoadFatBinary - rejects a null fat-binary image pointer with a non-success error
HIP_TEST_CASE(Contract_ModuleLoadFile_LoadFatBinaryNullFatbin_IsRejected) {
  hipModule_t module = nullptr;

  REQUIRE(hipModuleLoadFatBinary(&module, nullptr) != hipSuccess);
}

// @asserts: hipModuleLoadFatBinary - rejects a null module out-parameter (with a non-null image) with a non-success error
HIP_TEST_CASE(Contract_ModuleLoadFile_LoadFatBinaryNullModule_IsRejected) {
  // Use a non-null dummy pointer so this exercises the null module out-parameter.
  REQUIRE(hipModuleLoadFatBinary(nullptr, &kNotAFatbin) != hipSuccess);
}
