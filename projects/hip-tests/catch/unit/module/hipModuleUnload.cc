/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <hip/hip_runtime_api.h>
#include <hip_test_temp_path.hh>

#include <filesystem>
#include <string>
#include <system_error>

HIP_TEST_CASE(Unit_hipModuleUnload_Negative_Module_Is_Nullptr) {
  HIP_CHECK(hipFree(nullptr));

  HIP_CHECK_ERROR(hipModuleUnload(nullptr), hipErrorInvalidResourceHandle);
}

HIP_TEST_CASE(Unit_hipModuleUnload_Negative_Double_Unload) {
  HIP_CHECK(hipFree(nullptr));

  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, "empty_module.code"));
  HIP_CHECK(hipModuleUnload(module));
#if HT_AMD
  HIP_CHECK_ERROR(hipModuleUnload(module), hipErrorNotFound);
#else
  HIP_CHECK_ERROR(hipModuleUnload(module), hipErrorInvalidResourceHandle);
#endif
}
/**
 * @addtogroup hipModuleUnload
 * @{
 * @ingroup ModuleTest
 * `hipError_t hipModuleUnload(hipModule_t module)` -
 * Frees the module
 */

/**
 * Test Description
 * ------------------------
 * - Test case to verify the module release.
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleUnload.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */
HIP_TEST_CASE(Unit_hipModuleUnload_basic) {
  CTX_CREATE();
  constexpr auto fileName = "vcpy_kernel.code";
  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, fileName));
  REQUIRE(module != nullptr);
  HIP_CHECK(hipModuleUnload(module));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 * - Load a code object file, unload the module, then delete the file.
 *   The runtime must not keep the file open after hipModuleUnload.
 *   On Windows an open handle makes the delete fail. On POSIX, unlink of an
 *   open file succeeds, so Linux fd leaks are covered by
 *   Unit_hipModuleLoad_Positive_NoFdLeak instead.
 *   vcpy_kernel.code must stay a native, uncompressed code object (raw ELF or
 *   uncompressed bundle, not SPIR-V): only file-backed code objects take the
 *   fd hand-off path (FatBinaryInfo::AddDevProgram).
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleUnload.cc
 */
HIP_TEST_CASE(Unit_hipModuleUnload_Positive_FileDeletableAfterUnload) {
  CTX_CREATE();
  const std::string path = hip_test::TempPath("hipModuleUnload", ".code");
  std::filesystem::copy_file("vcpy_kernel.code", path);
  // Best-effort cleanup if an assertion below fails.
  struct RemoveOnExit {
    const std::string& p;
    ~RemoveOnExit() {
      std::error_code ec;
      std::filesystem::remove(p, ec);
    }
  } cleanup{path};

  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, path.c_str()));
  REQUIRE(module != nullptr);
  HIP_CHECK(hipModuleUnload(module));

  std::error_code error;
  const bool removed = std::filesystem::remove(path, error);
  INFO("remove error: " << error.message());
  REQUIRE(removed);
  CTX_DESTROY();
}
