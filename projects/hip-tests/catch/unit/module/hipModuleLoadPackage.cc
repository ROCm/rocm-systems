/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

#include <filesystem>
#include <string>

// Happy-path coverage for the llvm-offload-binary package container.
//
// package_module.code holds one entry per arch the tests were built for, each declaring the plain
// arch with no sramecc/xnack qualifiers, the way a --offload-arch=gfx90a build names it. On an
// agent whose ISA does carry those qualifiers that is a compatible-but-not-exact match, which is
// the case a naive exact-name lookup silently drops, so this is the interesting direction to test.

namespace {

constexpr char kPackageModule[] = "package_module.code";
constexpr char kPackageMismatchModule[] = "package_mismatch_module.code";

bool FixtureMissing(const char* path) {
  if (std::filesystem::exists(path)) {
    return false;
  }
  HIP_SKIP_TEST("Fixture was not built: the toolchain ships no llvm-offload-binary packager");
  return true;
}

}  // namespace

HIP_TEST_CASE(Unit_hipModuleLoad_Positive_Package) {
  if (FixtureMissing(kPackageModule)) {
    return;
  }

  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, kPackageModule));
  REQUIRE(module != nullptr);

  hipFunction_t kernel = nullptr;
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "Kernel42"));
  REQUIRE(kernel != nullptr);

  int* result = nullptr;
  HIP_CHECK(hipMalloc(&result, sizeof(int)));
  HIP_CHECK(hipMemset(result, 0, sizeof(int)));

  void* args[] = {&result};
  HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
  HIP_CHECK(hipDeviceSynchronize());

  int host = 0;
  HIP_CHECK(hipMemcpy(&host, result, sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(host == 42);

  HIP_CHECK(hipFree(result));
  HIP_CHECK(hipModuleUnload(module));
}

// A structurally valid package holding no entry this device can run. The fixture is built for
// gfx900 only (see CMakeLists.txt), so it skips on a gfx900 runner the same way
// Unit_FatBin_NoCompatibleCodeObject_ReturnsInvalidImage does.
HIP_TEST_CASE(Unit_hipModuleLoad_Negative_PackageNoCompatibleCodeObject) {
  if (FixtureMissing(kPackageMismatchModule)) {
    return;
  }

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  if (std::string(prop.gcnArchName).rfind("gfx900", 0) == 0) {
    HIP_SKIP_TEST("Fixture is built for gfx900; current device matches it");
    return;
  }

  hipModule_t module = nullptr;
  // Matches the bundle path: an image whose entries are all incompatible is reported as an
  // invalid image, not as hipErrorNoBinaryForGpu. Both are accepted for the same reason
  // Unit_FatBin_NoCompatibleCodeObject_ReturnsInvalidImage accepts both.
  HIP_CHECK_ERRORS(hipModuleLoad(&module, kPackageMismatchModule), hipErrorInvalidImage,
                   hipErrorNoBinaryForGpu);
}
