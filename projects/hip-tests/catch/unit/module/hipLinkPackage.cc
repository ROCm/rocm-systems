/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Coverage for llvm-offload-binary packages fed to the JIT linker, the branch
// LinkProgram::AddLinkerDataImpl takes when CheckIfPackaged() is true. The two halves of that
// branch are reachable through different entry points: packaged device bitcode only through
// hiprtcLinkAdd* (hipLinkAddData/hipLinkAddFile reject hipJitInputLLVMBundledBitcode outright),
// and packaged SPIR-V only through hipLinkAdd* (hiprtcLinkAdd* reject HIPRTC_JIT_INPUT_SPIRV).
//
// The bitcode entries declare the plain arch with no sramecc/xnack qualifiers, so on a
// feature-qualified agent they have to be selected as compatible rather than as an exact name
// match. The SPIR-V entry is keyed on the fixed (spirv64-amd-amdhsa-unknown, amdgcnspirv) target
// the linker hard-codes for hipJitInputSpirv.

#if HT_AMD

namespace {

constexpr char kPackagedBitcode[] = "addKernel-package.bc";
constexpr char kPackagedMismatchBitcode[] = "addKernel-mismatch-package.bc";
constexpr char kPackagedSpirv[] = "addKernel-package.spv";

constexpr int kArraySize = 4;
constexpr int kRefValue = 5;
constexpr int kKernelIncrement = 2;

// Returns false (and skips the calling test) when the toolchain shipped no offload packager, in
// which case the fixtures were not built. Mirrors hipModuleLoadPackage.cc.
bool FixtureMissing(const char* path) {
  if (std::filesystem::exists(path)) {
    return false;
  }
  HIP_SKIP_TEST("Fixture was not built: the toolchain ships no llvm-offload-binary packager");
  return true;
}

std::vector<char> ReadFixture(const char* path) {
  std::ifstream stream{path, std::ios_base::in | std::ios_base::binary};
  REQUIRE(stream.good());
  return std::vector<char>{std::istreambuf_iterator<char>(stream),
                           std::istreambuf_iterator<char>()};
}

// Launches addKernel out of a freshly linked image and checks every element was incremented.
void LaunchAndVerifyAddKernel(void* linked_image) {
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoadData(&module, linked_image));

  hipFunction_t kernel = nullptr;
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "addKernel"));

  int* device_array = nullptr;
  HIP_CHECK(hipMalloc(&device_array, kArraySize * sizeof(int)));

  int host_array[kArraySize];
  for (int i = 0; i < kArraySize; ++i) {
    host_array[i] = kRefValue + i;
  }
  HIP_CHECK(hipMemcpy(device_array, host_array, sizeof(host_array), hipMemcpyHostToDevice));

  int size = kArraySize;
  void* args[] = {&device_array, &size};
  HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, kArraySize, 1, 1, 0, nullptr, args, nullptr));

  HIP_CHECK(hipMemcpy(host_array, device_array, sizeof(host_array), hipMemcpyDeviceToHost));
  for (int i = 0; i < kArraySize; ++i) {
    REQUIRE(host_array[i] == kRefValue + i + kKernelIncrement);
  }

  HIP_CHECK(hipFree(device_array));
  HIP_CHECK(hipModuleUnload(module));
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Links device bitcode delivered as an llvm-offload-binary package through the hiprtc JIT
 *    linker, then loads and launches the result. Covers both hiprtcLinkAddData and
 *    hiprtcLinkAddFile, which reach the same packaged-input branch.
 *
 * Test source
 * ------------------------
 *  - unit/module/hipLinkPackage.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
HIP_TEST_CASE(Unit_hiprtcLinkAdd_Positive_PackagedBitcode) {
  if (FixtureMissing(kPackagedBitcode)) {
    return;
  }

  hiprtcLinkState link_state = nullptr;
  HIPRTC_CHECK(hiprtcLinkCreate(0, nullptr, nullptr, &link_state));

  SECTION("hiprtcLinkAddData") {
    std::vector<char> package = ReadFixture(kPackagedBitcode);
    HIPRTC_CHECK(hiprtcLinkAddData(link_state, HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE,
                                   package.data(), package.size(), "PackagedBitcode", 0, nullptr,
                                   nullptr));
  }

  SECTION("hiprtcLinkAddFile") {
    HIPRTC_CHECK(hiprtcLinkAddFile(link_state, HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE,
                                   kPackagedBitcode, 0, nullptr, nullptr));
  }

  void* linked_image = nullptr;
  size_t linked_size = 0;
  HIPRTC_CHECK(hiprtcLinkComplete(link_state, &linked_image, &linked_size));
  REQUIRE(linked_size > 0);

  LaunchAndVerifyAddKernel(linked_image);

  HIPRTC_CHECK(hiprtcLinkDestroy(link_state));
}

/**
 * Test Description
 * ------------------------
 *  - Passes a well-formed package holding bitcode for an unrelated architecture to the hiprtc JIT
 *    linker. The unpackage action finds no entry for the device, so adding the input has to fail
 *    rather than silently linking nothing.
 *
 * Test source
 * ------------------------
 *  - unit/module/hipLinkPackage.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
HIP_TEST_CASE(Unit_hiprtcLinkAdd_Negative_PackagedBitcodeNoCompatibleEntry) {
  if (FixtureMissing(kPackagedMismatchBitcode)) {
    return;
  }

  // The fixture is built for gfx900 only, which is a compatible entry on a gfx900 agent.
  hipDeviceProp_t properties;
  HIP_CHECK(hipGetDeviceProperties(&properties, 0));
  if (std::string(properties.gcnArchName).rfind("gfx900", 0) == 0) {
    HIP_SKIP_TEST("The deliberately-mismatched package is compatible with a gfx900 device");
    return;
  }

  std::vector<char> package = ReadFixture(kPackagedMismatchBitcode);

  hiprtcLinkState link_state = nullptr;
  HIPRTC_CHECK(hiprtcLinkCreate(0, nullptr, nullptr, &link_state));

  HIPRTC_CHECK_ERROR(hiprtcLinkAddData(link_state, HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE,
                                       package.data(), package.size(), "PackagedBitcode", 0,
                                       nullptr, nullptr),
                     HIPRTC_ERROR_PROGRAM_CREATION_FAILURE);

  HIPRTC_CHECK(hiprtcLinkDestroy(link_state));
}

#if !defined(_WIN32)
/**
 * Test Description
 * ------------------------
 *  - Links SPIR-V delivered as an llvm-offload-binary package through the hip JIT linker, then
 *    loads and launches the result. This is the hipJitInputSpirv half of the packaged-input
 *    branch, which selects the fixed (spirv64-amd-amdhsa-unknown, amdgcnspirv) entry rather than
 *    the device ISA.
 *
 * Test source
 * ------------------------
 *  - unit/module/hipLinkPackage.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
HIP_TEST_CASE(Unit_hipLinkAddData_Positive_PackagedSpirv) {
  if (FixtureMissing(kPackagedSpirv)) {
    return;
  }

  std::vector<char> package = ReadFixture(kPackagedSpirv);

  hipLinkState_t link_state = nullptr;
  HIP_CHECK(hipLinkCreate(0, nullptr, nullptr, &link_state));

  HIP_CHECK(hipLinkAddData(link_state, hipJitInputSpirv, package.data(), package.size(),
                           "PackagedSpirv", 0, nullptr, nullptr));

  void* linked_image = nullptr;
  size_t linked_size = 0;
  HIP_CHECK(hipLinkComplete(link_state, &linked_image, &linked_size));
  REQUIRE(linked_size > 0);

  LaunchAndVerifyAddKernel(linked_image);

  HIP_CHECK(hipLinkDestroy(link_state));
}
#endif  // !defined(_WIN32)

#endif  // HT_AMD
