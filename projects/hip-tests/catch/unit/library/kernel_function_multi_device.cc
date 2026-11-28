/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include <vector>

#if !HT_NVIDIA || CUDA_VERSION >= CUDA_12080

namespace {
constexpr int kElements = 32;

bool HasTwoDevices() {
  int count = 0;
  HIP_CHECK(hipGetDeviceCount(&count));
  return count >= 2;
}

hipError_t LaunchAdd(int device, hipFunction_t function, unsigned int shared_bytes,
                     std::vector<float>* result) {
  HIP_CHECK(hipSetDevice(device));
  std::vector<float> ones(kElements, 1.0f);
  float* out = nullptr;
  float* in = nullptr;
  HIP_CHECK(hipMalloc(&out, kElements * sizeof(float)));
  HIP_CHECK(hipMalloc(&in, kElements * sizeof(float)));
  HIP_CHECK(hipMemcpy(in, ones.data(), kElements * sizeof(float), hipMemcpyHostToDevice));
  void* args[] = {&out, &in, &in};
  const hipError_t status =
      hipModuleLaunchKernel(function, 1, 1, 1, kElements, 1, 1, shared_bytes, nullptr, args, nullptr);
  (void)hipGetLastError();
  if (status == hipSuccess && result != nullptr) {
    result->resize(kElements);
    HIP_CHECK(hipMemcpy(result->data(), out, kElements * sizeof(float), hipMemcpyDeviceToHost));
  }
  HIP_CHECK(hipFree(out));
  HIP_CHECK(hipFree(in));
  return status;
}
}  // namespace

HIP_TEST_CASE(Unit_hipKernelGetFunction_MultiDevice_LaunchesOnEachDevice) {
  if (!HasTwoDevices()) {
    HIP_SKIP_TEST("Requires at least two devices");
  }
  HIP_CHECK(hipSetDevice(0));
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&library, "library_code_load.code", nullptr, nullptr, 0, nullptr,
                                   nullptr, 0));
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, "add_kernel"));

  constexpr int kLimit = 1024;
  HIP_CHECK(hipKernelSetAttribute(HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kLimit, kernel,
                                  1));
  HIP_CHECK(hipKernelSetCacheConfig(kernel, hipFuncCachePreferL1, 1));

  hipFunction_t functions[2] = {};
  for (int device = 0; device < 2; ++device) {
    HIP_CHECK(hipSetDevice(device));
    HIP_CHECK(hipKernelGetFunction(&functions[device], kernel));
  }
  REQUIRE(functions[0] != functions[1]);

  for (int device = 0; device < 2; ++device) {
    std::vector<float> result;
    HIP_CHECK(LaunchAdd(device, functions[device], 0, &result));
    for (float value : result) {
      REQUIRE(value == 2.0f);
    }
  }
  REQUIRE(LaunchAdd(1, functions[1], kLimit, nullptr) == hipSuccess);
  REQUIRE(LaunchAdd(1, functions[1], kLimit + 1, nullptr) == hipErrorInvalidValue);
  REQUIRE(LaunchAdd(0, functions[0], kLimit + 1, nullptr) == hipSuccess);

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Unit_hipKernelGetFunction_MultiDevice_SharesManagedVariable) {
  if (!HasTwoDevices()) {
    HIP_SKIP_TEST("Requires at least two devices");
  }
  HIP_CHECK(hipSetDevice(0));
  hipLibrary_t library = nullptr;
  hipKernel_t reader = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&library, "library_code_load.code", nullptr, nullptr, 0, nullptr,
                                   nullptr, 0));
  HIP_CHECK(hipLibraryGetKernel(&reader, library, "read_m_var"));

  void* managed = nullptr;
  size_t bytes = 0;
  HIP_CHECK(hipLibraryGetManaged(&managed, &bytes, library, "m_var"));
  REQUIRE(bytes == kElements * sizeof(float));
  auto* values = static_cast<float*>(managed);
  for (int i = 0; i < kElements; ++i) {
    values[i] = static_cast<float>(i);
  }

  for (int device = 0; device < 2; ++device) {
    HIP_CHECK(hipSetDevice(device));
    hipFunction_t function = nullptr;
    HIP_CHECK(hipKernelGetFunction(&function, reader));
    float* out = nullptr;
    HIP_CHECK(hipMalloc(&out, kElements * sizeof(float)));
    void* args[] = {&out};
    HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, kElements, 1, 1, 0, nullptr, args, nullptr));
    std::vector<float> result(kElements);
    HIP_CHECK(hipMemcpy(result.data(), out, kElements * sizeof(float), hipMemcpyDeviceToHost));
    for (int i = 0; i < kElements; ++i) {
      REQUIRE(result[i] == static_cast<float>(i + 1));
    }
    HIP_CHECK(hipFree(out));
  }

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Unit_hipLibrary_MultiDevice_RawKernelLaunch) {
  if (!HasTwoDevices()) {
    HIP_SKIP_TEST("Requires at least two devices");
  }
  HIP_CHECK(hipSetDevice(0));
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&library, "library_code_load.code", nullptr, nullptr, 0, nullptr,
                                   nullptr, 0));
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, "add_kernel"));

  for (int device = 0; device < 2; ++device) {
    std::vector<float> result;
    HIP_CHECK(LaunchAdd(device, reinterpret_cast<hipFunction_t>(kernel), 0, &result));
    for (float value : result) {
      REQUIRE(value == 2.0f);
    }
  }

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Unit_hipLibraryGetGlobal_MultiDevice_PerDeviceStorage) {
  if (!HasTwoDevices()) {
    HIP_SKIP_TEST("Requires at least two devices");
  }
  HIP_CHECK(hipSetDevice(0));
  hipLibrary_t library = nullptr;
  hipKernel_t reader = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&library, "library_code_load.code", nullptr, nullptr, 0, nullptr,
                                   nullptr, 0));
  HIP_CHECK(hipLibraryGetKernel(&reader, library, "read_d_var"));

  void* globals[2] = {};
  for (int device = 0; device < 2; ++device) {
    HIP_CHECK(hipSetDevice(device));
    size_t bytes = 0;
    HIP_CHECK(hipLibraryGetGlobal(&globals[device], &bytes, library, "d_var"));
    REQUIRE(bytes == kElements * sizeof(float));
    std::vector<float> values(kElements, 10.0f * (device + 1));
    HIP_CHECK(hipMemcpy(globals[device], values.data(), bytes, hipMemcpyHostToDevice));
  }
  REQUIRE(globals[0] != globals[1]);

  for (int device = 0; device < 2; ++device) {
    HIP_CHECK(hipSetDevice(device));
    hipFunction_t function = nullptr;
    HIP_CHECK(hipKernelGetFunction(&function, reader));
    float* out = nullptr;
    HIP_CHECK(hipMalloc(&out, kElements * sizeof(float)));
    void* args[] = {&out};
    HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, kElements, 1, 1, 0, nullptr, args, nullptr));
    std::vector<float> result(kElements);
    HIP_CHECK(hipMemcpy(result.data(), out, kElements * sizeof(float), hipMemcpyDeviceToHost));
    for (float value : result) {
      REQUIRE(value == 10.0f * (device + 1) + 1.0f);
    }
    HIP_CHECK(hipFree(out));
  }

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Unit_hipLibraryGetModule_MultiDevice_MatchesKernelFunction) {
  if (!HasTwoDevices()) {
    HIP_SKIP_TEST("Requires at least two devices");
  }
  HIP_CHECK(hipSetDevice(0));
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&library, "library_code_load.code", nullptr, nullptr, 0, nullptr,
                                   nullptr, 0));
  HIP_CHECK(hipLibraryGetKernel(&kernel, library, "mul_kernel"));

  hipModule_t modules[2] = {};
  for (int device = 0; device < 2; ++device) {
    HIP_CHECK(hipSetDevice(device));
    HIP_CHECK(hipLibraryGetModule(&modules[device], library));
    hipFunction_t from_module = nullptr;
    hipFunction_t from_kernel = nullptr;
    HIP_CHECK(hipModuleGetFunction(&from_module, modules[device], "mul_kernel"));
    HIP_CHECK(hipKernelGetFunction(&from_kernel, kernel));
    REQUIRE(from_module == from_kernel);
  }
  REQUIRE(modules[0] != modules[1]);

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipLibraryUnload(library));
}
#endif
