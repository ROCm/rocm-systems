/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Tests for hipLibraryGetModule against the ahead-of-time-compiled code object
// (library_code_load.code, built from library_code_load.cc by the CMake custom
// target).
//

#include <hip_test_common.hh>

#include <string>
#include <vector>

namespace {

constexpr size_t kArrLen = 32;
const std::string kCodeFile = "library_code_load.code";

}  // namespace

// The handle must be non-null and stable across repeated queries. Callers cache
// it, and the runtime registers it exactly once.
HIP_TEST_CASE(Unit_hipLibraryGetModule_Positive_Basic) {
  HIP_TEST_DRIVER_INIT();
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                   0));

  hipModule_t mod = nullptr;
  HIP_CHECK(hipLibraryGetModule(&mod, lib));
  REQUIRE(mod != nullptr);

  hipModule_t mod_again = nullptr;
  HIP_CHECK(hipLibraryGetModule(&mod_again, lib));
  REQUIRE(mod_again == mod);

  HIP_CHECK(hipLibraryUnload(lib));
}

// hipLibraryGetModule as the first call on a fresh library must trigger the
// lazy build itself rather than handing back a module for an unbuilt library.
HIP_TEST_CASE(Unit_hipLibraryGetModule_Positive_BuildsOnFirstUse) {
  HIP_TEST_DRIVER_INIT();
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                   0));

  // No hipLibraryGetKernel/GetGlobal first. this is the only query.
  hipModule_t mod = nullptr;
  HIP_CHECK(hipLibraryGetModule(&mod, lib));
  REQUIRE(mod != nullptr);

  hipFunction_t func = nullptr;
  HIP_CHECK(hipModuleGetFunction(&func, mod, "add_kernel"));
  REQUIRE(func != nullptr);

  HIP_CHECK(hipLibraryUnload(lib));
}

// The whole reason the API exists is that the returned module must be usable with the
// module entry points, and a function resolved that way must actually launch.
HIP_TEST_CASE(Unit_hipLibraryGetModule_Positive_ModuleGetFunctionAndLaunch) {
  CTX_CREATE();
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                   0));

  hipModule_t mod = nullptr;
  HIP_CHECK(hipLibraryGetModule(&mod, lib));
  REQUIRE(mod != nullptr);

  hipFunction_t func = nullptr;
  HIP_CHECK(hipModuleGetFunction(&func, mod, "add_kernel"));
  REQUIRE(func != nullptr);

  float* d_out = nullptr;
  float* d_a = nullptr;
  float* d_b = nullptr;
  const size_t bytes = kArrLen * sizeof(float);
  HIP_CHECK(hipMalloc(&d_out, bytes));
  HIP_CHECK(hipMalloc(&d_a, bytes));
  HIP_CHECK(hipMalloc(&d_b, bytes));

  std::vector<float> h_a(kArrLen), h_b(kArrLen);
  for (size_t i = 0; i < kArrLen; ++i) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(2 * i);
  }
  HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));

  void* args[] = {&d_out, &d_a, &d_b};
  HIP_CHECK(hipModuleLaunchKernel(func, 1, 1, 1, kArrLen, 1, 1, 0, nullptr, args, nullptr));
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> h_out(kArrLen, 0.0f);
  HIP_CHECK(hipMemcpy(h_out.data(), d_out, bytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < kArrLen; ++i) {
    INFO("index " << i);
    REQUIRE(h_out[i] == h_a[i] + h_b[i]);
  }

  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipLibraryUnload(lib));
  CTX_DESTROY();
}

// A global resolved through the library and through its module must agree.
// both are backed by the same loaded code object, so the size must match and
// neither address may be null.
HIP_TEST_CASE(Unit_hipLibraryGetModule_Positive_ModuleGetGlobalParity) {
  HIP_TEST_DRIVER_INIT();
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                   0));

  void* lib_address = nullptr;
  size_t lib_bytes = 0;
  HIP_CHECK(hipLibraryGetGlobal(&lib_address, &lib_bytes, lib, "d_var"));
  REQUIRE(lib_address != nullptr);
  REQUIRE(lib_bytes == sizeof(float) * kArrLen);

  hipModule_t mod = nullptr;
  HIP_CHECK(hipLibraryGetModule(&mod, lib));

  hipDeviceptr_t mod_address = 0;
  size_t mod_bytes = 0;
  HIP_CHECK(hipModuleGetGlobal(&mod_address, &mod_bytes, mod, "d_var"));
  REQUIRE(mod_address != 0);
  REQUIRE(mod_bytes == lib_bytes);
  // Same library, same load. the two paths must resolve the same storage.
  REQUIRE(reinterpret_cast<void*>(mod_address) == lib_address);

  HIP_CHECK(hipLibraryUnload(lib));
}

// A kernel reached via hipLibraryGetKernel and the function reached via the
// module must denote the same device function.
HIP_TEST_CASE(Unit_hipLibraryGetModule_Positive_MatchesLibraryGetKernel) {
  HIP_TEST_DRIVER_INIT();
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                   0));

  hipKernel_t kernel = nullptr;
  HIP_CHECK(hipLibraryGetKernel(&kernel, lib, "mul_kernel"));
  hipFunction_t from_kernel = nullptr;
  HIP_CHECK(hipKernelGetFunction(&from_kernel, kernel));

  hipModule_t mod = nullptr;
  HIP_CHECK(hipLibraryGetModule(&mod, lib));
  hipFunction_t from_module = nullptr;
  HIP_CHECK(hipModuleGetFunction(&from_module, mod, "mul_kernel"));

  REQUIRE(from_module == from_kernel);

  HIP_CHECK(hipLibraryUnload(lib));
}

// The module is owned by the library. Releasing it through hipModuleUnload
// would leave the library holding a freed code object, so the runtime must
// refuse. CUDA reports CUDA_ERROR_NOT_PERMITTED here, which has no HIP
// equivalent, so only the AMD path pins the exact error code.
HIP_TEST_CASE(Unit_hipLibraryGetModule_Negative_ModuleUnloadRefused) {
  HIP_TEST_DRIVER_INIT();
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                   0));

  hipModule_t mod = nullptr;
  HIP_CHECK(hipLibraryGetModule(&mod, lib));
  REQUIRE(mod != nullptr);

#ifdef __HIP_PLATFORM_AMD__
  HIP_CHECK_ERROR(hipModuleUnload(mod), hipErrorIllegalState);
#else
  REQUIRE(hipModuleUnload(mod) != hipSuccess);
#endif

  // Refusing must be non-destructive. The library is still fully usable.
  hipFunction_t func = nullptr;
  HIP_CHECK(hipModuleGetFunction(&func, mod, "sub_kernel"));
  REQUIRE(func != nullptr);

  // And the library still owns it, so this is the call that frees it.
  HIP_CHECK(hipLibraryUnload(lib));
}

// Unloading the library must revoke the module registration rather than leave
// a dangling entry behind for the next lookup to walk into.
HIP_TEST_CASE(Unit_hipLibraryGetModule_Negative_StaleAfterLibraryUnload) {
  HIP_TEST_DRIVER_INIT();
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                   0));

  hipModule_t mod = nullptr;
  HIP_CHECK(hipLibraryGetModule(&mod, lib));
  REQUIRE(mod != nullptr);

  HIP_CHECK(hipLibraryUnload(lib));

  hipFunction_t func = nullptr;
  REQUIRE(hipModuleGetFunction(&func, mod, "add_kernel") != hipSuccess);
}

HIP_TEST_CASE(Unit_hipLibraryGetModule_Negative_Parameters) {
  HIP_TEST_DRIVER_INIT();
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                   0));

  hipModule_t mod = nullptr;

  SECTION("null module out-param") {
    HIP_CHECK_ERROR(hipLibraryGetModule(nullptr, lib), hipErrorInvalidValue);
  }
  SECTION("null library") {
    HIP_CHECK_ERROR(hipLibraryGetModule(&mod, nullptr), hipErrorInvalidResourceHandle);
  }

  HIP_CHECK(hipLibraryUnload(lib));
}
