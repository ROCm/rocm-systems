/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Regression coverage for the LibraryContainer→DynCO refactor that landed
// alongside hipLibraryGetGlobal / hipLibraryGetManaged. The refactor:
//   * replaced LibraryContainer's own (FatBinaryInfo + functions_ map) with a
//     std::unique_ptr<hip::DynCO>;
//   * routes hipModuleGetGlobal and hipLibraryGetGlobal through the same
//     DynCO::GetGlobal entry point;
//   * rewrote LibraryContainer::Kernel / EnumerateKernels / KernelCount to
//     delegate to DynCO methods;
//   * uses an atomic_bool with double-checked locking in BuildIt().
//
// These tests exercise the surface area most likely to regress under those
// changes. They use the offline-compiled library_code_load.code so the
// kernel/global counts are stable.

#include <hip_test_common.hh>

#include <string>
#include <vector>

namespace {
// 9 user-visible kernels (3 arithmetic + 6 d_var/m_var write/read/read_modify
// from library_code_load.cc) plus runtime-injected helpers for the
// __device__ / __managed__ variable initialization. The exact total is
// runtime-version-dependent; checking ">= 9" is more robust than a literal.
constexpr unsigned int kMinKernelCount = 9;
const std::string kCodeFile = "library_code_load.code";
}  // namespace

// EnumerateKernels: maxKernels == 0 must be a no-op success (no out-of-bounds
// write). Easy bug to introduce if a refactor switches from <= to <.
HIP_TEST_CASE(Unit_LibraryRefactor_EnumerateKernels_ZeroMax) {
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                   0));

  // Sentinel — must remain untouched.
  hipKernel_t k = reinterpret_cast<hipKernel_t>(0xDEADBEEF);
  HIP_CHECK(hipLibraryEnumerateKernels(&k, 0, lib));
  REQUIRE(k == reinterpret_cast<hipKernel_t>(0xDEADBEEF));

  HIP_CHECK(hipLibraryUnload(lib));
}

// EnumerateKernels: enumerate exactly KernelCount user-callable kernels and
// verify the trailing guard slot isn't touched. Originally this test would
// SIGABRT inside Function::BuildKernel because DynCO::populateDynGlobalFuncs
// surfaced C++ ABI symbols (e.g., __cxa_deleted_virtual) that the runtime's
// findSymbol() couldn't resolve. Fixed in hip_code_object.cpp by filtering
// the function list against amd::Program::findSymbol() at population time.
HIP_TEST_CASE(Unit_LibraryRefactor_EnumerateKernels_PartialFill) {
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                   0));

  unsigned int count = 0;
  HIP_CHECK(hipLibraryGetKernelCount(&count, lib));
  REQUIRE(count >= kMinKernelCount);

  const size_t alloc = static_cast<size_t>(count) + 1;
  std::vector<hipKernel_t> ks(alloc, reinterpret_cast<hipKernel_t>(0xDEADBEEF));

  HIP_CHECK(hipLibraryEnumerateKernels(ks.data(), count, lib));

  for (unsigned int i = 0; i < count; ++i) {
    INFO("filled slot " << i);
    REQUIRE(ks[i] != reinterpret_cast<hipKernel_t>(0xDEADBEEF));
    REQUIRE(ks[i] != nullptr);
  }
  INFO("guard slot at index " << count);
  REQUIRE(ks[count] == reinterpret_cast<hipKernel_t>(0xDEADBEEF));

  HIP_CHECK(hipLibraryUnload(lib));
}

// Every enumerated handle must resolve to a hipFunction_t — catches refactor
// bugs where Kernel() and EnumerateKernels() get out of sync about the
// kernel cache (kernels_ keyed by name+device).
HIP_TEST_CASE(Unit_LibraryRefactor_EnumerateKernels_HandlesUsable) {
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                   0));

  unsigned int count = 0;
  HIP_CHECK(hipLibraryGetKernelCount(&count, lib));
  REQUIRE(count >= kMinKernelCount);

  std::vector<hipKernel_t> ks(count, nullptr);
  HIP_CHECK(hipLibraryEnumerateKernels(ks.data(), count, lib));

  for (auto k : ks) {
    REQUIRE(k != nullptr);
    hipFunction_t hf = nullptr;
    HIP_CHECK(hipKernelGetFunction(&hf, k));
    REQUIRE(hf != nullptr);
  }

  HIP_CHECK(hipLibraryUnload(lib));
}

// Lazy build path: the first user-visible API call drives BuildIt(); a
// subsequent call must be cheap and idempotent (atomic double-check). Hammer
// it from a single thread to catch any "second build runs and corrupts state"
// bugs without adding threading complexity to the test.
HIP_TEST_CASE(Unit_LibraryRefactor_BuildIt_Idempotent) {
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                   0));

  unsigned int first = 0;
  HIP_CHECK(hipLibraryGetKernelCount(&first, lib));
  REQUIRE(first >= kMinKernelCount);

  for (int i = 0; i < 64; ++i) {
    unsigned int n = 0;
    HIP_CHECK(hipLibraryGetKernelCount(&n, lib));
    REQUIRE(n == first);
  }

  // Same kernel name must always resolve to the same handle (cached in the
  // kernels_ map keyed by (name, device)).
  hipKernel_t a = nullptr, b = nullptr;
  HIP_CHECK(hipLibraryGetKernel(&a, lib, "add_kernel"));
  HIP_CHECK(hipLibraryGetKernel(&b, lib, "add_kernel"));
  REQUIRE(a == b);

  HIP_CHECK(hipLibraryUnload(lib));
}

// Cross-API regression: hipModuleGetGlobal and hipLibraryGetGlobal now share
// DynCO::GetGlobal. Loading the same code object via both entry points and
// looking up the same global should produce equivalent results (same size,
// non-null pointer; pointers may differ since each is a separate load).
HIP_TEST_CASE(Unit_LibraryRefactor_ModuleVsLibrary_GetGlobalParity) {
  // Module path
  hipModule_t mod = nullptr;
  HIP_CHECK(hipModuleLoad(&mod, kCodeFile.c_str()));
  hipDeviceptr_t mod_dptr = nullptr;
  size_t mod_bytes = 0;
  HIP_CHECK(hipModuleGetGlobal(&mod_dptr, &mod_bytes, mod, "d_var"));

  // Library path
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                   0));
  void* lib_dptr = nullptr;
  size_t lib_bytes = 0;
  HIP_CHECK(hipLibraryGetGlobal(&lib_dptr, &lib_bytes, lib, "d_var"));

  REQUIRE(mod_dptr != nullptr);
  REQUIRE(lib_dptr != nullptr);
  REQUIRE(mod_bytes == lib_bytes);
  REQUIRE(mod_bytes == sizeof(float) * 32);

  HIP_CHECK(hipLibraryUnload(lib));
  HIP_CHECK(hipModuleUnload(mod));
}

// Ordering / cache-coherency regression: hipLibraryGetKernel writes into the
// kernels_ cache; hipLibraryGetGlobal touches DynCO::vars_. The two must not
// step on each other regardless of the order they're invoked in. Mirrors the
// HIPRTC-side Unit_hipLibraryGetKernel_OrderingWithGetGlobal test but uses
// the offline-compiled artifact so we cover both code paths.
HIP_TEST_CASE(Unit_LibraryRefactor_Ordering_KernelThenGlobalThenKernel) {
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                   0));

  hipKernel_t k1 = nullptr;
  HIP_CHECK(hipLibraryGetKernel(&k1, lib, "add_kernel"));

  void* dptr = nullptr;
  size_t bytes = 0;
  HIP_CHECK(hipLibraryGetGlobal(&dptr, &bytes, lib, "d_var"));
  REQUIRE(dptr != nullptr);

  hipKernel_t k2 = nullptr;
  HIP_CHECK(hipLibraryGetKernel(&k2, lib, "add_kernel"));
  REQUIRE(k1 == k2);  // cache must still resolve identically

  unsigned int count = 0;
  HIP_CHECK(hipLibraryGetKernelCount(&count, lib));
  REQUIRE(count >= kMinKernelCount);

  HIP_CHECK(hipLibraryUnload(lib));
}

// Load → Unload → Load: a common pattern that surfaces lifetime bugs in the
// DynCO destruction path (which now also has to release managed-memory
// allocations registered for __managed__ vars).
HIP_TEST_CASE(Unit_LibraryRefactor_Reload_Lifecycle) {
  for (int iter = 0; iter < 4; ++iter) {
    INFO("iteration " << iter);
    hipLibrary_t lib = nullptr;
    HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr,
                                     nullptr, 0));
    void* dptr = nullptr;
    size_t bytes = 0;
    HIP_CHECK(hipLibraryGetGlobal(&dptr, &bytes, lib, "d_var"));
    REQUIRE(dptr != nullptr);
    REQUIRE(bytes == sizeof(float) * 32);
    HIP_CHECK(hipLibraryUnload(lib));
  }
}
