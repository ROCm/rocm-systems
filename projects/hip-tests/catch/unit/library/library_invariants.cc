/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// API-level invariants for the hipLibrary* surface, exercised against the
// offline-compiled library_code_load.code so kernel/global counts are stable.
// Covers EnumerateKernels boundary behavior, repeated-query stability,
// hipModuleGetGlobal/hipLibraryGetGlobal parity, ordering between kernel and
// global lookups, and load/unload lifecycle.

#include <hip_test_common.hh>

#include <string>
#include <vector>

namespace {
// User-visible kernels defined in library_code_load.cc. Runtime may inject
// additional helper kernels for __device__ / __managed__ initialization, so
// assert a lower bound rather than an exact count.
constexpr unsigned int kMinKernelCount = 9;
const std::string kCodeFile = "library_code_load.code";
}  // namespace

// maxKernels == 0 must succeed without writing to the output buffer.
HIP_TEST_CASE(Unit_hipLibraryEnumerateKernels_ZeroMax) {
  HIP_TEST_DRIVER_INIT();
  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                   0));

  hipKernel_t k = reinterpret_cast<hipKernel_t>(0xDEADBEEF);
  HIP_CHECK(hipLibraryEnumerateKernels(&k, 0, lib));
  REQUIRE(k == reinterpret_cast<hipKernel_t>(0xDEADBEEF));

  HIP_CHECK(hipLibraryUnload(lib));
}

// Enumerating exactly KernelCount slots fills every slot with a non-null
// handle and leaves a trailing guard slot untouched.
HIP_TEST_CASE(Unit_hipLibraryEnumerateKernels_PartialFill) {
  HIP_TEST_DRIVER_INIT();
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

// Every enumerated handle must resolve to a hipFunction_t via
// hipKernelGetFunction.
HIP_TEST_CASE(Unit_hipLibraryEnumerateKernels_HandlesUsable) {
  CTX_CREATE();
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
  CTX_DESTROY();
}

// Repeated queries on the same library must be stable: KernelCount returns
// the same value, and GetKernel for the same name returns the same handle.
HIP_TEST_CASE(Unit_hipLibrary_RepeatedQueriesAreStable) {
  HIP_TEST_DRIVER_INIT();
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

  hipKernel_t a = nullptr, b = nullptr;
  HIP_CHECK(hipLibraryGetKernel(&a, lib, "add_kernel"));
  HIP_CHECK(hipLibraryGetKernel(&b, lib, "add_kernel"));
  REQUIRE(a == b);

  HIP_CHECK(hipLibraryUnload(lib));
}

// Looking up the same global through hipModuleGetGlobal and
// hipLibraryGetGlobal must report identical sizes and non-null pointers
// (pointers themselves may differ since each is a separate load).
HIP_TEST_CASE(Unit_hipLibraryGetGlobal_MatchesModuleGetGlobal) {
  CTX_CREATE();
  hipModule_t mod = nullptr;
  HIP_CHECK(hipModuleLoad(&mod, kCodeFile.c_str()));
  hipDeviceptr_t mod_dptr = 0;
  size_t mod_bytes = 0;
  HIP_CHECK(hipModuleGetGlobal(&mod_dptr, &mod_bytes, mod, "d_var"));

  hipLibrary_t lib = nullptr;
  HIP_CHECK(hipLibraryLoadFromFile(&lib, kCodeFile.c_str(), nullptr, nullptr, 0, nullptr, nullptr,
                                   0));
  void* lib_dptr = nullptr;
  size_t lib_bytes = 0;
  HIP_CHECK(hipLibraryGetGlobal(&lib_dptr, &lib_bytes, lib, "d_var"));

  REQUIRE(mod_dptr != 0);
  REQUIRE(lib_dptr != nullptr);
  REQUIRE(mod_bytes == lib_bytes);
  REQUIRE(mod_bytes == sizeof(float) * 32);

  HIP_CHECK(hipLibraryUnload(lib));
  HIP_CHECK(hipModuleUnload(mod));
  CTX_DESTROY();
}

// Kernel and global lookup must compose in any order without one corrupting
// the other's cache.
HIP_TEST_CASE(Unit_hipLibrary_KernelGlobalLookupOrderIndependent) {
  HIP_TEST_DRIVER_INIT();
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
  REQUIRE(k1 == k2);

  unsigned int count = 0;
  HIP_CHECK(hipLibraryGetKernelCount(&count, lib));
  REQUIRE(count >= kMinKernelCount);

  HIP_CHECK(hipLibraryUnload(lib));
}

// Repeated load/unload of the same code object must remain clean across
// iterations (no leaked state, no stale managed-memory registrations).
HIP_TEST_CASE(Unit_hipLibrary_LoadUnloadCycle) {
  HIP_TEST_DRIVER_INIT();
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
