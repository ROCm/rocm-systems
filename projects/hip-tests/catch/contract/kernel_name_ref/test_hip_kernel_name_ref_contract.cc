/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstring>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

// BACKEND-DIFF: hipKernelNameRef and hipKernelNameRefByPtr are AMD-only
// name-reflection entry points with no NVIDIA-backend equivalent, so this whole
// translation unit builds only on AMD. Parity would require a NVIDIA-side
// kernel-name reflection API.
#if HT_AMD

// Kernel-name reflection contracts for the AMD name-lookup entry points
// hipKernelNameRef (by function handle) and hipKernelNameRefByPtr (by host
// function pointer). Both return a C string naming the kernel, or nullptr when
// no name can be resolved. These exercise the reflection round-trip: a known
// in-source kernel must resolve to a non-empty name that mentions the kernel's
// identifier.

namespace {
__global__ void KernelNameRefProbe(int* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = 1;
  }
}

// A resolved kernel name must be non-null, non-empty, and mention the kernel's
// source identifier (the AMD backend returns the mangled symbol, which embeds
// the identifier "KernelNameRefProbe").
void RequireNamesKernel(const char* name) {
  REQUIRE(name != nullptr);
  REQUIRE(name[0] != '\0');
  REQUIRE(std::strstr(name, "KernelNameRefProbe") != nullptr);
}
}  // namespace

// hipKernelNameRefByPtr resolves a host function pointer to its kernel name
// without any module load; the returned string must name the kernel.
// @asserts: hipKernelNameRefByPtr - resolves a host kernel function pointer to a non-empty name mentioning the kernel identifier
HIP_TEST_CASE(Contract_KernelNameRef_ByPtr_NamesHostKernel) {
  const char* name =
      hipKernelNameRefByPtr(reinterpret_cast<const void*>(KernelNameRefProbe), nullptr);
  RequireNamesKernel(name);
}

// hipKernelNameRef resolves a hipFunction_t (obtained from the host symbol via
// hipGetFuncBySymbol) to its kernel name; the returned string must name the
// kernel.
// @asserts: hipKernelNameRef - resolves a hipFunction_t (from hipGetFuncBySymbol) to a non-empty name mentioning the kernel identifier
HIP_TEST_CASE(Contract_KernelNameRef_ByFunction_NamesResolvedKernel) {
  hipFunction_t function = nullptr;
  HIP_CHECK(hipGetFuncBySymbol(&function, reinterpret_cast<const void*>(KernelNameRefProbe)));
  REQUIRE(function != nullptr);

  const char* name = hipKernelNameRef(function);
  RequireNamesKernel(name);
}
#endif  // HT_AMD
