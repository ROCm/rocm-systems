/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Isolated TU for hipGetDevicePropertiesR0600. HostUnitTests link -no-hip-rt
 * and wrapper_link_stubs.cpp uses int-returning HIP fakes that conflict with
 * hip_runtime.h, so the versioned properties ABI lives here instead.
 *
 * Always zero the struct and write a NUL-terminated gcnArchName so
 * rcclSkipCuMemFree() / rcclSkipLsaFlatAddressFree() do not strstr
 * uninitialized memory. Default arch is gfx900 (skip-free auto-off).
 * Override with RCCL_TEST_GCN_ARCH (e.g. gfx950, gfx1250).
 *************************************************************************/

#include <cstdio>
#include <cstdlib>

#include <hip/hip_runtime_api.h>

hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_t* prop, int /*device*/) {
  if (prop == nullptr) return hipErrorInvalidValue;
  *prop = hipDeviceProp_t{};
  const char* arch = std::getenv("RCCL_TEST_GCN_ARCH");
  if (arch == nullptr || arch[0] == '\0') arch = "gfx900";
  std::snprintf(prop->gcnArchName, sizeof(prop->gcnArchName), "%s", arch);
  return hipSuccess;
}
