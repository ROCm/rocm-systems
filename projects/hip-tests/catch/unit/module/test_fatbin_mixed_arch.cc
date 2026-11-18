/*
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
*/

/*
Regression tests for the fat-binary code-object selection fix in
projects/clr/hipamd/src/hip_fatbin.cpp
(FatBinaryInfo::ExtractFatBinaryUsingCOMGR).

Bug (ROCm/TheRock#5543): the extraction loop iterated over *all* enumerated
devices and executed `break` on the first device that had no compatible code
object in the fat binary. Every device enumerated *after* that one therefore
never had its program registered, and a later kernel launch on it failed with
hipErrorInvalidImage -- even though its code object was present in the binary.
The fix replaces that `break` with `continue`, so an incompatible device is
skipped instead of poisoning registration for the others.

Two things are verified:

  1. Unit_FatBin_NoCompatibleCodeObject_ReturnsInvalidImage
       Contract guard (deterministic, single GPU): loading a fat binary that
       contains no code object matching the current device must still fail with
       hipErrorInvalidImage. This exercises the modified else-branch and proves
       `continue` preserves the "no compatible code object" failure (hip_status
       is initialized to hipErrorInvalidImage and is left untouched when every
       device is skipped).

  2. Unit_FatBin_MixedArch_IncompatibleDeviceDoesNotPoisonOthers
       End-to-end regression (multi-GPU): with an incompatible device present
       (and possibly enumerated first), a compiled-in kernel must still run on
       a device whose arch IS in the binary. This is the actual #5543 scenario
       and only reaches the buggy path via the static __hipRegisterFatBinary /
       StatCO::DigestFatBinary(g_devices) route, hence a compiled-in kernel
       (not hipModuleLoad*, which is single-device).

Note: #2 only exercises the incompatible-device path when at least one
enumerated device's arch is absent from the fat binary, i.e. on a system with
>= 2 distinct archs where the binary is built for a subset. It skips otherwise
and never produces a false failure. There is no public API to inject a
controlled multi-device fat binary, so a fully hardware-independent bug-catcher
for the multi-device loop is not possible.
*/

#include <hip_test_common.hh>
#include <hip/hip_runtime.h>

#include <set>
#include <string>
#include <vector>

__global__ void writeValueKernel(int* out, int value) { *out = value; }

// Attempt to launch the compiled-in kernel on `dev`. Returns true if the kernel
// ran and produced the expected value; false if the device had no compatible
// code object (a legitimate outcome for an unsupported device).
static bool LaunchKernelOnDevice(int dev, int expected) {
  HIP_CHECK(hipSetDevice(dev));
  int* d = nullptr;
  HIP_CHECK(hipMalloc(&d, sizeof(int)));
  HIP_CHECK(hipMemset(d, 0, sizeof(int)));

  writeValueKernel<<<1, 1>>>(d, expected);
  hipError_t launchErr = hipGetLastError();
  hipError_t syncErr = hipDeviceSynchronize();

  bool ran = (launchErr == hipSuccess && syncErr == hipSuccess);
  if (ran) {
    int host = 0;
    HIP_CHECK(hipMemcpy(&host, d, sizeof(int), hipMemcpyDeviceToHost));
    REQUIRE(host == expected);
  } else {
    // A device with no compatible code object in the binary may fail here; that
    // is acceptable. What must NOT happen (pre-fix behavior) is that this failure
    // is observed on *every* device because an incompatible one aborted the whole
    // registration loop.
    REQUIRE((launchErr == hipErrorInvalidImage || syncErr == hipErrorInvalidImage ||
             launchErr == hipErrorInvalidKernelFile ||
             syncErr == hipErrorInvalidKernelFile ||
             launchErr == hipErrorSharedObjectInitFailed ||
             syncErr == hipErrorSharedObjectInitFailed));
    (void)hipGetLastError();  // clear sticky error
  }

  HIP_CHECK(hipFree(d));
  return ran;
}

/*
 * Deterministic contract guard: a fat binary with no code object for the current
 * device's arch must fail with hipErrorInvalidImage. fatbin_mismatch_module.code
 * is built for gfx900 + gfx906 only (see CMakeLists.txt); on any non-Vega device
 * neither arch matches, driving ExtractFatBinaryUsingCOMGR into the modified
 * else-branch. Skips on gfx900/gfx906 so the fixture is guaranteed incompatible.
 */
HIP_TEST_CASE(Unit_FatBin_NoCompatibleCodeObject_ReturnsInvalidImage) {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const std::string arch(prop.gcnArchName);
  if (arch.rfind("gfx900", 0) == 0 || arch.rfind("gfx906", 0) == 0) {
    HIP_SKIP_TEST("Fixture is built for gfx900/gfx906; current device matches it");
    return;
  }

  hipModule_t module = nullptr;
  // Observed as hipErrorInvalidImage; accept hipErrorNoBinaryForGpu too since some
  // stacks surface the "no code object for this device" case with that code.
  HIP_CHECK_ERRORS(hipModuleLoad(&module, "fatbin_mismatch_module.code"),
                   hipErrorInvalidImage, hipErrorNoBinaryForGpu);
}

/*
 * End-to-end #5543 regression: on a multi-GPU system with >= 2 distinct archs,
 * a compiled-in kernel must run on at least one device even when an incompatible
 * device is also enumerated. Pre-fix, an incompatible device sorted ahead of a
 * supported one aborted registration for all devices and no launch succeeded.
 */
HIP_TEST_CASE(Unit_FatBin_MixedArch_IncompatibleDeviceDoesNotPoisonOthers) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices < 2) {
    HIP_SKIP_TEST("Requires >= 2 GPUs to exercise the multi-device extraction loop");
    return;
  }

  std::set<std::string> archs;
  for (int i = 0; i < numDevices; ++i) {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, i));
    archs.insert(std::string(prop.gcnArchName));
  }
  if (archs.size() < 2) {
    HIP_SKIP_TEST(
        "Requires >= 2 distinct GPU archs (a single-arch fat binary covers "
        "every device, so the incompatible-device path is never reached)");
    return;
  }

  int ranCount = 0;
  for (int dev = 0; dev < numDevices; ++dev) {
    if (LaunchKernelOnDevice(dev, 0xABC + dev)) ++ranCount;
  }
  INFO("devices=" << numDevices << " distinctArchs=" << archs.size()
                  << " devicesThatRan=" << ranCount);

  // The crux of the fix: presence of an incompatible device must not prevent the
  // supported device(s) from running the kernel.
  REQUIRE(ranCount >= 1);
}
