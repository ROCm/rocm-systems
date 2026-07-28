/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "device/device.hpp"
#include "device/rocm/rocsettings.hpp"

#include <cstdio>

// ROCclr is a static library. Its enclosing HIP or OpenCL runtime normally
// provides this dispatch table, so a direct unit-test link supplies a stub.
cl_icd_dispatch amd::ICDDispatchedObject::icdVendorDispatch_[] = {0};

namespace {

bool check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
  }
  return condition;
}

}  // namespace

int main() {
  const amd::Isa* presented = amd::Isa::findIsa("amdgcn-amd-amdhsa--gfx1250");
  const amd::Isa* presentedWgp = amd::Isa::findIsa("amdgcn-amd-amdhsa--gfx1100");
  const amd::Isa* execution = amd::Isa::findIsa("amdgcn-amd-amdhsa--gfx942:xnack+");
  const amd::Isa* legacyExecution = amd::Isa::findIsa("amdgcn-amd-amdhsa--gfx900");
  if (!check(presented != nullptr, "missing gfx1250 test ISA") ||
      !check(presentedWgp != nullptr, "missing gfx1100 test ISA") ||
      !check(execution != nullptr, "missing gfx942 test ISA") ||
      !check(legacyExecution != nullptr, "missing gfx900 test ISA")) {
    return 1;
  }

  amd::roc::Settings nativeExecution;
  amd::roc::Settings nativeLegacyExecution;
  amd::roc::Settings nativePresented;
  amd::roc::Settings nativePresentedWgp;
  amd::roc::Settings split;
  amd::roc::Settings splitLegacyExecution;
  amd::roc::Settings splitWgp;
  if (!check(nativeExecution.create(false, *execution, *execution),
             "failed to create native execution settings") ||
      !check(nativeLegacyExecution.create(false, *legacyExecution, *legacyExecution),
             "failed to create native legacy execution settings") ||
      !check(nativePresented.create(false, *presented, *presented),
             "failed to create native presented settings") ||
      !check(nativePresentedWgp.create(false, *presentedWgp, *presentedWgp),
             "failed to create native presented WGP settings") ||
      !check(split.create(false, *presented, *execution), "failed to create split settings") ||
      !check(splitLegacyExecution.create(false, *presented, *legacyExecution),
             "failed to create split legacy execution settings") ||
      !check(splitWgp.create(false, *presentedWgp, *execution),
             "failed to create split WGP settings")) {
    return 1;
  }

  bool passed = true;
  passed &= check(split.barrier_value_packet_ == nativeExecution.barrier_value_packet_ &&
                      split.barrier_value_packet_ != nativePresented.barrier_value_packet_,
                  "barrier packet policy did not follow the execution ISA");
  passed &= check(split.ext_dispatch_packet_ == nativeExecution.ext_dispatch_packet_ &&
                      split.ext_dispatch_packet_ != nativePresented.ext_dispatch_packet_,
                  "dispatch packet policy did not follow the execution ISA");
  passed &= check(split.sdma_indirect_supported_ == nativeExecution.sdma_indirect_supported_ &&
                      split.sdma_indirect_supported_ != nativePresented.sdma_indirect_supported_,
                  "SDMA packet policy did not follow the execution ISA");
  passed &= check(splitLegacyExecution.kernel_arg_impl_ == nativeLegacyExecution.kernel_arg_impl_ &&
                      splitLegacyExecution.kernel_arg_impl_ != nativePresented.kernel_arg_impl_,
                  "kernel argument policy did not follow the execution ISA");
  passed &= check(split.enableWave32Mode_ == nativePresented.enableWave32Mode_ &&
                      split.lcWavefrontSize64_ == nativePresented.lcWavefrontSize64_ &&
                      split.enableWave32Mode_ != nativeExecution.enableWave32Mode_ &&
                      split.lcWavefrontSize64_ != nativeExecution.lcWavefrontSize64_,
                  "wave code-generation policy did not follow the presented ISA");
  passed &=
      check(!nativePresented.enableXNACK_ && !split.enableXNACK_ && nativeExecution.enableXNACK_,
            "XNACK target feature did not follow the presented ISA");
  passed &= check(!splitWgp.enableWgpMode_ && splitWgp.lcWgpMode_,
                  "physical and compiler WGP modes were not kept separate");
  passed &= check(splitWgp.enableWgpMode_ == nativeExecution.enableWgpMode_ &&
                      splitWgp.lcWgpMode_ == nativePresentedWgp.lcWgpMode_,
                  "WGP policies did not follow their respective ISA views");

  return passed ? 0 : 1;
}
