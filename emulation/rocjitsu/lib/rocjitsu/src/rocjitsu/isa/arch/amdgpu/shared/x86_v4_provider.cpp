// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/x86_v4_provider.h"

#ifndef ROCJITSU_X86_V4_BUILD_DIGEST
#error "ROCJITSU_X86_V4_BUILD_DIGEST must match the shared v4 resolver"
#endif

namespace rocjitsu {
namespace {

using namespace amdgpu::fp_math;

// Keep the whole wave callback with its bit kernel. The shared-library caller
// makes one indirect call per wave, not one per eight-lane pack.
template <Operation Op, F32Bits8 (*Kernel)(F32Bits8, bool) noexcept>
void execute_eight(const F32UnaryWave &wave) {
  constexpr uint32_t inactive_input = Op == Operation::Exp ? 0u : 0x3f800000u;
  for (unsigned base = 0; base < wave.size(); base += 8) {
    const LaneMask<8> active = wave.active().chunk<8>(base);
    if (active.empty())
      continue;
    const F32Bits8 input{wave.source().load<8>(base, active, inactive_input)};
    const F32Bits8 result = Kernel(input, wave.mode().quiet_snan);
    wave.destination().store<8>(base, result.lane, active);
  }
}

} // namespace

extern "C" const X86V4ProviderDescriptor *rj_x86_v4_provider_v1() noexcept {
  // Initialized only when the baseline resolver invokes the qualified getter.
  static const X86V4ProviderDescriptor descriptor = [] {
    const auto *wmma = cdna5::rj_cdna5_wmma_v4_provider_v1();
    return X86V4ProviderDescriptor{
        kX86V4ProviderAbi,
        ROCJITSU_X86_V4_BUILD_DIGEST,
        wmma->callbacks,
        wmma->bind_backend,
        {&execute_eight<Operation::Exp, &exp_v4x8>, MathKernelKind::Avx512x8, 8, 1},
        {&execute_eight<Operation::Log, &log_v4x8>, MathKernelKind::Avx512x8, 8, 1},
    };
  }();
  return &descriptor;
}

} // namespace rocjitsu
