// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/cdna5/wmma_v4_provider.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_math_provider.h"

namespace rocjitsu {

// Private, same-build interface. Native SIMD types never cross this boundary.
inline constexpr uint32_t kX86V4ProviderAbi = 1;

struct X86V4ProviderDescriptor {
  uint32_t abi_version;
  const char *build_digest;
  std::array<Instruction::ExecuteFn, cdna5::kWmmaV4FormCount> wmma_callbacks;
  void (*bind_wmma_backend)(const IsaExecutionBackend *) noexcept;
  amdgpu::fp_math::UnaryF32Kernel exp;
  amdgpu::fp_math::UnaryF32Kernel log;
};

using GetX86V4Provider = const X86V4ProviderDescriptor *(*)() noexcept;

extern "C" RJ_API_EXPORT const X86V4ProviderDescriptor *rj_x86_v4_provider_v1() noexcept;

bool cpu_and_os_support_x86_v4() noexcept;

// Optional selection falls back on an unsupported host or load/ABI failure.
// Required selection reports the failure here, before any noexcept hot lookup.
// A successful provider stays loaded for the lifetime of its callback pointers.
const X86V4ProviderDescriptor *resolve_x86_v4_provider(bool required);

} // namespace rocjitsu
