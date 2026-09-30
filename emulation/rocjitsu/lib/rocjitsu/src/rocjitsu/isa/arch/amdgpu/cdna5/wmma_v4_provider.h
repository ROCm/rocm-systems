// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/isa/execution_backend.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rocjitsu::cdna5 {

// Private, same-build contract between librocjitsu.so and its focused v4 DSO.
// This is not a stable public plugin ABI.
enum class WmmaV4Form : size_t {
  F32F16,
  F16F16,
  F32BF16,
  BF16BF16,
  BF16F32BF16,
  Count,
};

inline constexpr uint32_t kWmmaV4ProviderAbi = 1;
inline constexpr size_t kWmmaV4FormCount = static_cast<size_t>(WmmaV4Form::Count);

struct WmmaV4ProviderDescriptor {
  uint32_t abi_version;
  const char *build_digest;
  std::array<Instruction::ExecuteFn, kWmmaV4FormCount> callbacks;
  void (*bind_backend)(const IsaExecutionBackend *) noexcept;
};

using GetWmmaV4Provider = const WmmaV4ProviderDescriptor *(*)() noexcept;

extern "C" RJ_API_EXPORT const WmmaV4ProviderDescriptor *rj_cdna5_wmma_v4_provider_v1() noexcept;

} // namespace rocjitsu::cdna5
