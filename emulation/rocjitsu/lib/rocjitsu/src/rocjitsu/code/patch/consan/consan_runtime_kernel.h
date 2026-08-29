// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_runtime_kernel.h
/// @brief Shared classification of ROCclr implementation kernels.

#pragma once

#include <string_view>

namespace rocjitsu {

/// ROCclr reserves this prefix for runtime implementation kernels that are not
/// user synchronization domains and must not be selected for instrumentation.
[[nodiscard]] constexpr bool is_rocclr_runtime_kernel_name(std::string_view name) {
  return name.starts_with("__amd_rocclr_");
}

} // namespace rocjitsu
