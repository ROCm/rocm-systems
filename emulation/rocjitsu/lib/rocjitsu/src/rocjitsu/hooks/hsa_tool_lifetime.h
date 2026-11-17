// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace rocjitsu::hooks {

/// @brief Retains the calling HSA tool DSO until process termination.
///
/// ROCR closes analysis tools from inside its own shutdown path. If a client
/// dynamically unloads the HIP runtime, that nested close can otherwise make
/// the dynamic loader unmap ROCR before shutdown returns. Every HSA tool must
/// call this once from OnLoad before installing API wrappers.
[[nodiscard]] bool retain_hsa_tool_dso() noexcept;

} // namespace rocjitsu::hooks
