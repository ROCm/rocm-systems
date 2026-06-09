// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file plugin_export.cpp
/// @brief ABI exports for librocjitsu_plugin_logging.so.
///
/// Kept separate from plugin.cpp so the plugin sources can also be linked
/// into the unit-test binary without colliding on the shared ABI symbol
/// names (rocjitsu_plugin_metadata / rocjitsu_plugin_create).

#include "rocjitsu/vm/plugins/logging/plugin.h"
#include "rocjitsu/vm/plugins/plugin_abi.h"

ROCJITSU_DEFINE_PLUGIN(rocjitsu::amdgpu::KernelLoggingPlugin, "logging",
                       "rocjitsu@amd.com", "1.0", "{}")
