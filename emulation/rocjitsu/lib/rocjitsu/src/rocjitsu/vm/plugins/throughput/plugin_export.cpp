// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file plugin_export.cpp
/// @brief Loader exports for librocjitsu_plugin_throughput.so.
///
/// Kept separate from plugin.cpp so the plugin sources can also be linked
/// into the unit-test binary without colliding on the shared loader exports.

#include "rocjitsu/vm/plugins/plugin_exports.h"
#include "rocjitsu/vm/plugins/throughput/plugin.h"

ROCJITSU_DEFINE_PLUGIN(rocjitsu::plugins::throughput::ThroughputPlugin, "throughput", "{}")
