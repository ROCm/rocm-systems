// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file plugin_export.cpp
/// @brief ABI exports for librocjitsu_plugin_data_hazard.so.
///
/// Kept separate from plugin.cpp so the plugin sources can also be linked
/// into the unit-test binary without colliding on the shared ABI symbol
/// names (rocjitsu_plugin_metadata / rocjitsu_plugin_create).

#include "rocjitsu/vm/plugins/data_hazard/plugin.h"
#include "rocjitsu/vm/plugins/plugin_exports.h"

namespace {

/// Declares the keys the loader accepts under
/// `"plugins": { "data_hazard": { ... } }`, and the defaults it fills in.
constexpr char kConfigSchema[] = R"({
  "report_path": {"type": "string", "default": ""},
  "verbose": {"type": "boolean", "default": false}
})";

} // namespace

ROCJITSU_DEFINE_PLUGIN(rocjitsu::plugins::data_hazard::DataHazardPlugin, "data_hazard",
                       kConfigSchema)
