// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file plugin_export.cpp
/// @brief Loader exports for librocjitsu_plugin_perfsim.so.
///
/// The `library_path` setting names the separate Perfsim backend loaded by this
/// adapter, not this RocJITsu plugin shared object.

#include "rocjitsu/vm/plugins/perfsim/plugin.h"
#include "rocjitsu/vm/plugins/plugin_exports.h"

ROCJITSU_DEFINE_PLUGIN(
    rocjitsu::plugins::perfsim::PerfsimPlugin, "perfsim",
    R"({"library_path":{"type":"string","description":"Absolute path to the separately supplied Perfsim backend shared object, such as libgpucsim_ffm_plugin.so"},"max_staged_bytes":{"type":"number","description":"Maximum charged bytes for staged event records","default":268435456}})")
