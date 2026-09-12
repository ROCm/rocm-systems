// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file plugin_export.cpp
/// @brief Loader exports for librocjitsu_plugin_pffm.so.

#include "rocjitsu/vm/plugins/pffm/plugin.h"
#include "rocjitsu/vm/plugins/plugin_exports.h"

ROCJITSU_DEFINE_PLUGIN(
    rocjitsu::plugins::pffm::PffmPlugin, "pffm",
    R"({"library_path":{"type":"string","description":"Path to libgpucsim_ffm_plugin.so"}})")
