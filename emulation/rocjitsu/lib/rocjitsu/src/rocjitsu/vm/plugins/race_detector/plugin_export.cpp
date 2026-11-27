// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file plugin_export.cpp
/// @brief Loader exports for librocjitsu_plugin_race.so.
///
/// Kept separate from plugin.cpp so the plugin sources can also be linked
/// into the unit-test binary without colliding on the shared loader exports.

#include "rocjitsu/vm/plugins/plugin_exports.h"
#include "rocjitsu/vm/plugins/race_detector/plugin.h"

ROCJITSU_DEFINE_PLUGIN(rocjitsu::plugins::race_detector::RaceDetectorPlugin, "race",
                       R"({"trace_context":{"type":"number","default":128,
                       "description":"Instruction context before elision (2-256); waits, barriers and control flow are retained."}})")
