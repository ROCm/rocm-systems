# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Concrete plugins (each will become a standalone .so in the future — see #6628).
rj_add_object_library(
    rocjitsu_plugin_logging
    logging/plugin.cpp
)

rj_add_object_library(
    rocjitsu_plugin_race_detector
    race_detector/plugin.cpp
    race_detector/core/race_detector.cpp
    race_detector/core/wave_race_state.cpp
)
