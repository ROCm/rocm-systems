# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_vm_plugins
    kernel_logging_plugin.cpp
    race_detection_plugin.cpp
)
target_include_directories(
    rocjitsu_vm_plugins
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/.. # resolves "plugins/execution_plugin.h"
)
target_link_libraries(rocjitsu_vm_plugins PUBLIC race_detector)
