# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_vm_plugins
    kernel_logging_plugin.cpp
    race_detection_plugin.cpp
    cycle_model_plugin.cpp
    cu_cycle_model.cpp
    mem_sys_cycle_model.cpp
)
target_include_directories(
    rocjitsu_vm_plugins
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/.. # resolves "plugins/execution_plugin.h"
)
target_link_libraries(rocjitsu_vm_plugins PUBLIC race_detector cycle_model)
# M1: single cdna4 config baked in (full gfx_target_version selection comes later).
target_compile_definitions(rocjitsu_vm_plugins PRIVATE
    CYCLE_MODEL_CDNA4_JSON="${CMAKE_CURRENT_SOURCE_DIR}/../configs/cycle_model/cdna4.json")
