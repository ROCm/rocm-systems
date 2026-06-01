# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_config
    config_loader.cpp
    checkpoint.cpp
)

# Config loader depends on FlatBuffers library (for runtime JSON parsing)
# and on the generated schema headers.
target_link_libraries(rocjitsu_config PRIVATE flatbuffers)
target_include_directories(rocjitsu_config PRIVATE ${GENERATED_DIR})
add_dependencies(rocjitsu_config flatbuffers_schemas)
