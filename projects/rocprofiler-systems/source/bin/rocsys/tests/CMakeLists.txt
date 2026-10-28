# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(
    rocsys-tests
    OBJECT
    test_catalog.cpp
    test_cli_dispatcher.cpp
    test_traces.cpp
    test_delivery.cpp
    test_dry_run.cpp
    test_help.cpp
    test_launcher.cpp
    test_options.cpp
)

target_link_libraries(
    rocsys-tests
    PUBLIC
        rocprofiler-systems::rocprofiler-systems-rocsys-core
        rocprofiler-systems-googletest-library
)

target_compile_definitions(
    rocsys-tests
    PRIVATE ROCPROFSYS_PRESET_DIR="${PROJECT_SOURCE_DIR}/source/bin/common/presets"
)
