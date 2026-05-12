# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(
    bin-common-tests
    OBJECT
    test_preset_registry.cpp
    test_json_config.cpp
    test_help_system.cpp
    test_print_output.cpp
)

target_include_directories(bin-common-tests PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../..)

target_link_libraries(
    bin-common-tests
    PUBLIC
        rocprofiler-systems-common-library
        rocprofiler-systems-googletest-library
        rocprofiler-systems::rocprofiler-systems-bin-common
        rocprofiler-systems::rocprofiler-systems-json
)
