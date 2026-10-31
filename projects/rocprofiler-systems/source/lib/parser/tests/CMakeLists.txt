# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(
    parser-tests
    OBJECT
    test_schema.cpp
    test_schema_validator.cpp
    test_config_store.cpp
    test_source_parsers.cpp
    test_file_preset.cpp
)

target_link_libraries(
    parser-tests
    PRIVATE rocprofiler-systems-googletest-library rocprofiler-systems-parser
)
