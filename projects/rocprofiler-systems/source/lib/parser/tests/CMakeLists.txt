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
    test_value_provider.cpp
)

target_link_libraries(
    parser-tests
    PRIVATE rocprofiler-systems-googletest-library rocprofiler-systems-parser
)

if(TARGET rocprofiler-systems::rocprofiler-systems-rocprofiler-sdk-backend)
    target_compile_definitions(parser-tests PRIVATE ROCPROFSYS_PARSER_HAS_SDK_BACKEND)
    target_link_libraries(
        parser-tests
        PRIVATE rocprofiler-systems::rocprofiler-systems-rocprofiler-sdk-backend
    )
endif()
