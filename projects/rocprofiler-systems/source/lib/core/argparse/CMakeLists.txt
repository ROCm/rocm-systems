# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

target_sources(
    rocprofiler-systems-core-library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/core_flags.cpp
        ${CMAKE_CURRENT_LIST_DIR}/core_flags.hpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/parser_engine.hpp
        ${CMAKE_CURRENT_LIST_DIR}/flag_descriptor.hpp
        ${CMAKE_CURRENT_LIST_DIR}/interpreter.hpp
        ${CMAKE_CURRENT_LIST_DIR}/interpreter.cpp
        ${CMAKE_CURRENT_LIST_DIR}/parsed_values.cpp
        ${CMAKE_CURRENT_LIST_DIR}/parsed_values.hpp
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
