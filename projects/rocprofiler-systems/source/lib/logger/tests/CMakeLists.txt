# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(logger_tests_sources test_logger.cpp)

add_library(logger-tests OBJECT ${logger_tests_sources})

target_link_libraries(
    logger-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-logger
        rocprofiler-systems-spdlog
)
