# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.0)
project(generateHipAPICoverage)

set(CMAKE_CXX_STANDARD 17)

set(SRC
    mainCoverage.cpp
    hipAPI.cpp
    hipAPIGroup.cpp
    reportGenerators.cpp
    hipAPICoverageUtils.cpp)

add_executable(generateHipAPICoverage ${SRC})
target_compile_options(generateHipAPICoverage PRIVATE -Wall -Wextra -Wvla -Werror -Wno-deprecated -Wno-option-ignored)

add_custom_command(TARGET generateHipAPICoverage POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_LIST_DIR}/coverageReportHTML ${CMAKE_CURRENT_BINARY_DIR}/coverageReportHTML
)

add_custom_target(clean-cov-report
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/*.xml
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/coverageReportHTML/testModules
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/coverageReportHTML/testAPIs
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/coverageReportHTML/*.html
)
