# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(rocprofiler-systems-user-api-example LANGUAGES CXX)

# Support standalone builds
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(${CMAKE_CURRENT_LIST_DIR}/../cmake/standalone-helpers.cmake OPTIONAL)
endif()

set(CMAKE_BUILD_TYPE "Debug")
find_package(Threads REQUIRED)
if(NOT TARGET rocprofiler-systems::rocprofiler-systems-user-library)
    find_package(rocprofiler-systems REQUIRED COMPONENTS user)
endif()
add_executable(user-api user-api.cpp)
target_link_libraries(
    user-api
    PRIVATE Threads::Threads rocprofiler-systems::rocprofiler-systems-user-library
)

if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET user-api)
    install(
        TARGETS user-api
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
        COMPONENT rocprofiler-systems-examples
    )
endif()
