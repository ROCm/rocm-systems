cmake_minimum_required(VERSION 3.20)

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

project(aql_shm_monitor VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

include(GNUInstallDirs)

find_package(Threads REQUIRED)

set(ROCR_INC_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../projects/rocr-runtime/runtime/hsa-runtime/inc")

add_library(aqlmon_runtime_contract SHARED
    source/lib/runtime_contract/runtime_contract.cpp
)

target_include_directories(aqlmon_runtime_contract
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/source/include"
)

target_compile_options(aqlmon_runtime_contract PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

add_library(aql_shm_monitor SHARED
    source/lib/aqlmon/aqlmon.cpp
)

target_include_directories(aql_shm_monitor
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/source/include"
    SYSTEM PRIVATE
        "${ROCR_INC_DIR}"
)

target_link_libraries(aql_shm_monitor
    PRIVATE
        aqlmon_runtime_contract
        Threads::Threads
        dl
        rt
)

target_compile_options(aql_shm_monitor PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wno-gnu-anonymous-struct
    -Wno-nested-anon-types
)

add_executable(aqlmon_read
    samples/aqlmon_read.cpp
)

target_include_directories(aqlmon_read
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/source/include"
)

target_link_libraries(aqlmon_read PRIVATE rt)

add_executable(aqlmon_trace_json
    samples/aqlmon_trace_json.cpp
)

target_include_directories(aqlmon_trace_json
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/source/include"
)

target_link_libraries(aqlmon_trace_json PRIVATE rt)

add_executable(aqlmon_launch
    samples/aqlmon_launch.cpp
)

target_compile_options(aqlmon_launch PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

install(TARGETS aqlmon_runtime_contract aql_shm_monitor aqlmon_read aqlmon_trace_json
                aqlmon_launch)
install(DIRECTORY source/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
