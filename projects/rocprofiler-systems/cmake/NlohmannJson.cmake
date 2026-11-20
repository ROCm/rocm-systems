# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

if(ROCPROFSYS_BUILD_NLOHMANN_JSON)
    message(STATUS "Building nlohmann/json from source")
    include(FetchContent)

    rocprofiler_systems_checkout_git_submodule(
        RELATIVE_PATH external/json
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        TEST_FILE CMakeLists.txt
    )
    FetchContent_Declare(nlohmann_json SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/json)
    FetchContent_MakeAvailable(nlohmann_json)

    # An earlier FetchContent_Declare may resolve via find_package, leaving only the imported target
    set(_rocprofsys_json_target nlohmann_json)
    if(NOT TARGET nlohmann_json)
        find_package(nlohmann_json REQUIRED)
        set(_rocprofsys_json_target nlohmann_json::nlohmann_json)
    endif()

    target_include_directories(
        rocprofiler-systems-json
        SYSTEM
        INTERFACE
            $<TARGET_PROPERTY:${_rocprofsys_json_target},INTERFACE_INCLUDE_DIRECTORIES>
    )
    target_link_libraries(rocprofiler-systems-json INTERFACE ${_rocprofsys_json_target})
else()
    message(STATUS "Using system nlohmann/json library")
    find_package(nlohmann_json REQUIRED)
    target_link_libraries(rocprofiler-systems-json INTERFACE nlohmann_json::nlohmann_json)
endif()
