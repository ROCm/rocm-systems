# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)
project(x86_v4_provider_build_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/rj_x86_v4_provider.cmake")

rj_x86_v4_build_digest(
    digest
    INPUT "${CMAKE_CXX_COMPILER}|${CMAKE_CXX_COMPILER_VERSION}"
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/provider.cpp"
)
add_library(provider_objects OBJECT provider.cpp)
file(
    GENERATE OUTPUT "${PROJECT_BINARY_DIR}/generated.$<CONFIG>.cpp"
    CONTENT
        "extern \"C\" const char *provider_generated_config() { return \"$<CONFIG>\"; }\n"
)
add_library(
    generated_objects
    OBJECT
    "${PROJECT_BINARY_DIR}/generated.$<CONFIG>.cpp"
)
target_compile_definitions(
    provider_objects
    PRIVATE "PROVIDER_CONFIG=\"$<CONFIG>\"" "PROVIDER_DIGEST=\"${digest}\""
)
rj_add_x86_v4_local_object(
    provider_local provider local_object
    OBJECT_TARGETS provider_objects generated_objects
    OBJCOPY_OPTIONS
        --keep-global-symbol=provider_config
        --keep-global-symbol=provider_digest
        --keep-global-symbol=provider_flag
        --keep-global-symbol=provider_generated_config
)
add_library(provider SHARED "${local_object}")
add_dependencies(provider provider_local)
set_target_properties(
    provider
    PROPERTIES
        LINKER_LANGUAGE CXX
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin/$<CONFIG>"
)
add_executable(probe main.cpp)
target_compile_definitions(
    probe
    PRIVATE "PROVIDER_CONFIG=\"$<CONFIG>\"" "PROVIDER_DIGEST=\"${digest}\""
)
target_link_libraries(probe PRIVATE provider ${CMAKE_DL_LIBS})
set_target_properties(
    probe
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin/$<CONFIG>"
)
