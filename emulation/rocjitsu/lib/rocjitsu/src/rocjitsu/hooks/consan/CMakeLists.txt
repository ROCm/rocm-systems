# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rocjitsu_consan_runtime_core STATIC runtime_core.cpp)
target_link_libraries(
    rocjitsu_consan_runtime_core
    PUBLIC rocjitsu_consan_contract
)
rj_configure_target(rocjitsu_consan_runtime_core WARNINGS HIDDEN)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_library(hsa_consan SHARED hsa_plugin.cpp)
    target_link_libraries(
        hsa_consan
        PRIVATE
            rocjitsu_hsa_tools_headers
            rocjitsu_consan_runtime_core
            rocjitsu_consan_image
            Threads::Threads
    )
    rj_configure_target(hsa_consan PRIVATE_INCLUDES WARNINGS HIDDEN)
    set(_exports ${CMAKE_CURRENT_SOURCE_DIR}/consan.map)
    set_property(TARGET hsa_consan APPEND PROPERTY LINK_DEPENDS ${_exports})
    target_link_options(
        hsa_consan
        PRIVATE
            "LINKER:--version-script=${_exports}"
            "LINKER:--no-undefined"
            "LINKER:-Map,${CMAKE_CURRENT_BINARY_DIR}/hsa_consan.link.map"
    )
endif()
