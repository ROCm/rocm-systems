# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rocjitsu_consan_runtime_core STATIC runtime_core.cpp)
target_link_libraries(
    rocjitsu_consan_runtime_core
    PUBLIC rocjitsu_consan_contract
)
rj_configure_target(rocjitsu_consan_runtime_core WARNINGS HIDDEN)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_runtime_libraries
        rocjitsu_hsa_tools_headers
        rocjitsu_consan_runtime_core
        rocjitsu_consan_image
        Threads::Threads
    )
    # Compile the runtime once; only the export maps differ between variants.
    add_library(hsa_consan_objects OBJECT hsa_plugin.cpp)
    target_link_libraries(hsa_consan_objects PRIVATE ${_runtime_libraries})
    set_target_properties(
        hsa_consan_objects
        PROPERTIES POSITION_INDEPENDENT_CODE ON
    )
    rj_configure_target(hsa_consan_objects PRIVATE_INCLUDES WARNINGS HIDDEN)
    set(_plugins hsa_consan)
    if(BUILD_TESTING)
        list(APPEND _plugins hsa_consan_testable)
    endif()
    foreach(_plugin IN LISTS _plugins)
        add_library(${_plugin} SHARED)
        target_link_libraries(
            ${_plugin}
            PRIVATE hsa_consan_objects ${_runtime_libraries}
        )
        rj_configure_target(${_plugin} WARNINGS HIDDEN)
        set(_exports ${CMAKE_CURRENT_SOURCE_DIR}/consan.map)
        if(_plugin STREQUAL "hsa_consan_testable")
            set(_exports ${CMAKE_CURRENT_SOURCE_DIR}/consan_testable.map)
        endif()
        set_property(TARGET ${_plugin} APPEND PROPERTY LINK_DEPENDS ${_exports})
        target_link_options(
            ${_plugin}
            PRIVATE
                "LINKER:--version-script=${_exports}"
                "LINKER:--no-undefined"
                "LINKER:-Map,${CMAKE_CURRENT_BINARY_DIR}/${_plugin}.link.map"
        )
    endforeach()
endif()
