# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(ROCJITSU_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ROCJITSU_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

include(rj_add_object_library)
include(rj_configure_target)

add_subdirectory(src/rocjitsu/analysis)
add_subdirectory(src/rocjitsu/code)
add_subdirectory(src/rocjitsu/isa)

# Private base archive for code-object translation frontends. Architecture
# models and decoder factories are supplied by each frontend, so this target
# does not pull in the simulator implementation or the all-ISA registry.
set(ROCJITSU_DBT_BASE_OBJECT_LIBS rocjitsu_analysis rocjitsu_code rocjitsu_isa)
add_library(rocjitsu_dbt_internal STATIC)
foreach(_object_lib IN LISTS ROCJITSU_DBT_BASE_OBJECT_LIBS)
    target_sources(
        rocjitsu_dbt_internal
        PRIVATE $<TARGET_OBJECTS:${_object_lib}>
    )
endforeach()
target_link_libraries(rocjitsu_dbt_internal PUBLIC util Threads::Threads)
set_target_properties(
    rocjitsu_dbt_internal
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Fixed-profile shared library for the load-time gfx1250 B0-to-A0
    # consumer. Its default registry contains only the gfx1250 model provider,
    # excluding the full simulator aggregate and execution objects. Keep this
    # Linux-only until equivalent export control is provided for other
    # shared-library formats.
    add_library(
        rocjitsu_gfx1250_b0_to_a0
        SHARED
        src/rocjitsu/code/dbt/gfx1250_b0_to_a0_library.cpp
    )
    set_target_properties(
        rocjitsu_gfx1250_b0_to_a0
        PROPERTIES
            OUTPUT_NAME rocjitsu_gfx1250_b0_to_a0
            VERSION ${PROJECT_VERSION}
            SOVERSION ${PROJECT_VERSION_MAJOR}
    )
    target_link_libraries(
        rocjitsu_gfx1250_b0_to_a0
        PRIVATE
            rocjitsu_dbt_internal
            rocjitsu_gfx1250_model_registry
            simdojo_headers
    )
    rj_configure_target(
        rocjitsu_gfx1250_b0_to_a0
        PRIVATE_INCLUDES
        WARNINGS
        HIDDEN
    )
    set(_gfx1250_b0_to_a0_exports
        ${ROCJITSU_SRC_DIR}/rocjitsu/code/dbt/gfx1250_b0_to_a0.exports
    )
    set_property(
        TARGET rocjitsu_gfx1250_b0_to_a0
        APPEND
        PROPERTY LINK_DEPENDS ${_gfx1250_b0_to_a0_exports}
    )
    target_link_options(
        rocjitsu_gfx1250_b0_to_a0
        PRIVATE
            "LINKER:--exclude-libs,ALL"
            "LINKER:--version-script=${_gfx1250_b0_to_a0_exports}"
    )
endif()

add_subdirectory(src/rocjitsu/vm)
add_subdirectory(src/rocjitsu/kmd)
add_subdirectory(src/rocjitsu/config)
add_subdirectory(src/rocjitsu/hooks)
