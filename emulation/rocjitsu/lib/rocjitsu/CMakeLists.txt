# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(ROCJITSU_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ROCJITSU_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

include(rj_add_object_library)

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

add_subdirectory(src/rocjitsu/vm)
add_subdirectory(src/rocjitsu/kmd)
add_subdirectory(src/rocjitsu/config)
add_subdirectory(src/rocjitsu/hooks)
