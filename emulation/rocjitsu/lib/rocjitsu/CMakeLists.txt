# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(ROCJITSU_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ROCJITSU_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

include(rj_add_object_library)

add_subdirectory(src/rocjitsu/analysis)
add_subdirectory(src/rocjitsu/code)
add_subdirectory(src/rocjitsu/isa)

# Private archive for the fixed-profile gfx1250 HotSwap translator. Keep this as
# a static library so the shared-library frontend pulls only the generic DBT
# implementation and gfx1250 model selected by its entry point. In particular,
# it must not acquire the full decoder registry or simulator implementation.
set(ROCJITSU_DBT_OBJECT_LIBS
    rocjitsu_analysis
    rocjitsu_code
    rocjitsu_isa
    rocjitsu_isa_gfx1250_model
)
add_library(rocjitsu_dbt_internal STATIC)
foreach(_object_lib IN LISTS ROCJITSU_DBT_OBJECT_LIBS)
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
    target_link_options(rocjitsu_dbt_internal INTERFACE "LINKER:--gc-sections")
endif()

add_subdirectory(src/rocjitsu/vm)
add_subdirectory(src/rocjitsu/kmd)
add_subdirectory(src/rocjitsu/config)
add_subdirectory(src/rocjitsu/hooks)
