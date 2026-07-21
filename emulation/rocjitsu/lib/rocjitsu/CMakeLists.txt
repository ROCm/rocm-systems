# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(ROCJITSU_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ROCJITSU_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

include(rj_add_object_library)

add_subdirectory(src/rocjitsu/analysis)
add_subdirectory(src/rocjitsu/code)
add_subdirectory(src/rocjitsu/isa)

# Private archive for consumers which only need code-object translation.  Keep
# this as a static library so each shared-library frontend pulls only the
# object files needed by its entry points.  In particular, the COMGR HotSwap
# adapter must not acquire the simulator implementation through this target.
set(ROCJITSU_DBT_OBJECT_LIBS
    rocjitsu_analysis
    rocjitsu_code
    rocjitsu_isa
    rocjitsu_isa_cdna1_model
    rocjitsu_isa_cdna2_model
    rocjitsu_isa_cdna3_model
    rocjitsu_isa_cdna4_model
    rocjitsu_isa_rdna1_model
    rocjitsu_isa_rdna2_model
    rocjitsu_isa_rdna3_model
    rocjitsu_isa_rdna3_5_model
    rocjitsu_isa_rdna4_model
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
