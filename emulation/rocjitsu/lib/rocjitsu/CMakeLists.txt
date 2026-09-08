# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(ROCJITSU_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ROCJITSU_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

include(rj_add_object_library)
include(rj_configure_target)

# One compiled source of truth for version information. The object is embedded
# directly in librocjitsu.so so its C API remains exported despite the shared
# library's archive-hiding link option; native consumers that call the API use
# the same object through the private static archive.
add_library(rocjitsu_version_objects OBJECT src/rocjitsu/base/rj_version.cpp)
set_target_properties(
    rocjitsu_version_objects
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
    rocjitsu_version_objects
    PRIVATE ${ROCJITSU_INCLUDE_DIR} ${RJ_VERSION_INCLUDE_DIR}
)
rj_configure_target(rocjitsu_version_objects WARNINGS HIDDEN)

add_library(rocjitsu_version STATIC $<TARGET_OBJECTS:rocjitsu_version_objects>)
target_include_directories(
    rocjitsu_version
    PUBLIC
        $<BUILD_INTERFACE:${ROCJITSU_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Vendored DRM/amdgpu UAPI headers. kfd_ioctl.h includes <libdrm/drm.h>, so
# attach this target privately at the HSA include chokepoints.
add_library(rocjitsu_drm_headers INTERFACE)
target_include_directories(
    rocjitsu_drm_headers
    SYSTEM
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external_headers/drm_headers
)

add_subdirectory(src/rocjitsu/code/analysis)
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
    # Fixed-profile gfx1250 B0-to-A0 translator. Its default registry contains
    # only the gfx1250 model provider, excluding the full simulator aggregate and
    # execution objects.
    #
    # Shared rather than static because the translation cache keys entries on the
    # build id of the module that produced them. A static archive is copied into
    # every consumer, so each would carry a different build id and none could read
    # another's entries -- which is exactly what a translation performed ahead of
    # time needs to be able to do. One shared object gives every consumer the same
    # identity, derived by the linker from emitted content, so a rebuild that
    # changes translation output invalidates the cache with nobody maintaining a
    # version number.
    add_library(
        rocjitsu_gfx1250_b0_to_a0
        SHARED
        src/rocjitsu/code/dbt/gfx1250_b0_to_a0_library.cpp
    )
    foreach(_object_lib IN LISTS ROCJITSU_DBT_BASE_OBJECT_LIBS)
        target_sources(
            rocjitsu_gfx1250_b0_to_a0
            PRIVATE $<TARGET_OBJECTS:${_object_lib}>
        )
    endforeach()
    set_target_properties(
        rocjitsu_gfx1250_b0_to_a0
        PROPERTIES
            OUTPUT_NAME rocjitsu_gfx1250_b0_to_a0
            POSITION_INDEPENDENT_CODE ON
            VERSION ${PROJECT_VERSION}
            SOVERSION ${PROJECT_VERSION_MAJOR}
    )
    target_link_libraries(
        rocjitsu_gfx1250_b0_to_a0
        PRIVATE
            rocjitsu_cdna5_model_registry
            simdojo_headers
            util
            Threads::Threads
    )
    rj_configure_target(
        rocjitsu_gfx1250_b0_to_a0
        PRIVATE_INCLUDES
        WARNINGS
        HIDDEN
    )
    # --build-id is load bearing: the translation cache keys every entry on this
    # note. Without it the store finds no identity for the translator and turns
    # itself off, which would look like a cache that simply never hits.
    #
    # The version script is what keeps the exported surface to the C API. HIDDEN
    # above only governs this target's own sources; the object libraries linked in
    # are not built that way, so unscripted this library exports 126 symbols.
    set(_gfx1250_map ${CMAKE_CURRENT_SOURCE_DIR}/rocjitsu_gfx1250_b0_to_a0.map)
    set_property(
        TARGET rocjitsu_gfx1250_b0_to_a0
        APPEND
        PROPERTY LINK_DEPENDS ${_gfx1250_map}
    )
    target_link_options(
        rocjitsu_gfx1250_b0_to_a0
        PRIVATE
            "LINKER:--build-id=sha1"
            "LINKER:--version-script=${_gfx1250_map}"
    )
endif()

add_subdirectory(src/rocjitsu/vm)
add_subdirectory(src/rocjitsu/kmd)
add_subdirectory(src/rocjitsu/config)
add_subdirectory(src/rocjitsu/hooks)
add_subdirectory(src/rocjitsu/vmm)
