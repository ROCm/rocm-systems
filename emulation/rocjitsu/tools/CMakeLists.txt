# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

set(RJ_TOOL_OBJECT_LIBS
    rocjitsu_analysis
    rocjitsu_code
    rocjitsu_isa
    rocjitsu_amdgpu_isa_registry
    simdojo
    rocjitsu_vm_amdgpu
)

add_library(rocjitsu_tooling STATIC dbt_translate.cpp)
target_include_directories(rocjitsu_tooling PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rocjitsu_tooling PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(
    rocjitsu_tooling
    PUBLIC ${RJ_TOOL_OBJECT_LIBS} util flatbuffers Threads::Threads
)
rj_configure_target(rocjitsu_tooling TOOL)

add_library(rocjitsu_translate_cli STATIC dbt_translate_cli.cpp)
target_include_directories(
    rocjitsu_translate_cli
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(rocjitsu_translate_cli PUBLIC rocjitsu_tooling)
rj_configure_target(rocjitsu_translate_cli TOOL)

add_executable(rj_dbt_translate rj_dbt_translate_main.cpp)
target_link_libraries(rj_dbt_translate PRIVATE rocjitsu_translate_cli)
rj_configure_target(rj_dbt_translate TOOL)

add_executable(rj_waitcheck waitcheck_main.cpp)
target_sources(
    rj_waitcheck
    PRIVATE ${ROCJITSU_SRC_DIR}/rocjitsu/code/executable.cpp
)
target_link_libraries(
    rj_waitcheck
    PRIVATE
        rocjitsu_waitcheck_objects
        util
        Threads::Threads
        ZLIB::ZLIB
        zstd::libzstd
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(
        rj_waitcheck
        PRIVATE "LINKER:--gc-sections,--exclude-libs,ALL"
    )
endif()
rj_configure_target(rj_waitcheck TOOL)

add_executable(rj_co code_object_tool_main.cpp)
target_link_libraries(rj_co PRIVATE rocjitsu_tooling)
rj_configure_target(rj_co TOOL)

add_custom_target(
    rj_waitcheck_gfx950_tensile_e2e
    COMMAND
        ${CMAKE_SOURCE_DIR}/scripts/run_waitcheck_gfx950_tensile_e2e.sh
        --waitcheck $<TARGET_FILE:rj_waitcheck> --work-dir
        ${CMAKE_BINARY_DIR}/waitcheck-e2e/tensile-gfx950
    DEPENDS rj_waitcheck
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    USES_TERMINAL
    VERBATIM
    COMMENT "Build and waitcheck representative gfx950 Tensile kernels"
)

# Ahead-of-time population of the shared translation tier.
#
# Deliberately narrow: the fixed-profile translator, the store, and nothing else
# -- not rocjitsu_tooling, and not the hook, which contributes nothing to
# translation. The tier is keyed on the TRANSLATOR's build id, so this tool and
# the hook share entries only for as long as they resolve the same shared object;
# keeping the dependency list this short is what makes that easy to see.
#
# The store is compiled in rather than linked, as the hook does: neither wants a
# library dependency for one source file. TOOL is not used here because its
# generated-header and schema dependencies would assert a relationship this tool
# does not have.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(
        rj_pretranslate
        rj_pretranslate.cpp
        ${ROCJITSU_SRC_DIR}/rocjitsu/code/dbt/translation_store.cpp
    )
    target_link_libraries(
        rj_pretranslate
        PRIVATE rocjitsu_gfx1250_b0_to_a0 Threads::Threads ${CMAKE_DL_LIBS}
    )
    rj_configure_target(rj_pretranslate PRIVATE_INCLUDES WARNINGS)
    # Bind to the translator installed beside us. Two copies would carry two
    # build ids, and entries written against one would be invisible to a runtime
    # resolving the other -- a silent loss of every pre-translated object.
    set_target_properties(
        rj_pretranslate
        PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
    )
endif()
