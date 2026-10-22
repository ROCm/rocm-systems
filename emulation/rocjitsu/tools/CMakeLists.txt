# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

set(RJ_TOOL_OBJECT_LIBS
    rocjitsu_analysis
    rocjitsu_code
    rocjitsu_code_object
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

add_executable(rj_decode_benchmark rj_decode_benchmark.cpp)
target_link_libraries(
    rj_decode_benchmark
    PRIVATE rocjitsu_analysis_internal rocjitsu_amdgpu_isa_model_registry
)
rj_configure_target(rj_decode_benchmark PRIVATE_INCLUDES WARNINGS)

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

# Private shared-RAM keeper. Exec gives it an independent address space, so it
# cannot retain application mappings inherited when an allocation forks it.
add_executable(
    rocjitsu_memory
    rocjitsu_memory.cpp
    ${ROCJITSU_SRC_DIR}/rocjitsu/kmd/linux/memory_backing.cpp
    ${ROCJITSU_SRC_DIR}/rocjitsu/kmd/linux/libc_passthrough.cpp
)
target_link_libraries(
    rocjitsu_memory
    PRIVATE util Threads::Threads ${CMAKE_DL_LIBS}
)
rj_configure_target(rocjitsu_memory PRIVATE_INCLUDES WARNINGS)
set_target_properties(rocjitsu_memory PROPERTIES OUTPUT_NAME rocjitsu-memory)
file(
    RELATIVE_PATH
    _rj_keeper_relative
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBEXECDIR}/rocjitsu/rocjitsu-memory"
)
file(
    RELATIVE_PATH
    _rj_keeper_bin_relative
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}"
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBEXECDIR}/rocjitsu/rocjitsu-memory"
)
file(
    RELATIVE_PATH
    _rj_keeper_test_relative
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/rocjitsu/tests/bin"
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBEXECDIR}/rocjitsu/rocjitsu-memory"
)
foreach(_target rocjitsu_memory rocjitsu_kmd)
    target_compile_definitions(
        ${_target}
        PRIVATE
            RJ_MEMORY_KEEPER_BUILD_PATH="$<TARGET_FILE:rocjitsu_memory>"
            RJ_MEMORY_KEEPER_INSTALL_PATH="${_rj_keeper_relative}"
            RJ_MEMORY_KEEPER_BIN_INSTALL_PATH="${_rj_keeper_bin_relative}"
            RJ_MEMORY_KEEPER_TEST_INSTALL_PATH="${_rj_keeper_test_relative}"
    )
endforeach()
add_dependencies(rocjitsu_kmd rocjitsu_memory)
install(
    TARGETS rocjitsu_memory
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/rocjitsu
)
