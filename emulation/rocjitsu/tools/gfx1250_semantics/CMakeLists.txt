# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

find_package(Python3 COMPONENTS Interpreter QUIET)

set(RJ_GFX1250_SEMANTICS_CORPUS_DIR
    "${PROJECT_SOURCE_DIR}/../../../../gfx1250/gfx1250-semantics-corpus"
    CACHE PATH
    "gfx1250 semantics corpus checkout containing cases, artifacts, and golden outputs"
)

set(RJ_GFX1250_SEMANTICS_OBJECT_LIBS
    rocjitsu_analysis
    rocjitsu_code
    rocjitsu_isa
    rocjitsu_isa_cdna1
    rocjitsu_isa_cdna2
    rocjitsu_isa_cdna3
    rocjitsu_isa_cdna4
    rocjitsu_isa_rdna1
    rocjitsu_isa_rdna2
    rocjitsu_isa_rdna3
    rocjitsu_isa_rdna3_5
    rocjitsu_isa_rdna4
    rocjitsu_isa_gfx1250
    rocjitsu_isa_risc_v
    simdojo
    rocjitsu_vm
    rocjitsu_vm_amdgpu
    rocjitsu_vm_risc_v
    rocjitsu_config
)

find_library(
    RJ_GFX1250_HSA_RUNTIME64
    hsa-runtime64
    HINTS
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES lib lib64
)
if(RJ_GFX1250_HSA_RUNTIME64)
    get_filename_component(
        RJ_GFX1250_HSA_LIBRARY_DIR
        "${RJ_GFX1250_HSA_RUNTIME64}"
        DIRECTORY
    )
    get_filename_component(
        RJ_GFX1250_HSA_ROCM_ROOT
        "${RJ_GFX1250_HSA_LIBRARY_DIR}"
        DIRECTORY
    )
    set(RJ_GFX1250_HSA_INCLUDE_DIR "${RJ_GFX1250_HSA_ROCM_ROOT}/include")

    add_executable(gfx1250_semantics_dbt gfx1250_semantics_dbt.cpp)
    target_include_directories(
        gfx1250_semantics_dbt
        PRIVATE ${HSA_INCLUDE_DIR} ${RJ_GFX1250_HSA_INCLUDE_DIR}
    )
    target_link_libraries(
        gfx1250_semantics_dbt
        PRIVATE
            ${RJ_GFX1250_SEMANTICS_OBJECT_LIBS}
            util
            flatbuffers
            Threads::Threads
            ${RJ_GFX1250_HSA_RUNTIME64}
    )
    target_link_options(
        gfx1250_semantics_dbt
        PRIVATE -Wl,-rpath,${RJ_GFX1250_HSA_LIBRARY_DIR}
    )
    rj_configure_target(gfx1250_semantics_dbt TOOL)

    if(
        BUILD_TESTING
        AND Python3_Interpreter_FOUND
        AND EXISTS "${RJ_GFX1250_SEMANTICS_CORPUS_DIR}/cases"
    )
        set(RJ_GFX1250_SEMANTICS_DBT_CORPUS_OUTPUT_DIR
            "${CMAKE_BINARY_DIR}/gfx1250-semantics-dbt-corpus"
        )
        add_test(
            NAME Gfx1250SemanticsDbtCorpus
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/run_dbt_corpus.py --runner
                $<TARGET_FILE:gfx1250_semantics_dbt> --corpus-dir
                ${RJ_GFX1250_SEMANTICS_CORPUS_DIR} --output-dir
                ${RJ_GFX1250_SEMANTICS_DBT_CORPUS_OUTPUT_DIR}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        set_tests_properties(Gfx1250SemanticsDbtCorpus PROPERTIES TIMEOUT 300)
        add_custom_target(
            gfx1250_semantics_dbt_corpus
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/run_dbt_corpus.py --runner
                $<TARGET_FILE:gfx1250_semantics_dbt> --corpus-dir
                ${RJ_GFX1250_SEMANTICS_CORPUS_DIR} --output-dir
                ${RJ_GFX1250_SEMANTICS_DBT_CORPUS_OUTPUT_DIR}
            DEPENDS gfx1250_semantics_dbt
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            USES_TERMINAL
        )
        message(
            STATUS
            "gfx1250 semantics DBT corpus test enabled (${RJ_GFX1250_SEMANTICS_CORPUS_DIR})"
        )
    elseif(BUILD_TESTING)
        message(
            STATUS
            "gfx1250 semantics DBT corpus test not enabled "
            "(missing Python3, runner, or ${RJ_GFX1250_SEMANTICS_CORPUS_DIR}/cases)"
        )
    endif()
else()
    message(
        STATUS
        "gfx1250_semantics_dbt not built (libhsa-runtime64 not found)"
    )
endif()
