# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

set(ROCFUZZ_ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation used by rocfuzz smoke targets")
set(ROCFUZZ_GPU_ARCH "gfx1201" CACHE STRING "GPU architecture for the rocfuzz AFL DBI smoke HSACO")
set(ROCFUZZ_RDNA3_VISIBLE_DEVICES "1" CACHE STRING
    "HIP_VISIBLE_DEVICES selector used by the optional gfx1100 rocfuzz smoke target")
set(ROCFUZZ_LAZY_CCOB_VISIBLE_DEVICES "${ROCFUZZ_RDNA3_VISIBLE_DEVICES}" CACHE STRING
    "HIP_VISIBLE_DEVICES selector used by the optional lazy CCOB lifecycle smoke target")
set(ROCFUZZ_LAZY_CCOB_MODULE
    "${CMAKE_CURRENT_SOURCE_DIR}/../examples/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/rocblas/library/TensileLibrary_Type_SS_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1100.co"
    CACHE FILEPATH "HIP-loadable CCOB module used by the optional lazy CCOB lifecycle smoke target")
set(ROCFUZZ_REAL_CCOB_ROCBLAS_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../examples/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/rocblas/library"
    CACHE PATH "Directory of real rocBLAS CCOB library assets used by the optional inventory smoke")
set(ROCFUZZ_REAL_CCOB_HIPBLASLT_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../examples/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/hipblaslt/library"
    CACHE PATH "Directory of real hipBLASLt CCOB library assets used by the optional inventory smoke")
set(ROCFUZZ_REAL_CCOB_SHARED_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../shared/kpack/test_assets/ccob"
    CACHE PATH "Directory of shared CCOB test assets used by the optional inventory smoke")
set(ROCFUZZ_AFL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/AFLplusplus"
    CACHE PATH "Optional AFL++ checkout used by rocfuzz AFL showmap smoke targets")

find_program(ROCFUZZ_HIPCC_EXECUTABLE hipcc
    HINTS "${ROCFUZZ_ROCM_PATH}" ENV ROCM_PATH
    PATH_SUFFIXES bin)
find_library(ROCFUZZ_HIP_RUNTIME amdhip64
    HINTS "${ROCFUZZ_ROCM_PATH}" ENV ROCM_PATH
    PATH_SUFFIXES lib lib64)
find_library(ROCFUZZ_HSA_RUNTIME hsa-runtime64
    HINTS "${ROCFUZZ_ROCM_PATH}" ENV ROCM_PATH
    PATH_SUFFIXES lib lib64)
find_library(ROCFUZZ_ZSTD_LIBRARY zstd)
find_path(ROCFUZZ_ZSTD_INCLUDE_DIR zstd.h)
find_package(OpenSSL QUIET COMPONENTS Crypto)
find_program(ROCFUZZ_AFL_CXX_EXECUTABLE afl-clang-fast++
    HINTS "${ROCFUZZ_AFL_ROOT}")
find_program(ROCFUZZ_AFL_SHOWMAP_EXECUTABLE afl-showmap
    HINTS "${ROCFUZZ_AFL_ROOT}")
find_program(ROCFUZZ_PYTHON3_EXECUTABLE python3)

set(ROCFUZZ_RJ_OBJECT_LIBS
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
    rocjitsu_isa_risc_v
    simdojo
    rocjitsu_vm
    rocjitsu_vm_amdgpu
    rocjitsu_vm_risc_v
    rocjitsu_config)

if(ROCFUZZ_HIP_RUNTIME AND ROCFUZZ_ZSTD_LIBRARY AND ROCFUZZ_ZSTD_INCLUDE_DIR AND
        OpenSSL_FOUND)
    add_library(rocjitsu_afl_preload SHARED
        runtime/afl_preload.cpp
        runtime/code_object_image.cpp
        runtime/instruction_relocator.cpp
        runtime/instrumentation_planner.cpp
        runtime/md5_util.cpp
        runtime/patchability_classifier.cpp
        runtime/vopd_liveness.cpp)
    target_include_directories(rocjitsu_afl_preload PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime
        ${ROCFUZZ_ZSTD_INCLUDE_DIR}
        "${ROCFUZZ_ROCM_PATH}/include")
    target_link_libraries(rocjitsu_afl_preload PRIVATE
        ${ROCFUZZ_RJ_OBJECT_LIBS}
        util
        flatbuffers
        Threads::Threads
        ${CMAKE_DL_LIBS}
        ${ROCFUZZ_ZSTD_LIBRARY}
        OpenSSL::Crypto
        ${ROCFUZZ_HIP_RUNTIME})
    rj_configure_target(rocjitsu_afl_preload PRIVATE_INCLUDES)
    target_compile_definitions(rocjitsu_afl_preload PRIVATE __HIP_PLATFORM_AMD__)
    set_target_properties(rocjitsu_afl_preload PROPERTIES
        OUTPUT_NAME rocjitsu_afl_preload)
else()
    if(NOT ROCFUZZ_HIP_RUNTIME)
        message(STATUS "rocfuzz AFL DBI preload disabled (amdhip64 not found)")
    elseif(NOT ROCFUZZ_ZSTD_LIBRARY OR NOT ROCFUZZ_ZSTD_INCLUDE_DIR)
        message(STATUS "rocfuzz AFL DBI preload disabled (zstd not found)")
    elseif(NOT OpenSSL_FOUND)
        message(STATUS "rocfuzz AFL DBI preload disabled (OpenSSL Crypto not found)")
    endif()
endif()

add_executable(rocjitsu_afl_edge_plan_unit tests/edge_plan_unit.cpp)
target_include_directories(rocjitsu_afl_edge_plan_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
rj_configure_target(rocjitsu_afl_edge_plan_unit PRIVATE_INCLUDES)
add_test(NAME RocjitsuAflDbi.EdgePlanUnit COMMAND rocjitsu_afl_edge_plan_unit)

add_executable(rocjitsu_afl_patchability_classifier_unit
    tests/patchability_classifier_unit.cpp
    runtime/patchability_classifier.cpp)
target_include_directories(rocjitsu_afl_patchability_classifier_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime)
rj_configure_target(rocjitsu_afl_patchability_classifier_unit PRIVATE_INCLUDES)
add_test(NAME RocjitsuAflDbi.PatchabilityClassifierUnit
    COMMAND rocjitsu_afl_patchability_classifier_unit)

add_executable(rocjitsu_afl_probe_decode_unit
    tests/probe_decode_unit.cpp
    runtime/instruction_relocator.cpp
    runtime/instrumentation_planner.cpp
    runtime/vopd_liveness.cpp)
target_include_directories(rocjitsu_afl_probe_decode_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime)
target_link_libraries(rocjitsu_afl_probe_decode_unit PRIVATE
    ${ROCFUZZ_RJ_OBJECT_LIBS}
    util
    flatbuffers
    Threads::Threads
    ${CMAKE_DL_LIBS})
rj_configure_target(rocjitsu_afl_probe_decode_unit PRIVATE_INCLUDES)
add_test(NAME RocjitsuAflDbi.ProbeDecodeUnit COMMAND rocjitsu_afl_probe_decode_unit)

add_executable(rocjitsu_afl_instruction_relocator_unit
    tests/instruction_relocator_unit.cpp
    runtime/instruction_relocator.cpp
    runtime/vopd_liveness.cpp)
target_include_directories(rocjitsu_afl_instruction_relocator_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime)
target_link_libraries(rocjitsu_afl_instruction_relocator_unit PRIVATE
    ${ROCFUZZ_RJ_OBJECT_LIBS}
    util
    flatbuffers
    Threads::Threads
    ${CMAKE_DL_LIBS})
rj_configure_target(rocjitsu_afl_instruction_relocator_unit PRIVATE_INCLUDES)
add_test(NAME RocjitsuAflDbi.InstructionRelocatorUnit
    COMMAND rocjitsu_afl_instruction_relocator_unit)

add_executable(rocjitsu_afl_patch_plan_transaction_unit
    tests/patch_plan_transaction_unit.cpp
    runtime/instruction_relocator.cpp
    runtime/instrumentation_planner.cpp
    runtime/vopd_liveness.cpp)
target_include_directories(rocjitsu_afl_patch_plan_transaction_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime)
target_link_libraries(rocjitsu_afl_patch_plan_transaction_unit PRIVATE
    ${ROCFUZZ_RJ_OBJECT_LIBS}
    util
    flatbuffers
    Threads::Threads
    ${CMAKE_DL_LIBS})
rj_configure_target(rocjitsu_afl_patch_plan_transaction_unit PRIVATE_INCLUDES)
add_test(NAME RocjitsuAflDbi.PatchPlanTransactionUnit
    COMMAND rocjitsu_afl_patch_plan_transaction_unit)

if(ROCFUZZ_PYTHON3_EXECUTABLE)
    add_custom_target(rocjitsu_afl_patch_report_summary_unit
        COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/patch_report_summary_unit.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/summarize_patch_report.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/patch_report_summary_fixture.jsonl
        DEPENDS
                tests/patch_report_summary_unit.py
                tools/summarize_patch_report.py
                tests/patch_report_summary_fixture.jsonl
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running rocfuzz patch-report summary unit test"
        VERBATIM)
    add_test(NAME RocjitsuAflDbi.PatchReportSummaryUnit
        COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/patch_report_summary_unit.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/summarize_patch_report.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/patch_report_summary_fixture.jsonl)
    add_custom_target(rocjitsu_afl_patch_report_events_unit
        COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_patch_report_events_unit.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
        DEPENDS
                tests/check_patch_report_events_unit.py
                tools/check_patch_report_events.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running rocfuzz patch-report event checker unit test"
        VERBATIM)
    add_test(NAME RocjitsuAflDbi.PatchReportEventsUnit
        COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_patch_report_events_unit.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py)
    add_custom_target(rocjitsu_afl_opaque_fresh_minimizer_unit
        COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/minimize_opaque_fresh_growth_unit.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/minimize_opaque_fresh_growth.py
        DEPENDS
                tests/minimize_opaque_fresh_growth_unit.py
                tools/minimize_opaque_fresh_growth.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running rocfuzz branch-edge minimizer unit test"
        VERBATIM)
    add_custom_target(rocjitsu_afl_edge_minimizer_unit
        DEPENDS
                rocjitsu_afl_opaque_fresh_minimizer_unit)
    add_test(NAME RocjitsuAflDbi.OpaqueFreshMinimizerUnit
        COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/minimize_opaque_fresh_growth_unit.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/minimize_opaque_fresh_growth.py)
    add_test(NAME RocjitsuAflDbi.EdgeMinimizerUnit
        COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/minimize_opaque_fresh_growth_unit.py
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/minimize_opaque_fresh_growth.py)
else()
    message(STATUS "rocfuzz patch-report summary unit disabled (python3 not found)")
endif()

if(ROCFUZZ_ZSTD_LIBRARY AND ROCFUZZ_ZSTD_INCLUDE_DIR AND OpenSSL_FOUND)
    add_executable(rocjitsu_afl_make_ccob_bundle
        tools/make_ccob_bundle.cpp
        runtime/md5_util.cpp)
    target_include_directories(rocjitsu_afl_make_ccob_bundle PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime
        ${ROCFUZZ_ZSTD_INCLUDE_DIR})
    target_link_libraries(rocjitsu_afl_make_ccob_bundle PRIVATE
        ${ROCFUZZ_ZSTD_LIBRARY}
        OpenSSL::Crypto)
    rj_configure_target(rocjitsu_afl_make_ccob_bundle PRIVATE_INCLUDES)

    add_executable(rocjitsu_afl_inspect_code_object_image
        tools/inspect_code_object_image.cpp
        runtime/code_object_image.cpp
        runtime/md5_util.cpp)
    target_include_directories(rocjitsu_afl_inspect_code_object_image PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime
        ${ROCFUZZ_ZSTD_INCLUDE_DIR})
    target_link_libraries(rocjitsu_afl_inspect_code_object_image PRIVATE
        ${ROCFUZZ_ZSTD_LIBRARY}
        OpenSSL::Crypto)
    rj_configure_target(rocjitsu_afl_inspect_code_object_image PRIVATE_INCLUDES)

    if(ROCFUZZ_PYTHON3_EXECUTABLE)
        set(ROCFUZZ_REAL_CCOB_FILES)
        foreach(real_ccob_dir IN LISTS
                ROCFUZZ_REAL_CCOB_ROCBLAS_DIR
                ROCFUZZ_REAL_CCOB_HIPBLASLT_DIR
                ROCFUZZ_REAL_CCOB_SHARED_DIR)
            if(EXISTS "${real_ccob_dir}")
                file(GLOB real_ccob_files CONFIGURE_DEPENDS
                    "${real_ccob_dir}/*.co")
                list(APPEND ROCFUZZ_REAL_CCOB_FILES ${real_ccob_files})
            endif()
        endforeach()
        list(REMOVE_DUPLICATES ROCFUZZ_REAL_CCOB_FILES)
        if(ROCFUZZ_REAL_CCOB_FILES)
            add_custom_target(rocjitsu_afl_real_ccob_inventory_smoke
                COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_code_object_inventory.py
                        --inspector $<TARGET_FILE:rocjitsu_afl_inspect_code_object_image>
                        --summary-label real-ccob-inventory
                        --min-files 1
                        --require-any-top-level-ccob
                        --require-device-images
                        ${ROCFUZZ_REAL_CCOB_FILES}
                DEPENDS rocjitsu_afl_inspect_code_object_image
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Inspecting real rocfuzz CCOB library inventory"
                VERBATIM)
        else()
            message(STATUS "rocfuzz real CCOB inventory smoke disabled (no .co files found)")
        endif()
    else()
        message(STATUS "rocfuzz real CCOB inventory smoke disabled (python3 not found)")
    endif()

    add_executable(rocjitsu_afl_code_object_image_unit
        tests/code_object_image_unit.cpp
        runtime/code_object_image.cpp
        runtime/md5_util.cpp)
    target_include_directories(rocjitsu_afl_code_object_image_unit PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime
        ${ROCFUZZ_ZSTD_INCLUDE_DIR})
    target_link_libraries(rocjitsu_afl_code_object_image_unit PRIVATE
        ${ROCFUZZ_ZSTD_LIBRARY}
        OpenSSL::Crypto)
    rj_configure_target(rocjitsu_afl_code_object_image_unit PRIVATE_INCLUDES)
    add_test(NAME RocjitsuAflDbi.CodeObjectImageUnit COMMAND rocjitsu_afl_code_object_image_unit)
else()
    message(STATUS "rocfuzz CCOB image tests disabled (zstd or OpenSSL Crypto not found)")
endif()

if(ROCFUZZ_HIPCC_EXECUTABLE AND ROCFUZZ_HIP_RUNTIME AND
        ROCFUZZ_ZSTD_LIBRARY AND ROCFUZZ_ZSTD_INCLUDE_DIR AND OpenSSL_FOUND)
    set(ROCFUZZ_SMOKE_HSACO
        ${CMAKE_CURRENT_BINARY_DIR}/branchy_kernels.${ROCFUZZ_GPU_ARCH}.hsaco)
    set(ROCFUZZ_SMOKE_CCOB
        ${CMAKE_CURRENT_BINARY_DIR}/branchy_kernels.${ROCFUZZ_GPU_ARCH}.co)
    set(ROCFUZZ_SMOKE_MULTI_CCOB
        ${CMAKE_CURRENT_BINARY_DIR}/branchy_kernels.${ROCFUZZ_GPU_ARCH}.multi.co)
    set(ROCFUZZ_SMOKE_RDC_MULTI_BUNDLE
        ${CMAKE_CURRENT_BINARY_DIR}/branchy_kernels.${ROCFUZZ_GPU_ARCH}.rdc.multi.bundle)
    set(ROCFUZZ_SMOKE_RDC_MULTI_CCOB
        ${CMAKE_CURRENT_BINARY_DIR}/branchy_kernels.${ROCFUZZ_GPU_ARCH}.rdc.multi.co)
    set(ROCFUZZ_SMOKE_WAVE64_HSACO
        ${CMAKE_CURRENT_BINARY_DIR}/branchy_kernels.${ROCFUZZ_GPU_ARCH}.wave64.hsaco)
    set(ROCFUZZ_WAVE64_DEFAULT_REPORT
        ${CMAKE_CURRENT_BINARY_DIR}/wave64_default_report.jsonl)
    set(ROCFUZZ_WAVE64_PREVIOUS_BB_REPORT
        ${CMAKE_CURRENT_BINARY_DIR}/wave64_previous_bb_report.jsonl)
    set(ROCFUZZ_HIGH_LANE_WAVE64_HSACO
        ${CMAKE_CURRENT_BINARY_DIR}/high_lane_kernels.${ROCFUZZ_GPU_ARCH}.wave64.hsaco)
    set(ROCFUZZ_ENTRY_LIVENESS_HSACO
        ${CMAKE_CURRENT_BINARY_DIR}/entry_liveness_kernels.${ROCFUZZ_GPU_ARCH}.hsaco)
    set(ROCFUZZ_SGPR_GROWTH_HSACO
        ${CMAKE_CURRENT_BINARY_DIR}/sgpr_growth_kernels.${ROCFUZZ_GPU_ARCH}.hsaco)
    set(ROCFUZZ_SMOKE_RDNA3_HSACO
        ${CMAKE_CURRENT_BINARY_DIR}/branchy_kernels.gfx1100.hsaco)
    set(ROCFUZZ_SMOKE_RDNA3_WAVE64_HSACO
        ${CMAKE_CURRENT_BINARY_DIR}/branchy_kernels.gfx1100.wave64.hsaco)
    set(ROCFUZZ_RDNA3_WAVE64_PREVIOUS_BB_REPORT
        ${CMAKE_CURRENT_BINARY_DIR}/wave64_previous_bb_gfx1100_report.jsonl)
    set(ROCFUZZ_HIGH_LANE_RDNA3_WAVE64_HSACO
        ${CMAKE_CURRENT_BINARY_DIR}/high_lane_kernels.gfx1100.wave64.hsaco)
    set(ROCFUZZ_SMOKE_SEED ${CMAKE_CURRENT_BINARY_DIR}/seed)
    set(ROCFUZZ_HIGH_LANE_SEED_A ${CMAKE_CURRENT_BINARY_DIR}/seed.high_lane.a)
    set(ROCFUZZ_HIGH_LANE_SEED_B ${CMAKE_CURRENT_BINARY_DIR}/seed.high_lane.b)
    set(ROCFUZZ_FATBIN_HOST
        ${CMAKE_CURRENT_BINARY_DIR}/branchy_fatbin_host)
    set(ROCFUZZ_PERSISTENT_SEED_A ${CMAKE_CURRENT_BINARY_DIR}/seed.persistent.a)
    set(ROCFUZZ_PERSISTENT_SEED_B ${CMAKE_CURRENT_BINARY_DIR}/seed.persistent.b)
    file(WRITE ${ROCFUZZ_SMOKE_SEED} "123")
    file(WRITE ${ROCFUZZ_HIGH_LANE_SEED_A} "0")
    file(WRITE ${ROCFUZZ_HIGH_LANE_SEED_B} "1")
    file(WRITE ${ROCFUZZ_PERSISTENT_SEED_A} "00000000")
    file(WRITE ${ROCFUZZ_PERSISTENT_SEED_B} "11111111")
    set(ROCFUZZ_SMOKE_RDC_PEER_ARCH "gfx1100")
    if(ROCFUZZ_GPU_ARCH STREQUAL ROCFUZZ_SMOKE_RDC_PEER_ARCH)
        set(ROCFUZZ_SMOKE_RDC_PEER_ARCH "gfx1201")
    endif()

    add_custom_command(
        OUTPUT ${ROCFUZZ_SMOKE_HSACO}
        COMMAND ${ROCFUZZ_HIPCC_EXECUTABLE}
                --genco
                --offload-arch=${ROCFUZZ_GPU_ARCH}
                -O2
                ${CMAKE_CURRENT_SOURCE_DIR}/smoke/branchy_kernels.hip
                -o ${ROCFUZZ_SMOKE_HSACO}
        DEPENDS smoke/branchy_kernels.hip
        COMMENT "Building rocfuzz AFL DBI smoke HSACO for ${ROCFUZZ_GPU_ARCH}"
        VERBATIM)
    add_custom_target(rocjitsu_afl_branchy_kernels_hsaco ALL
        DEPENDS ${ROCFUZZ_SMOKE_HSACO})

    add_custom_command(
        OUTPUT ${ROCFUZZ_SMOKE_CCOB}
        COMMAND $<TARGET_FILE:rocjitsu_afl_make_ccob_bundle>
                ${ROCFUZZ_SMOKE_HSACO}
                ${ROCFUZZ_SMOKE_CCOB}
                hipv4-amdgcn-amd-amdhsa--${ROCFUZZ_GPU_ARCH}
        DEPENDS
                rocjitsu_afl_make_ccob_bundle
                ${ROCFUZZ_SMOKE_HSACO}
        COMMENT "Wrapping rocfuzz AFL DBI smoke HSACO in a CCOB container"
        VERBATIM)
    add_custom_target(rocjitsu_afl_branchy_kernels_ccob
        DEPENDS ${ROCFUZZ_SMOKE_CCOB})

    add_custom_command(
        OUTPUT ${ROCFUZZ_SMOKE_MULTI_CCOB}
        COMMAND $<TARGET_FILE:rocjitsu_afl_make_ccob_bundle>
                --multi
                ${ROCFUZZ_SMOKE_MULTI_CCOB}
                ${ROCFUZZ_SMOKE_HSACO}
                hipv4-amdgcn-amd-amdhsa--${ROCFUZZ_GPU_ARCH}
                ${ROCFUZZ_SMOKE_RDNA3_HSACO}
                hipv4-amdgcn-amd-amdhsa--gfx1100
        DEPENDS
                rocjitsu_afl_make_ccob_bundle
                ${ROCFUZZ_SMOKE_HSACO}
                ${ROCFUZZ_SMOKE_RDNA3_HSACO}
        COMMENT "Wrapping rocfuzz AFL DBI smoke HSACOs in a multi-payload CCOB container"
        VERBATIM)
    add_custom_target(rocjitsu_afl_branchy_kernels_multi_ccob
        DEPENDS ${ROCFUZZ_SMOKE_MULTI_CCOB})

    add_custom_command(
        OUTPUT ${ROCFUZZ_SMOKE_RDC_MULTI_BUNDLE}
        COMMAND ${ROCFUZZ_HIPCC_EXECUTABLE}
                --genco
                -fgpu-rdc
                --offload-arch=${ROCFUZZ_GPU_ARCH}
                --offload-arch=${ROCFUZZ_SMOKE_RDC_PEER_ARCH}
                -O2
                ${CMAKE_CURRENT_SOURCE_DIR}/smoke/branchy_kernels.hip
                -o ${ROCFUZZ_SMOKE_RDC_MULTI_BUNDLE}
        DEPENDS smoke/branchy_kernels.hip
        COMMENT "Building rocfuzz AFL DBI RDC multi-arch offload bundle"
        VERBATIM)
    add_custom_target(rocjitsu_afl_branchy_kernels_rdc_multi_bundle
        DEPENDS ${ROCFUZZ_SMOKE_RDC_MULTI_BUNDLE})

    add_custom_command(
        OUTPUT ${ROCFUZZ_SMOKE_RDC_MULTI_CCOB}
        COMMAND $<TARGET_FILE:rocjitsu_afl_make_ccob_bundle>
                ${ROCFUZZ_SMOKE_RDC_MULTI_BUNDLE}
                ${ROCFUZZ_SMOKE_RDC_MULTI_CCOB}
                hipv4-amdgcn-amd-amdhsa--${ROCFUZZ_GPU_ARCH}
        DEPENDS
                rocjitsu_afl_make_ccob_bundle
                ${ROCFUZZ_SMOKE_RDC_MULTI_BUNDLE}
        COMMENT "Wrapping compiler-produced RDC multi-arch bundle in a CCOB container"
        VERBATIM)
    add_custom_target(rocjitsu_afl_branchy_kernels_rdc_multi_ccob
        DEPENDS ${ROCFUZZ_SMOKE_RDC_MULTI_CCOB})

    add_custom_command(
        OUTPUT ${ROCFUZZ_FATBIN_HOST}
        COMMAND ${ROCFUZZ_HIPCC_EXECUTABLE}
                --offload-arch=${ROCFUZZ_GPU_ARCH}
                -O2
                -pthread
                ${CMAKE_CURRENT_SOURCE_DIR}/smoke/branchy_fatbin.hip
                -o ${ROCFUZZ_FATBIN_HOST}
        DEPENDS smoke/branchy_fatbin.hip
        COMMENT "Building rocfuzz AFL DBI compiler-registered HIP fatbin smoke"
        VERBATIM)
    add_custom_target(rocjitsu_afl_branchy_fatbin_host
        DEPENDS ${ROCFUZZ_FATBIN_HOST})

    add_custom_command(
        OUTPUT ${ROCFUZZ_SMOKE_WAVE64_HSACO}
        COMMAND ${ROCFUZZ_HIPCC_EXECUTABLE}
                --genco
                --offload-arch=${ROCFUZZ_GPU_ARCH}
                -mwavefrontsize64
                -O2
                ${CMAKE_CURRENT_SOURCE_DIR}/smoke/branchy_kernels.hip
                -o ${ROCFUZZ_SMOKE_WAVE64_HSACO}
        DEPENDS smoke/branchy_kernels.hip
        COMMENT "Building rocfuzz AFL DBI wave64 smoke HSACO for ${ROCFUZZ_GPU_ARCH}"
        VERBATIM)
    add_custom_target(rocjitsu_afl_branchy_kernels_wave64_hsaco
        DEPENDS ${ROCFUZZ_SMOKE_WAVE64_HSACO})

    add_custom_command(
        OUTPUT ${ROCFUZZ_SMOKE_RDNA3_HSACO}
        COMMAND ${ROCFUZZ_HIPCC_EXECUTABLE}
                --genco
                --offload-arch=gfx1100
                -O2
                ${CMAKE_CURRENT_SOURCE_DIR}/smoke/branchy_kernels.hip
                -o ${ROCFUZZ_SMOKE_RDNA3_HSACO}
        DEPENDS smoke/branchy_kernels.hip
        COMMENT "Building rocfuzz AFL DBI smoke HSACO for gfx1100"
        VERBATIM)
    add_custom_target(rocjitsu_afl_branchy_kernels_rdna3_hsaco
        DEPENDS ${ROCFUZZ_SMOKE_RDNA3_HSACO})

    add_custom_command(
        OUTPUT ${ROCFUZZ_SMOKE_RDNA3_WAVE64_HSACO}
        COMMAND ${ROCFUZZ_HIPCC_EXECUTABLE}
                --genco
                --offload-arch=gfx1100
                -mwavefrontsize64
                -O2
                ${CMAKE_CURRENT_SOURCE_DIR}/smoke/branchy_kernels.hip
                -o ${ROCFUZZ_SMOKE_RDNA3_WAVE64_HSACO}
        DEPENDS smoke/branchy_kernels.hip
        COMMENT "Building rocfuzz AFL DBI wave64 smoke HSACO for gfx1100"
        VERBATIM)
    add_custom_target(rocjitsu_afl_branchy_kernels_rdna3_wave64_hsaco
        DEPENDS ${ROCFUZZ_SMOKE_RDNA3_WAVE64_HSACO})

    add_custom_command(
        OUTPUT ${ROCFUZZ_HIGH_LANE_WAVE64_HSACO}
        COMMAND ${ROCFUZZ_HIPCC_EXECUTABLE}
                --genco
                --offload-arch=${ROCFUZZ_GPU_ARCH}
                -mwavefrontsize64
                -O2
                ${CMAKE_CURRENT_SOURCE_DIR}/smoke/high_lane_kernels.hip
                -o ${ROCFUZZ_HIGH_LANE_WAVE64_HSACO}
        DEPENDS smoke/high_lane_kernels.hip
        COMMENT "Building rocfuzz AFL DBI high-lane wave64 smoke HSACO for ${ROCFUZZ_GPU_ARCH}"
        VERBATIM)
    add_custom_target(rocjitsu_afl_high_lane_kernels_wave64_hsaco
        DEPENDS ${ROCFUZZ_HIGH_LANE_WAVE64_HSACO})

    add_custom_command(
        OUTPUT ${ROCFUZZ_ENTRY_LIVENESS_HSACO}
        COMMAND ${ROCFUZZ_HIPCC_EXECUTABLE}
                --genco
                --offload-arch=${ROCFUZZ_GPU_ARCH}
                -O2
                ${CMAKE_CURRENT_SOURCE_DIR}/smoke/entry_liveness_kernels.hip
                -o ${ROCFUZZ_ENTRY_LIVENESS_HSACO}
        DEPENDS smoke/entry_liveness_kernels.hip
        COMMENT "Building rocfuzz AFL DBI entry-liveness smoke HSACO for ${ROCFUZZ_GPU_ARCH}"
        VERBATIM)
    add_custom_target(rocjitsu_afl_entry_liveness_kernels_hsaco
        DEPENDS ${ROCFUZZ_ENTRY_LIVENESS_HSACO})

    # Keep this diagnostic on decoded instructions so it isolates SGPR descriptor
    # growth rather than the opaque/VOPD fallback.
    add_custom_command(
        OUTPUT ${ROCFUZZ_SGPR_GROWTH_HSACO}
        COMMAND ${ROCFUZZ_HIPCC_EXECUTABLE}
                --genco
                --offload-arch=${ROCFUZZ_GPU_ARCH}
                -O2
                -mllvm
                -amdgpu-enable-vopd=false
                ${CMAKE_CURRENT_SOURCE_DIR}/smoke/sgpr_growth_kernels.hip
                -o ${ROCFUZZ_SGPR_GROWTH_HSACO}
        DEPENDS smoke/sgpr_growth_kernels.hip
        COMMENT "Building rocfuzz AFL DBI SGPR-growth smoke HSACO for ${ROCFUZZ_GPU_ARCH}"
        VERBATIM)
    add_custom_target(rocjitsu_afl_sgpr_growth_kernels_hsaco
        DEPENDS ${ROCFUZZ_SGPR_GROWTH_HSACO})

    add_custom_command(
        OUTPUT ${ROCFUZZ_HIGH_LANE_RDNA3_WAVE64_HSACO}
        COMMAND ${ROCFUZZ_HIPCC_EXECUTABLE}
                --genco
                --offload-arch=gfx1100
                -mwavefrontsize64
                -O2
                ${CMAKE_CURRENT_SOURCE_DIR}/smoke/high_lane_kernels.hip
                -o ${ROCFUZZ_HIGH_LANE_RDNA3_WAVE64_HSACO}
        DEPENDS smoke/high_lane_kernels.hip
        COMMENT "Building rocfuzz AFL DBI high-lane wave64 smoke HSACO for gfx1100"
        VERBATIM)
    add_custom_target(rocjitsu_afl_high_lane_kernels_rdna3_wave64_hsaco
        DEPENDS ${ROCFUZZ_HIGH_LANE_RDNA3_WAVE64_HSACO})

    add_executable(rocjitsu_afl_branchy_host smoke/branchy_host.cpp)
    target_compile_definitions(rocjitsu_afl_branchy_host PRIVATE __HIP_PLATFORM_AMD__)
    target_include_directories(rocjitsu_afl_branchy_host PRIVATE
        "${ROCFUZZ_ROCM_PATH}/include")
    target_link_libraries(rocjitsu_afl_branchy_host PRIVATE ${ROCFUZZ_HIP_RUNTIME})
    add_dependencies(rocjitsu_afl_branchy_host rocjitsu_afl_branchy_kernels_hsaco)
    set(ROCFUZZ_BRANCHY_EXPECTED_RESULT 494549689)

    add_executable(rocjitsu_afl_high_lane_host smoke/high_lane_host.cpp)
    target_compile_definitions(rocjitsu_afl_high_lane_host PRIVATE __HIP_PLATFORM_AMD__)
    target_include_directories(rocjitsu_afl_high_lane_host PRIVATE
        "${ROCFUZZ_ROCM_PATH}/include")
    target_link_libraries(rocjitsu_afl_high_lane_host PRIVATE ${ROCFUZZ_HIP_RUNTIME})
    add_dependencies(rocjitsu_afl_high_lane_host rocjitsu_afl_high_lane_kernels_wave64_hsaco)

    add_executable(rocjitsu_afl_entry_liveness_host smoke/entry_liveness_host.cpp)
    target_compile_definitions(rocjitsu_afl_entry_liveness_host PRIVATE __HIP_PLATFORM_AMD__)
    target_include_directories(rocjitsu_afl_entry_liveness_host PRIVATE
        "${ROCFUZZ_ROCM_PATH}/include")
    target_link_libraries(rocjitsu_afl_entry_liveness_host PRIVATE ${ROCFUZZ_HIP_RUNTIME})
    add_dependencies(rocjitsu_afl_entry_liveness_host rocjitsu_afl_entry_liveness_kernels_hsaco)

    add_executable(rocjitsu_afl_sgpr_growth_host smoke/sgpr_growth_host.cpp)
    target_compile_definitions(rocjitsu_afl_sgpr_growth_host PRIVATE __HIP_PLATFORM_AMD__)
    target_include_directories(rocjitsu_afl_sgpr_growth_host PRIVATE
        "${ROCFUZZ_ROCM_PATH}/include")
    target_link_libraries(rocjitsu_afl_sgpr_growth_host PRIVATE ${ROCFUZZ_HIP_RUNTIME})
    add_dependencies(rocjitsu_afl_sgpr_growth_host rocjitsu_afl_sgpr_growth_kernels_hsaco)

    add_executable(rocjitsu_afl_metadata_resize_loader_host
        smoke/metadata_resize_loader_smoke.cpp
        runtime/code_object_image.cpp
        runtime/instruction_relocator.cpp
        runtime/instrumentation_planner.cpp
        runtime/md5_util.cpp
        runtime/vopd_liveness.cpp)
    target_compile_definitions(rocjitsu_afl_metadata_resize_loader_host
        PRIVATE __HIP_PLATFORM_AMD__)
    target_include_directories(rocjitsu_afl_metadata_resize_loader_host PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime
        ${ROCFUZZ_ZSTD_INCLUDE_DIR}
        "${ROCFUZZ_ROCM_PATH}/include")
    target_link_libraries(rocjitsu_afl_metadata_resize_loader_host PRIVATE
        ${ROCFUZZ_RJ_OBJECT_LIBS}
        util
        flatbuffers
        Threads::Threads
        ${CMAKE_DL_LIBS}
        ${ROCFUZZ_ZSTD_LIBRARY}
        OpenSSL::Crypto
        ${ROCFUZZ_HIP_RUNTIME})
    if(ROCFUZZ_HSA_RUNTIME)
        target_compile_definitions(rocjitsu_afl_metadata_resize_loader_host
            PRIVATE ROCFUZZ_ENABLE_HSA_READER_CHECK=1)
        target_link_libraries(rocjitsu_afl_metadata_resize_loader_host PRIVATE
            ${ROCFUZZ_HSA_RUNTIME})
    endif()
    rj_configure_target(rocjitsu_afl_metadata_resize_loader_host PRIVATE_INCLUDES)
    add_dependencies(rocjitsu_afl_metadata_resize_loader_host
        rocjitsu_afl_branchy_kernels_hsaco
        rocjitsu_afl_branchy_kernels_ccob)

    add_executable(rocjitsu_afl_ccob_module_probe
        smoke/ccob_module_probe.cpp
        runtime/code_object_image.cpp
        runtime/md5_util.cpp)
    target_compile_definitions(rocjitsu_afl_ccob_module_probe PRIVATE __HIP_PLATFORM_AMD__)
    target_include_directories(rocjitsu_afl_ccob_module_probe PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime
        ${ROCFUZZ_ZSTD_INCLUDE_DIR}
        "${ROCFUZZ_ROCM_PATH}/include")
    target_link_libraries(rocjitsu_afl_ccob_module_probe PRIVATE
        ${ROCFUZZ_ZSTD_LIBRARY}
        OpenSSL::Crypto
        ${ROCFUZZ_HIP_RUNTIME})
    rj_configure_target(rocjitsu_afl_ccob_module_probe PRIVATE_INCLUDES)

    add_executable(rocjitsu_afl_branchy_persistent_host
        smoke/branchy_persistent_host.cpp)
    target_compile_definitions(rocjitsu_afl_branchy_persistent_host
        PRIVATE __HIP_PLATFORM_AMD__)
    target_include_directories(rocjitsu_afl_branchy_persistent_host PRIVATE
        "${ROCFUZZ_ROCM_PATH}/include")
    target_link_libraries(rocjitsu_afl_branchy_persistent_host PRIVATE
        ${ROCFUZZ_HIP_RUNTIME}
        ${CMAKE_DL_LIBS})
    add_dependencies(rocjitsu_afl_branchy_persistent_host
        rocjitsu_afl_branchy_kernels_hsaco)

    if(ROCFUZZ_HSA_RUNTIME)
        add_executable(rocjitsu_afl_loader_table_host smoke/loader_table_host.cpp)
        target_include_directories(rocjitsu_afl_loader_table_host PRIVATE
            "${ROCFUZZ_ROCM_PATH}/include")
        target_link_libraries(rocjitsu_afl_loader_table_host PRIVATE
            ${ROCFUZZ_HSA_RUNTIME}
            ${CMAKE_DL_LIBS})

        add_executable(rocjitsu_afl_hsa_reader_probe smoke/hsa_reader_probe.cpp)
        target_include_directories(rocjitsu_afl_hsa_reader_probe PRIVATE
            "${ROCFUZZ_ROCM_PATH}/include")
        target_link_libraries(rocjitsu_afl_hsa_reader_probe PRIVATE
            ${ROCFUZZ_HSA_RUNTIME})

        add_custom_target(rocjitsu_afl_loader_table_smoke
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_VERBOSE=1
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_loader_table_host>
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_loader_table_host
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Checking HSA loader extension-table interposition"
            USES_TERMINAL
            VERBATIM)

        if(ROCFUZZ_PYTHON3_EXECUTABLE)
            set(ROCFUZZ_LOADER_TABLE_READER_REPORT
                ${CMAKE_CURRENT_BINARY_DIR}/loader_table_reader_report.jsonl)
            add_custom_target(rocjitsu_afl_loader_table_reader_report_smoke
                COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_LOADER_TABLE_READER_REPORT}
                COMMAND ${CMAKE_COMMAND} -E env
                        ROCJITSU_AFL_VERBOSE=1
                        ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_LOADER_TABLE_READER_REPORT}
                        LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                        $<TARGET_FILE:rocjitsu_afl_loader_table_host>
                        ${ROCFUZZ_SMOKE_HSACO}
                COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                        ${ROCFUZZ_LOADER_TABLE_READER_REPORT}
                        --require-event code_object_image
                        --require-event patch_device_elf
                        --require-event hsa_reader_patch
                        --require-field code_object_image context "HSA file-offset reader"
                        --require-field patch_device_elf success true
                        --require-field hsa_reader_patch context "HSA file-offset reader"
                        --require-field hsa_reader_patch final_status 0
                        --require-nonzero-hex hsa_reader_patch reader
                DEPENDS
                        rocjitsu_afl_preload
                        rocjitsu_afl_loader_table_host
                        rocjitsu_afl_branchy_kernels_hsaco
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Checking HSA loader table file-offset reader patch reports"
                USES_TERMINAL
                VERBATIM)
        endif()
    else()
        message(STATUS "rocfuzz HSA loader table smoke disabled (hsa-runtime64 not found)")
    endif()

    add_custom_target(rocjitsu_afl_dbi_smoke
        COMMAND ${CMAKE_COMMAND} -E env
                ROCJITSU_AFL_VERBOSE=1
                ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                ROCFUZZ_BRANCHY_EXPECT_RESULT=${ROCFUZZ_BRANCHY_EXPECTED_RESULT}
                LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                $<TARGET_FILE:rocjitsu_afl_branchy_host>
                ${ROCFUZZ_SMOKE_HSACO}
                ${ROCFUZZ_SMOKE_SEED}
        DEPENDS
                rocjitsu_afl_preload
                rocjitsu_afl_branchy_host
                rocjitsu_afl_branchy_kernels_hsaco
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running rocfuzz AFL DBI preload smoke target"
        USES_TERMINAL
        VERBATIM)

    if(ROCFUZZ_PYTHON3_EXECUTABLE)
        set(ROCFUZZ_RUNTIME_OVERHEAD_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/runtime_overhead_branchy.json)
        set(ROCFUZZ_RUNTIME_OVERHEAD_PATCH_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/runtime_overhead_branchy_patch.jsonl)
        add_custom_target(rocjitsu_afl_runtime_overhead_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f
                    ${ROCFUZZ_RUNTIME_OVERHEAD_REPORT}
                    ${ROCFUZZ_RUNTIME_OVERHEAD_PATCH_REPORT}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark_runtime_overhead.py
                    --output ${ROCFUZZ_RUNTIME_OVERHEAD_REPORT}
                    --label branchy-${ROCFUZZ_GPU_ARCH}
                    --warmups 1
                    --iterations 3
                    --timeout 30
                    --env ROCFUZZ_BRANCHY_EXPECT_RESULT=${ROCFUZZ_BRANCHY_EXPECTED_RESULT}
                    --env ROCFUZZ_REPEAT_BRANCHY_KERNEL_B=1
                    --instrumented-env LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    --instrumented-env ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    --instrumented-env ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_RUNTIME_OVERHEAD_PATCH_REPORT}
                    --
                    $<TARGET_FILE:rocjitsu_afl_branchy_host>
                    ${ROCFUZZ_SMOKE_HSACO}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_RUNTIME_OVERHEAD_PATCH_REPORT}
                    --require-event patch_device_elf
                    --require-event device_edge_delta
                    --require-field patch_device_elf success true
                    --require-min-field patch_device_elf previous_bb_branch_afl_map_budget 32768
                    --require-min-field patch_device_elf previous_bb_branch_trampoline_avg_bytes_x100 1
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_host
                    rocjitsu_afl_branchy_kernels_hsaco
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Measuring rocfuzz preload runtime overhead on branchy kernels"
            USES_TERMINAL
            VERBATIM)
    endif()

    set(ROCFUZZ_METADATA_RESIZE_HSA_READER_COMMAND)
    if(ROCFUZZ_HSA_RUNTIME)
        set(ROCFUZZ_METADATA_RESIZE_HSA_READER_COMMAND
            COMMAND $<TARGET_FILE:rocjitsu_afl_metadata_resize_loader_host>
                    ${ROCFUZZ_SMOKE_CCOB}
                    ${ROCFUZZ_GPU_ARCH}
                    --ccob-hsa-reader)
    endif()
    add_custom_target(rocjitsu_afl_metadata_resize_loader_smoke
        COMMAND $<TARGET_FILE:rocjitsu_afl_metadata_resize_loader_host>
                ${ROCFUZZ_SMOKE_HSACO}
                ${ROCFUZZ_GPU_ARCH}
                --raw-hip
        COMMAND $<TARGET_FILE:rocjitsu_afl_metadata_resize_loader_host>
                ${ROCFUZZ_SMOKE_CCOB}
                ${ROCFUZZ_GPU_ARCH}
                --ccob-hip
        ${ROCFUZZ_METADATA_RESIZE_HSA_READER_COMMAND}
        DEPENDS
                rocjitsu_afl_metadata_resize_loader_host
                rocjitsu_afl_branchy_kernels_hsaco
                rocjitsu_afl_branchy_kernels_ccob
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Checking HIP/HSA loading of rebuilt AMDGPU metadata notes"
        USES_TERMINAL
        VERBATIM)

    if(ROCFUZZ_PYTHON3_EXECUTABLE)
        set(ROCFUZZ_HYBRID_DEFAULT_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/hybrid_default_report.jsonl)
        add_custom_target(rocjitsu_afl_hybrid_default_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_HYBRID_DEFAULT_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_HYBRID_DEFAULT_REPORT}
                    ROCFUZZ_BRANCHY_EXPECT_RESULT=${ROCFUZZ_BRANCHY_EXPECTED_RESULT}
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_branchy_host>
                    ${ROCFUZZ_SMOKE_HSACO}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_HYBRID_DEFAULT_REPORT}
                    --require-event patch_device_elf
                    --require-event device_edge_delta
                    --require-field patch_device_elf success true
                    --require-field patch_device_elf coverage_strategy self-contained-hybrid-previous-bb-and-fixed-branch
                    --require-field patch_device_elf coverage_signal hybrid-previous-bb-and-fixed
                    --require-field patch_device_elf hybrid_edge_probes false
                    --require-field patch_device_elf branch_edge_slot_policy previous-bb-hash
                    --require-field patch_device_elf fixed_edge_slot_scope stable-site-hash
                    --require-field patch_device_elf fixed_slot_collisions 0
                    --require-field patch_device_elf branch_edge_site_limit 100
                    --require-field patch_device_elf branch_edge_site_limit_auto true
                    --require-field patch_device_elf previous_bb_branch_site_limit 4294967295
                    --require-field patch_device_elf previous_bb_branch_site_limit_auto true
                    --require-field patch_device_elf entry_backed_edge_kernels 0
                    --require-min-field patch_device_elf self_contained_edge_kernels 2
                    --require-field patch_device_elf entries_patched 0
                    --require-min-field patch_device_elf edge_sites_patched 1
                    --require-field patch_device_elf entry_liveness_register_kernels 0
                    --require-min-field patch_device_elf liveness_register_kernels 1
                    --require-field patch_device_elf vgpr_scratch_spills_requested true
                    --require-field patch_device_elf vgpr_scratch_spills_enabled true
                    --require-field patch_device_elf vgpr_scratch_spill_reason loader-context-scratch-spills-supported
                    --require-min-field patch_device_elf scratch_spill_probe_points 1
                    --require-min-field patch_device_elf vgpr_scratch_spill_probe_points 1
                    --require-field patch_device_elf sgpr_scratch_spill_probe_points 15
                    --require-min-field patch_device_elf probe_required_private_segment_bytes 1
                    --require-min-field patch_device_elf spill_bytes 1
                    --require-list-field patch_device_elf descriptor_resources private_segment_metadata_patch in-place-bytes
                    --require-min-field patch_device_elf hashed_edge_sites 3
                    --require-field patch_device_elf fixed_counter_branch_edge_safety_fallback_used 8
                    --require-min-field patch_device_elf edge_trampolines_planned 1
                    --require-min-field patch_device_elf planned_edge_trampoline_bytes 1
                    --require-min-field patch_device_elf planned_appended_edge_trampoline_bytes 1
                    --require-min-field patch_device_elf largest_edge_trampoline_bytes 1
                    --require-list-field patch_device_elf kernel_summaries self_contained_probe true
                    --require-list-field patch_device_elf kernel_summaries coverage_strategy self-contained-hybrid-previous-bb-and-fixed-branch
                    --require-list-field patch_device_elf kernel_summaries coverage_signal hybrid-previous-bb-and-fixed
                    --require-list-field patch_device_elf kernel_summaries previous_bb_branch_aggregate_limit_kind none
                    --require-list-field patch_device_elf kernel_summaries previous_bb_branch_aggregate_safety full-previous-bb-within-auto-budget
                    --require-list-item patch_device_elf kernel_summaries
                        kernel=branchy_kernel_a
                        coverage_strategy=self-contained-fixed-branch
                        coverage_signal=fixed-branch-counters
                        previous_bb_branch_aggregate_limit_kind=none
                        previous_bb_branch_aggregate_safety=full-previous-bb-within-auto-budget
                        fixed_counter_branch_edge_fallback_budget=4
                        fixed_counter_branch_edge_fallback_budget_reason=auto-candidate-branch-edge-count
                        fixed_counter_branch_edge_fallback_used=4
                        fixed_counter_branch_edge_aggregate_fallback_used=0
                        fixed_counter_branch_edge_safety_fallback_used=4
                        fixed_counter_branch_edge_liveness_fallback_used=0
                        fixed_counter_branch_edge_placement_fallback_used=0
                        branch_edges_degraded_to_fixed=4
                        skipped_branch_liveness=0
                    --require-list-item patch_device_elf kernel_summaries
                        kernel=branchy_kernel_b
                        coverage_strategy=self-contained-hybrid-previous-bb-and-fixed-branch
                        coverage_signal=hybrid-previous-bb-and-fixed
                        previous_bb_branch_aggregate_limit_kind=none
                        previous_bb_branch_aggregate_safety=full-previous-bb-within-auto-budget
                        fixed_counter_branch_edge_fallback_budget=29
                        fixed_counter_branch_edge_fallback_budget_reason=auto-candidate-branch-edge-count
                        fixed_counter_branch_edge_fallback_used=4
                        fixed_counter_branch_edge_aggregate_fallback_used=0
                        fixed_counter_branch_edge_safety_fallback_used=4
                        fixed_counter_branch_edge_liveness_fallback_used=0
                        fixed_counter_branch_edge_placement_fallback_used=0
                        sgpr_scratch_spill_probe_points=15
                        sgpr_scratch_spill_disabled_by_opaque_probe_points=0
                        sgpr_scratch_spill_disabled_by_exec_condition_probe_points=0
                        direct_exec_fixed_scratch_disabled_by_opaque_probe_points=0
                        branch_edges_degraded_to_fixed=4
                        skipped_branch_liveness=0
                    --require-list-item patch_device_elf kernel_summaries.degradation_reason_counts
                        kind=branch
                        "reason=EXEC-conditioned previous-BB branch probe is not yet proven safe"
                        count=4
                    --require-list-item patch_device_elf sampled_selected_edges
                        kernel=branchy_kernel_b
                        kind=branch
                        slot_policy=previous-bb-hash
                        scratch_spill=true
                        sgpr_scratch_spill=true
                        scratch_address_exec_source=exec
                        placement=appended-cave
                    --require-list-item patch_device_elf sampled_selected_edges
                        kernel=branchy_kernel_b
                        kind=cond-branch
                        slot_policy=previous-bb-hash
                        scratch_spill=true
                        sgpr_scratch_spill=true
                        scratch_address_exec_source=exec
                        placement=appended-cave
                    --require-list-item patch_device_elf sampled_selected_edges
                        kernel=branchy_kernel_b
                        kind=cond-branch
                        slot_policy=fixed-counter
                        scratch_spill=false
                        scratch_address_exec_source=none
                        placement=appended-cave
                    --require-min-field patch_device_elf branch_edges_degraded_to_fixed 1
                    --require-min-field device_edge_delta edge_slot_delta_count 1
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_host
                    rocjitsu_afl_branchy_kernels_hsaco
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Checking default adaptive self-contained branch coverage planning"
            USES_TERMINAL
            VERBATIM)

        set(ROCFUZZ_OPAQUE_SCRATCH_GUARD_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/opaque_scratch_guard_report.jsonl)
        add_custom_target(rocjitsu_afl_opaque_scratch_guard_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_OPAQUE_SCRATCH_GUARD_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_KERNEL_INCLUDE=branchy_kernel_b
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_DEBUG_DISABLE_VGPR_SCRATCH_SPILLS=1
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_OPAQUE_SCRATCH_GUARD_REPORT}
                    ROCFUZZ_BRANCHY_EXPECT_RESULT=${ROCFUZZ_BRANCHY_EXPECTED_RESULT}
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_branchy_host>
                    ${ROCFUZZ_SMOKE_HSACO}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_OPAQUE_SCRATCH_GUARD_REPORT}
                    --require-event patch_device_elf
                    --require-event device_edge_delta
                    --require-field patch_device_elf success true
                    --require-field patch_device_elf kernel_filter branchy_kernel_b
                    --require-field patch_device_elf coverage_strategy self-contained-fixed-branch
                    --require-field patch_device_elf coverage_signal fixed-branch-counters
                    --require-field patch_device_elf hybrid_edge_probes false
                    --require-field patch_device_elf skip_entry_probe true
                    --require-field patch_device_elf entry_backed_edge_kernels 0
                    --require-field patch_device_elf self_contained_edge_kernels 1
                    --require-field patch_device_elf edge_sites_patched 4
                    --require-field patch_device_elf fixed_edge_sites 4
                    --require-field patch_device_elf hashed_edge_sites 0
                    --require-field patch_device_elf sgpr_scratch_spill_probe_points 0
                    --require-field patch_device_elf scratch_spill_probe_points 0
                    --require-field patch_device_elf vgpr_scratch_spills_enabled false
                    --require-field patch_device_elf vgpr_scratch_spill_reason disabled-by-debug-env
                    --require-list-item patch_device_elf kernel_summaries
                        kernel=branchy_kernel_b
                        coverage_strategy=self-contained-fixed-branch
                        branch_edges_degraded_to_fixed=7
                        sgpr_scratch_spill_probe_points=0
                        skipped_branch_liveness=13
                    --require-list-item patch_device_elf kernel_summaries
                        kernel=branchy_kernel_b
                        fixed_counter_branch_edge_fallback_used=4
                    --require-list-item patch_device_elf kernel_summaries.degradation_reason_counts
                        kind=branch
                        "reason=EXEC-conditioned previous-BB branch probe is not yet proven safe"
                        count=4
                    --require-list-item patch_device_elf kernel_summaries
                        kernel=branchy_kernel_b
                        fixed_counter_branch_edge_liveness_fallback_used=3
                    --require-list-item patch_device_elf sampled_selected_edges
                        kernel=branchy_kernel_b
                        kind=cond-branch
                        slot_policy=fixed-counter
                        scratch_spill=false
                        scratch_address_exec_source=none
                        placement=appended-cave
                    --require-field patch_device_elf branch_edges_degraded_to_fixed 7
                    --require-min-field device_edge_delta edge_slot_delta_count 1
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_host
                    rocjitsu_afl_branchy_kernels_hsaco
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Checking adaptive branch sites recover fixed-counter fallback"
            USES_TERMINAL
            VERBATIM)

        add_custom_target(rocjitsu_afl_require_device_edges_gate_smoke
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/expect_command_failure.py
                    --expected-code 86
                    --env ROCJITSU_AFL_VERBOSE=1
                    --env ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    --env ROCJITSU_AFL_DEBUG_DISABLE_EDGES=1
                    --env ROCFUZZ_BRANCHY_IGNORE_SYNC_AND_EXIT=1
                    --env LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    --
                    $<TARGET_FILE:rocjitsu_afl_branchy_host>
                    ${ROCFUZZ_SMOKE_HSACO}
                    ${ROCFUZZ_SMOKE_SEED}
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_host
                    rocjitsu_afl_branchy_kernels_hsaco
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Checking strict device-edge gate fails when a harness ignores sync errors"
            USES_TERMINAL
            VERBATIM)
    endif()

    add_custom_target(rocjitsu_afl_ccob_smoke
        COMMAND ${CMAKE_COMMAND} -E env
                ROCJITSU_AFL_VERBOSE=1
                ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                ROCFUZZ_BRANCHY_EXPECT_RESULT=${ROCFUZZ_BRANCHY_EXPECTED_RESULT}
                LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                $<TARGET_FILE:rocjitsu_afl_branchy_host>
                ${ROCFUZZ_SMOKE_CCOB}
                ${ROCFUZZ_SMOKE_SEED}
        DEPENDS
                rocjitsu_afl_preload
                rocjitsu_afl_branchy_host
                rocjitsu_afl_branchy_kernels_ccob
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running rocfuzz AFL DBI CCOB preload smoke target"
        USES_TERMINAL
        VERBATIM)

    if(ROCFUZZ_PYTHON3_EXECUTABLE)
        set(ROCFUZZ_CCOB_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/ccob_report.jsonl)
        add_custom_target(rocjitsu_afl_ccob_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_CCOB_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_VERBOSE=1
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_CCOB_REPORT}
                    ROCFUZZ_REPEAT_BRANCHY_KERNEL_B=1
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_branchy_host>
                    ${ROCFUZZ_SMOKE_CCOB}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_CCOB_REPORT}
                    --require-event lazy_ccob_module
                    --require-event lazy_ccob_function
                    --require-event lazy_ccob_launch
                    --require-event lazy_ccob_release
                    --require-event device_edge_delta
                    --require-event patch_device_elf
                    --require-event code_object_image
                    --require-count lazy_ccob_module 1
                    --require-count lazy_ccob_function 2
                    --require-count lazy_ccob_launch 3
                    --require-count lazy_ccob_release 1
                    --require-count device_edge_delta 1
                    --require-count patch_device_elf 2
                    --require-field lazy_ccob_module top_level_ccob true
                    --require-field lazy_ccob_module lazy_shadow_policy raw_exact_kernel_elf
                    --require-field lazy_ccob_module lazy_shadow_preserves_sibling_payloads false
                    --require-field code_object_image top_level_ccob true
                    --require-field lazy_ccob_function inserted true
                    --require-field lazy_ccob_function shadow_image_policy raw_exact_kernel_elf
                    --require-field lazy_ccob_launch kernel branchy_kernel_b
                    --require-field lazy_ccob_release shadow_modules 2
                    --require-field lazy_ccob_release unload_attempts 2
                    --require-field lazy_ccob_release unload_failures 0
                    --require-field lazy_ccob_release last_unload_error 0
                    --require-field device_edge_delta lazy_ccob_launches 3
                    --require-min-field device_edge_delta edge_slot_delta_count 1
                    --require-field patch_device_elf success true
                    --require-nonzero-hex lazy_ccob_launch patched_module
                    --require-nonzero-hex lazy_ccob_launch function
                    --require-nonempty-list patch_device_elf descriptor_resources
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_host
                    rocjitsu_afl_branchy_kernels_ccob
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running rocfuzz AFL DBI CCOB launch report smoke target"
            USES_TERMINAL
            VERBATIM)

        set(ROCFUZZ_CCOB_STREAM_SYNC_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/ccob_stream_sync_report.jsonl)
        add_custom_target(rocjitsu_afl_ccob_stream_sync_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_CCOB_STREAM_SYNC_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_VERBOSE=1
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_CCOB_STREAM_SYNC_REPORT}
                    ROCFUZZ_REPEAT_BRANCHY_KERNEL_B=1
                    ROCFUZZ_BRANCHY_SYNC=stream
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_branchy_host>
                    ${ROCFUZZ_SMOKE_CCOB}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_CCOB_STREAM_SYNC_REPORT}
                    --require-event lazy_ccob_launch
                    --require-event device_edge_delta
                    --require-count lazy_ccob_launch 3
                    --require-count device_edge_delta 1
                    --require-field device_edge_delta trigger hipStreamSynchronize
                    --require-field device_edge_delta lazy_ccob_launches 3
                    --require-min-field device_edge_delta edge_slot_delta_count 1
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_host
                    rocjitsu_afl_branchy_kernels_ccob
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running rocfuzz AFL DBI CCOB stream-sync report smoke target"
            USES_TERMINAL
            VERBATIM)

        set(ROCFUZZ_CCOB_MEMCPY_SYNC_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/ccob_memcpy_sync_report.jsonl)
        add_custom_target(rocjitsu_afl_ccob_memcpy_sync_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_CCOB_MEMCPY_SYNC_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_VERBOSE=1
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_CCOB_MEMCPY_SYNC_REPORT}
                    ROCFUZZ_REPEAT_BRANCHY_KERNEL_B=1
                    ROCFUZZ_BRANCHY_SYNC=none
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_branchy_host>
                    ${ROCFUZZ_SMOKE_CCOB}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_CCOB_MEMCPY_SYNC_REPORT}
                    --require-event lazy_ccob_launch
                    --require-event device_edge_delta
                    --require-count lazy_ccob_launch 3
                    --require-count device_edge_delta 1
                    --require-field device_edge_delta trigger hipMemcpy
                    --require-field device_edge_delta lazy_ccob_launches 3
                    --require-min-field device_edge_delta edge_slot_delta_count 1
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_host
                    rocjitsu_afl_branchy_kernels_ccob
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running rocfuzz AFL DBI CCOB memcpy-sync report smoke target"
            USES_TERMINAL
            VERBATIM)

        set(ROCFUZZ_CCOB_DESTRUCTOR_RELEASE_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/ccob_destructor_release_report.jsonl)
        add_custom_target(rocjitsu_afl_ccob_destructor_release_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_CCOB_DESTRUCTOR_RELEASE_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_VERBOSE=1
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_CCOB_DESTRUCTOR_RELEASE_REPORT}
                    ROCFUZZ_REPEAT_BRANCHY_KERNEL_B=1
                    ROCFUZZ_SKIP_MODULE_UNLOAD=1
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_branchy_host>
                    ${ROCFUZZ_SMOKE_CCOB}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_CCOB_DESTRUCTOR_RELEASE_REPORT}
                    --require-event lazy_ccob_launch
                    --require-event lazy_ccob_release
                    --require-event device_edge_delta
                    --require-count lazy_ccob_launch 3
                    --require-count lazy_ccob_release 1
                    --require-count device_edge_delta 1
                    --require-field lazy_ccob_release shadow_modules 2
                    --require-field lazy_ccob_release unload_attempts 2
                    --require-field lazy_ccob_release unload_failures 0
                    --require-field device_edge_delta lazy_ccob_launches 3
                    --require-min-field device_edge_delta edge_slot_delta_count 1
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_host
                    rocjitsu_afl_branchy_kernels_ccob
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running rocfuzz AFL DBI CCOB destructor-release report smoke target"
            USES_TERMINAL
            VERBATIM)
        else()
            message(STATUS "rocfuzz CCOB launch report smoke disabled (python3 not found)")
        endif()

        set(ROCFUZZ_MULTI_CCOB_LAUNCH_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/multi_ccob_launch_report.jsonl)
        add_custom_target(rocjitsu_afl_multi_ccob_launch_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_MULTI_CCOB_LAUNCH_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_VERBOSE=1
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_MULTI_CCOB_LAUNCH_REPORT}
                    ROCFUZZ_REPEAT_BRANCHY_KERNEL_B=1
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_branchy_host>
                    ${ROCFUZZ_SMOKE_MULTI_CCOB}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_MULTI_CCOB_LAUNCH_REPORT}
                    --require-event lazy_ccob_module
                    --require-event code_object_image
                    --require-event patch_device_elf
                    --require-event lazy_ccob_function
                    --require-event lazy_ccob_launch
                    --require-event lazy_ccob_release
                    --require-event device_edge_delta
                    --require-count lazy_ccob_function 2
                    --require-count lazy_ccob_launch 3
                    --require-count device_edge_delta 1
                    --require-field lazy_ccob_module top_level_ccob true
                    --require-field lazy_ccob_module device_images 2
                    --require-field lazy_ccob_module raw_elf_bypass_drops_sibling_payloads true
                    --require-field lazy_ccob_module lazy_shadow_policy raw_exact_kernel_elf
                    --require-field lazy_ccob_module lazy_shadow_preserves_sibling_payloads false
                    --require-field code_object_image device_images 2
                    --require-field code_object_image raw_elf_bypass_drops_sibling_payloads true
                    --require-field patch_device_elf success true
                    --require-field lazy_ccob_function inserted true
                    --require-field lazy_ccob_function shadow_image_policy raw_exact_kernel_elf
                    --require-field lazy_ccob_release shadow_modules 2
                    --require-field lazy_ccob_release unload_failures 0
                    --require-field device_edge_delta lazy_ccob_launches 3
                    --require-min-field device_edge_delta edge_slot_delta_count 1
                    --require-nonzero-hex lazy_ccob_launch patched_module
                    --require-nonzero-hex lazy_ccob_launch function
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_host
                    rocjitsu_afl_branchy_kernels_multi_ccob
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running rocfuzz AFL DBI multi-payload CCOB launch report smoke target"
                USES_TERMINAL
                VERBATIM)

        if(ROCFUZZ_HSA_RUNTIME)
            set(ROCFUZZ_HSA_MULTI_CCOB_READER_REPORT
                ${CMAKE_CURRENT_BINARY_DIR}/hsa_multi_ccob_reader_report.jsonl)
            add_custom_target(rocjitsu_afl_hsa_multi_ccob_reader_report_smoke
                COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_HSA_MULTI_CCOB_READER_REPORT}
                COMMAND ${CMAKE_COMMAND} -E env
                        ROCJITSU_AFL_VERBOSE=1
                        ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_HSA_MULTI_CCOB_READER_REPORT}
                        LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                        $<TARGET_FILE:rocjitsu_afl_hsa_reader_probe>
                        ${ROCFUZZ_SMOKE_MULTI_CCOB}
                COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                        ${ROCFUZZ_HSA_MULTI_CCOB_READER_REPORT}
                        --require-event code_object_image
                        --require-event patch_device_elf
                        --require-event ccob_rebuild
                        --require-event hsa_reader_patch
                        --require-field code_object_image context "HSA memory reader"
                        --require-field code_object_image top_level_ccob true
                        --require-field code_object_image device_images 2
                        --require-field code_object_image raw_elf_bypass_drops_sibling_payloads true
                        --require-field patch_device_elf success true
                        --require-field patch_device_elf device_image_index 0
                        --require-field patch_device_elf edge_instrumentation_reason previous-bb-edge-probes-supported
                        --require-field ccob_rebuild success true
                        --require-field ccob_rebuild device_image_index 0
                        --require-field ccob_rebuild input_device_images 2
                        --require-field ccob_rebuild output_device_images 2
                        --require-field ccob_rebuild sibling_payloads_preserved true
                        --require-field hsa_reader_patch primary_rebuilt_ccob true
                        --require-field hsa_reader_patch device_image_index 0
                        --require-field hsa_reader_patch fallback_available true
                        --require-field hsa_reader_patch final_status 0
                        --require-nonzero-hex hsa_reader_patch reader
                DEPENDS
                        rocjitsu_afl_preload
                        rocjitsu_afl_hsa_reader_probe
                        rocjitsu_afl_branchy_kernels_multi_ccob
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Running rocfuzz AFL DBI HSA-reader multi-payload CCOB rebuild report smoke target"
                USES_TERMINAL
                VERBATIM)

            set(ROCFUZZ_HSA_RDC_MULTI_CCOB_READER_REPORT
                ${CMAKE_CURRENT_BINARY_DIR}/hsa_rdc_multi_ccob_reader_report.jsonl)
            add_custom_target(rocjitsu_afl_hsa_rdc_multi_ccob_reader_report_smoke
                COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_HSA_RDC_MULTI_CCOB_READER_REPORT}
                COMMAND ${CMAKE_COMMAND} -E env
                        ROCJITSU_AFL_VERBOSE=1
                        ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_HSA_RDC_MULTI_CCOB_READER_REPORT}
                        LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                        $<TARGET_FILE:rocjitsu_afl_hsa_reader_probe>
                        ${ROCFUZZ_SMOKE_RDC_MULTI_CCOB}
                COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                        ${ROCFUZZ_HSA_RDC_MULTI_CCOB_READER_REPORT}
                        --require-event code_object_image
                        --require-event patch_device_elf
                        --require-event ccob_rebuild
                        --require-event hsa_reader_patch
                        --require-field code_object_image context "HSA memory reader"
                        --require-field code_object_image top_level_ccob true
                        --require-field code_object_image device_images 2
                        --require-field code_object_image top_level_ccob_device_images 2
                        --require-field code_object_image raw_elf_bypass_drops_sibling_payloads true
                        --require-field patch_device_elf success true
                        --require-field patch_device_elf device_image ccob:hipv4-amdgcn-amd-amdhsa--${ROCFUZZ_GPU_ARCH}
                        --require-field patch_device_elf edge_instrumentation_reason previous-bb-edge-probes-supported
                        --require-field ccob_rebuild success true
                        --require-field ccob_rebuild device_image ccob:hipv4-amdgcn-amd-amdhsa--${ROCFUZZ_GPU_ARCH}
                        --require-field ccob_rebuild input_device_images 2
                        --require-field ccob_rebuild output_device_images 2
                        --require-field ccob_rebuild sibling_payloads_preserved true
                        --require-field hsa_reader_patch primary_rebuilt_ccob true
                        --require-field hsa_reader_patch device_image ccob:hipv4-amdgcn-amd-amdhsa--${ROCFUZZ_GPU_ARCH}
                        --require-field hsa_reader_patch fallback_available true
                        --require-field hsa_reader_patch final_status 0
                        --require-nonzero-hex hsa_reader_patch reader
                DEPENDS
                        rocjitsu_afl_preload
                        rocjitsu_afl_hsa_reader_probe
                        rocjitsu_afl_branchy_kernels_rdc_multi_ccob
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Running rocfuzz AFL DBI HSA-reader RDC multi-payload CCOB rebuild report smoke target"
                USES_TERMINAL
                VERBATIM)

            add_custom_target(rocjitsu_afl_multi_ccob_rebuild_report_smoke
                DEPENDS
                    rocjitsu_afl_hsa_multi_ccob_reader_report_smoke
                    rocjitsu_afl_hsa_rdc_multi_ccob_reader_report_smoke)
        endif()

        if(EXISTS "${ROCFUZZ_LAZY_CCOB_MODULE}" AND ROCFUZZ_PYTHON3_EXECUTABLE)
            set(ROCFUZZ_LAZY_CCOB_REPORT
                ${CMAKE_CURRENT_BINARY_DIR}/lazy_ccob_report.jsonl)
        add_custom_target(rocjitsu_afl_lazy_ccob_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_LAZY_CCOB_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    HIP_VISIBLE_DEVICES=${ROCFUZZ_LAZY_CCOB_VISIBLE_DEVICES}
                    ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE=1
                    ROCJITSU_AFL_DEBUG_EDGE_LIMIT=0
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOTS=1
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_LAZY_CCOB_REPORT}
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_ccob_module_probe>
                    ${ROCFUZZ_LAZY_CCOB_MODULE}
                    --repeat-get
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_LAZY_CCOB_REPORT}
                    --require-event lazy_ccob_module
                    --require-event lazy_ccob_function
                    --require-event lazy_ccob_release
                    --require-event patch_device_elf
                    --require-field lazy_ccob_module top_level_ccob true
                    --require-field lazy_ccob_module lazy_shadow_policy raw_exact_kernel_elf
                    --require-field lazy_ccob_module lazy_shadow_preserves_sibling_payloads false
                    --require-field lazy_ccob_function inserted true
                    --require-field lazy_ccob_function cache_hit true
                    --require-field lazy_ccob_function shadow_image_policy raw_exact_kernel_elf
                    --require-field lazy_ccob_release shadow_modules 1
                    --require-field lazy_ccob_release unload_attempts 1
                    --require-field lazy_ccob_release unload_failures 0
                    --require-field lazy_ccob_release last_unload_error 0
                    --require-field patch_device_elf success true
                    --require-nonempty-list patch_device_elf descriptor_resources
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_ccob_module_probe
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running rocfuzz AFL DBI lazy CCOB lifecycle report smoke target"
            USES_TERMINAL
            VERBATIM)

        set(ROCFUZZ_LAZY_CCOB_CONCURRENT_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/lazy_ccob_concurrent_report.jsonl)
        add_custom_target(rocjitsu_afl_lazy_ccob_concurrent_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_LAZY_CCOB_CONCURRENT_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    HIP_VISIBLE_DEVICES=${ROCFUZZ_LAZY_CCOB_VISIBLE_DEVICES}
                    ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE=1
                    ROCJITSU_AFL_DEBUG_EDGE_LIMIT=0
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOTS=1
                    ROCFUZZ_LAZY_CCOB_RACE_DELAY_US=200000
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_LAZY_CCOB_CONCURRENT_REPORT}
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_ccob_module_probe>
                    ${ROCFUZZ_LAZY_CCOB_MODULE}
                    --concurrent-get
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_LAZY_CCOB_CONCURRENT_REPORT}
                    --require-event lazy_ccob_module
                    --require-event lazy_ccob_function
                    --require-event lazy_ccob_release
                    --require-event patch_device_elf
                    --require-count lazy_ccob_function 2
                    --require-field lazy_ccob_module top_level_ccob true
                    --require-field lazy_ccob_function inserted true
                    --require-field lazy_ccob_function inserted false
                    --require-field lazy_ccob_function duplicate_unloaded true
                    --require-field lazy_ccob_function shadow_image_policy raw_exact_kernel_elf
                    --require-field lazy_ccob_release shadow_modules 1
                    --require-field lazy_ccob_release unload_attempts 1
                    --require-field lazy_ccob_release unload_failures 0
                    --require-field lazy_ccob_release last_unload_error 0
                    --require-field patch_device_elf success true
                    --require-nonempty-list patch_device_elf descriptor_resources
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_ccob_module_probe
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running rocfuzz AFL DBI lazy CCOB concurrent lookup report smoke target"
            USES_TERMINAL
            VERBATIM)
    else()
        if(NOT EXISTS "${ROCFUZZ_LAZY_CCOB_MODULE}")
            message(STATUS
                "rocfuzz lazy CCOB lifecycle smoke disabled "
                "(ROCFUZZ_LAZY_CCOB_MODULE not found: ${ROCFUZZ_LAZY_CCOB_MODULE})")
        elseif(NOT ROCFUZZ_PYTHON3_EXECUTABLE)
            message(STATUS "rocfuzz lazy CCOB lifecycle smoke disabled (python3 not found)")
        endif()
    endif()

    add_custom_target(rocjitsu_afl_fatbin_smoke
        COMMAND ${CMAKE_COMMAND} -E env
                ROCJITSU_AFL_VERBOSE=1
                ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                ${ROCFUZZ_FATBIN_HOST}
                ${ROCFUZZ_SMOKE_SEED}
        DEPENDS
                rocjitsu_afl_preload
                rocjitsu_afl_branchy_fatbin_host
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running rocfuzz AFL DBI compiler-registered HIP fatbin smoke target"
        USES_TERMINAL
        VERBATIM)

    if(ROCFUZZ_PYTHON3_EXECUTABLE)
        set(ROCFUZZ_KPACK_SHADOW_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/kpack_shadow_report.jsonl)
        add_custom_target(rocjitsu_afl_kpack_shadow_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_KPACK_SHADOW_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_VERBOSE=1
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_DEBUG_FORCE_RUNTIME_SHADOW_MODULES=1
                    ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE=1
                    ROCJITSU_AFL_DEBUG_EDGE_LIMIT=0
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOTS=1
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY=fixed
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_KPACK_SHADOW_REPORT}
                    ROCFUZZ_REPEAT_FATBIN_KERNEL_B=1
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    ${ROCFUZZ_FATBIN_HOST}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_KPACK_SHADOW_REPORT}
                    --require-event patch_device_elf
                    --require-event runtime_shadow_source
                    --require-event runtime_shadow_function
                    --require-event runtime_shadow_launch
                    --require-event runtime_shadow_registration_release
                    --require-event device_edge_delta
                    --require-event code_object_image
                    --require-event hsa_reader_patch
                    --require-count runtime_shadow_function 2
                    --require-count runtime_shadow_launch 2
                    --require-count device_edge_delta 1
                    --require-field code_object_image context "HSA memory reader"
                    --require-field code_object_image top_level_raw_elf true
                    --require-nonzero-hex code_object_image runtime_shadow_registration
                    --require-field hsa_reader_patch final_status 0
                    --require-nonzero-hex hsa_reader_patch reader
                    --require-field patch_device_elf success true
                    --require-field runtime_shadow_source inserted true
                    --require-min-field runtime_shadow_source kernel_count 2
                    --require-field runtime_shadow_function inserted true
                    --require-field runtime_shadow_function cache_hit true
                    --require-field runtime_shadow_function source_count 1
                    --require-field runtime_shadow_function attempted_sources 1
                    --require-field runtime_shadow_function matching_sources 1
                    --require-field runtime_shadow_function ambiguous_source_matches false
                    --require-min-field runtime_shadow_function patch_us 1
                    --require-min-field runtime_shadow_function load_us 1
                    --require-field runtime_shadow_launch kernel _Z15fatbin_kernel_bPKjPj
                    --require-field runtime_shadow_registration_release removed_sources 1
                    --require-field runtime_shadow_registration_release removed_functions 1
                    --require-field runtime_shadow_registration_release shadow_modules 1
                    --require-field runtime_shadow_registration_release unload_attempts 1
                    --require-field runtime_shadow_registration_release unload_failures 0
                    --require-field runtime_shadow_registration_release last_unload_error 0
                    --require-field device_edge_delta runtime_shadow_launches 2
                    --require-min-field device_edge_delta edge_slot_delta_count 1
                    --require-nonempty-list patch_device_elf descriptor_resources
                    --require-nonzero-hex runtime_shadow_function source_dedupe_key
                    --require-nonzero-hex runtime_shadow_launch runtime_function
                    --require-nonzero-hex runtime_shadow_launch module
                    --require-nonzero-hex runtime_shadow_launch function
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_fatbin_host
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running rocfuzz AFL DBI KPACK runtime-shadow report smoke target"
            USES_TERMINAL
            VERBATIM)

        set(ROCFUZZ_KPACK_CONCURRENT_SHADOW_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/kpack_concurrent_shadow_report.jsonl)
        add_custom_target(rocjitsu_afl_kpack_concurrent_shadow_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_KPACK_CONCURRENT_SHADOW_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_VERBOSE=1
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_DEBUG_FORCE_RUNTIME_SHADOW_MODULES=1
                    ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE=1
                    ROCJITSU_AFL_DEBUG_EDGE_LIMIT=0
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOTS=1
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY=fixed
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_KPACK_CONCURRENT_SHADOW_REPORT}
                    ROCFUZZ_CONCURRENT_FATBIN_KERNEL_B=1
                    ROCFUZZ_RUNTIME_SHADOW_RACE_DELAY_US=200000
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    ${ROCFUZZ_FATBIN_HOST}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_KPACK_CONCURRENT_SHADOW_REPORT}
                    --require-event runtime_shadow_source
                    --require-event runtime_shadow_function
                    --require-event runtime_shadow_launch
                    --require-event runtime_shadow_registration_release
                    --require-event device_edge_delta
                    --require-count runtime_shadow_launch 2
                    --require-count device_edge_delta 1
                    --require-min-field runtime_shadow_source kernel_count 2
                    --require-field runtime_shadow_function inserted true
                    --require-field runtime_shadow_function inserted false
                    --require-field runtime_shadow_function duplicate_unloaded true
                    --require-field runtime_shadow_function source_count 1
                    --require-field runtime_shadow_function attempted_sources 1
                    --require-field runtime_shadow_function matching_sources 1
                    --require-field runtime_shadow_function ambiguous_source_matches false
                    --require-min-field runtime_shadow_function patch_us 1
                    --require-min-field runtime_shadow_function load_us 1
                    --require-field runtime_shadow_launch kernel _Z15fatbin_kernel_bPKjPj
                    --require-field runtime_shadow_registration_release removed_sources 1
                    --require-field runtime_shadow_registration_release removed_functions 1
                    --require-field runtime_shadow_registration_release shadow_modules 1
                    --require-field runtime_shadow_registration_release unload_attempts 1
                    --require-field runtime_shadow_registration_release unload_failures 0
                    --require-field runtime_shadow_registration_release last_unload_error 0
                    --require-field device_edge_delta runtime_shadow_launches 2
                    --require-min-field device_edge_delta edge_slot_delta_count 1
                    --require-nonzero-hex runtime_shadow_function created_module
                    --require-nonzero-hex runtime_shadow_function source_dedupe_key
                    --require-nonzero-hex runtime_shadow_launch runtime_function
                    --require-nonzero-hex runtime_shadow_launch module
                    --require-nonzero-hex runtime_shadow_launch function
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_fatbin_host
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running rocfuzz AFL DBI KPACK concurrent runtime-shadow report smoke target"
            USES_TERMINAL
            VERBATIM)

        set(ROCFUZZ_KPACK_DESTRUCTOR_RELEASE_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/kpack_destructor_release_report.jsonl)
        add_custom_target(rocjitsu_afl_kpack_destructor_release_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_KPACK_DESTRUCTOR_RELEASE_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_VERBOSE=1
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_DEBUG_FORCE_RUNTIME_SHADOW_MODULES=1
                    ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE=1
                    ROCJITSU_AFL_DEBUG_EDGE_LIMIT=0
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOTS=1
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY=fixed
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_KPACK_DESTRUCTOR_RELEASE_REPORT}
                    ROCFUZZ_REPEAT_FATBIN_KERNEL_B=1
                    ROCFUZZ_SKIP_RUNTIME_UNREGISTER_RELEASE=1
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    ${ROCFUZZ_FATBIN_HOST}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_KPACK_DESTRUCTOR_RELEASE_REPORT}
                    --require-event runtime_shadow_source
                    --require-event runtime_shadow_function
                    --require-event runtime_shadow_launch
                    --require-event runtime_shadow_registration_release
                    --require-event device_edge_delta
                    --require-count runtime_shadow_launch 2
                    --require-count runtime_shadow_registration_release 1
                    --require-field runtime_shadow_registration_release context destructor
                    --require-field runtime_shadow_registration_release removed_sources 1
                    --require-field runtime_shadow_registration_release removed_functions 1
                    --require-field runtime_shadow_registration_release shadow_modules 1
                    --require-field runtime_shadow_registration_release unload_attempts 1
                    --require-field runtime_shadow_registration_release unload_failures 0
                    --require-field runtime_shadow_registration_release last_unload_error 0
                    --require-field device_edge_delta runtime_shadow_launches 2
                    --require-min-field device_edge_delta edge_slot_delta_count 1
                    --require-nonzero-hex runtime_shadow_registration_release registration
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_fatbin_host
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running rocfuzz AFL DBI KPACK destructor-release report smoke target"
            USES_TERMINAL
            VERBATIM)

        set(ROCFUZZ_KPACK_EVENT_SYNC_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/kpack_event_sync_report.jsonl)
        add_custom_target(rocjitsu_afl_kpack_event_sync_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_KPACK_EVENT_SYNC_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_VERBOSE=1
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_DEBUG_FORCE_RUNTIME_SHADOW_MODULES=1
                    ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE=1
                    ROCJITSU_AFL_DEBUG_EDGE_LIMIT=0
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOTS=1
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY=fixed
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_KPACK_EVENT_SYNC_REPORT}
                    ROCFUZZ_REPEAT_FATBIN_KERNEL_B=1
                    ROCFUZZ_FATBIN_SYNC=event
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    ${ROCFUZZ_FATBIN_HOST}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_KPACK_EVENT_SYNC_REPORT}
                    --require-event runtime_shadow_launch
                    --require-event device_edge_delta
                    --require-count runtime_shadow_launch 2
                    --require-count device_edge_delta 1
                    --require-field device_edge_delta trigger hipEventSynchronize
                    --require-field device_edge_delta runtime_shadow_launches 2
                    --require-min-field device_edge_delta edge_slot_delta_count 1
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_fatbin_host
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running rocfuzz AFL DBI KPACK event-sync report smoke target"
            USES_TERMINAL
            VERBATIM)

        set(ROCFUZZ_KPACK_SHADOW_MISS_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/kpack_shadow_miss_report.jsonl)
        add_custom_target(rocjitsu_afl_kpack_shadow_miss_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_KPACK_SHADOW_MISS_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_VERBOSE=1
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_DEBUG_FORCE_RUNTIME_SHADOW_MODULES=1
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_KPACK_SHADOW_MISS_REPORT}
                    ROCFUZZ_SKIP_FATBIN_KERNEL_A=1
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    ${ROCFUZZ_FATBIN_HOST}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_KPACK_SHADOW_MISS_REPORT}
                    --require-event runtime_shadow_miss
                    --require-event runtime_shadow_source
                    --require-event runtime_shadow_registration_release
                    --require-event patch_device_elf
                    --require-event code_object_image
                    --require-count runtime_shadow_miss 1
                    --require-field runtime_shadow_miss reason no_cached_source
                    --require-field runtime_shadow_miss source_count 0
                    --require-field runtime_shadow_miss attempted_sources 0
                    --require-field runtime_shadow_source inserted true
                    --require-min-field runtime_shadow_source kernel_count 2
                    --require-field runtime_shadow_registration_release removed_sources 1
                    --require-field runtime_shadow_registration_release removed_functions 0
                    --require-field runtime_shadow_registration_release shadow_modules 0
                    --require-field runtime_shadow_registration_release unload_attempts 0
                    --require-field runtime_shadow_registration_release unload_failures 0
                    --require-field runtime_shadow_registration_release last_unload_error 0
                    --require-field patch_device_elf success true
                    --require-field code_object_image top_level_raw_elf true
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_fatbin_host
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running rocfuzz AFL DBI KPACK runtime-shadow miss report smoke target"
            USES_TERMINAL
            VERBATIM)

        set(ROCFUZZ_SAFE_EDGE_CCOB_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/ccob_safe_edge_report.jsonl)
        set(ROCFUZZ_SAFE_EDGE_KPACK_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/kpack_safe_edge_report.jsonl)
        add_custom_target(rocjitsu_afl_kpack_ccob_safe_edge_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_SAFE_EDGE_CCOB_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_VERBOSE=1
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_DEBUG_EDGE_LIMIT=64
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_SAFE_EDGE_CCOB_REPORT}
                    ROCFUZZ_BRANCHY_EXPECT_RESULT=${ROCFUZZ_BRANCHY_EXPECTED_RESULT}
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_branchy_host>
                    ${ROCFUZZ_SMOKE_CCOB}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_SAFE_EDGE_CCOB_REPORT}
                    --require-event patch_device_elf
                    --require-event code_object_image
                    --require-field patch_device_elf success true
                    --require-field code_object_image top_level_ccob true
                    --require-field patch_device_elf coverage_strategy self-contained-fixed-branch
                    --require-field patch_device_elf coverage_signal fixed-branch-counters
                    --require-field patch_device_elf coverage_strategy_reason loader-context-self-contained
                    --require-field patch_device_elf branch_edge_slot_policy fixed-counter
                    --require-list-field patch_device_elf kernel_summaries previous_bb_branch_aggregate_limit_kind none
                    --require-list-field patch_device_elf kernel_summaries previous_bb_branch_aggregate_safety full-previous-bb-within-auto-budget
                    --require-field patch_device_elf entries_patched 0
                    --require-field patch_device_elf branch_edge_site_limit 100
                    --require-field patch_device_elf branch_edge_site_limit_auto true
                    --require-field patch_device_elf previous_bb_branch_site_limit 4294967295
                    --require-field patch_device_elf previous_bb_branch_site_limit_auto true
                    --require-sum-field patch_device_elf edge_sites_patched 6
                    --require-sum-field patch_device_elf fixed_edge_sites 6
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_SAFE_EDGE_KPACK_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_VERBOSE=1
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_DEBUG_FORCE_RUNTIME_SHADOW_MODULES=1
                    # KPACK instrumentation patches one launched kernel at a
                    # time. The first scoped HSA reader patch and the later
                    # runtime-shadow patch both validate the default
                    # loader-scoped fixed branch policy.
                    ROCJITSU_AFL_DEBUG_EDGE_LIMIT=13
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_SAFE_EDGE_KPACK_REPORT}
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    ${ROCFUZZ_FATBIN_HOST}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_SAFE_EDGE_KPACK_REPORT}
                    --require-event patch_device_elf
                    --require-event code_object_image
                    --require-event runtime_shadow_source
                    --require-event runtime_shadow_function
                    --require-event runtime_shadow_launch
                    --require-field patch_device_elf success true
                    --require-field code_object_image top_level_raw_elf true
                    --require-field patch_device_elf coverage_strategy self-contained-fixed-branch
                    --require-field patch_device_elf coverage_signal fixed-branch-counters
                    --require-list-field patch_device_elf kernel_summaries previous_bb_branch_aggregate_safety not-previous-bb-policy
                    --require-field patch_device_elf coverage_strategy_reason loader-context-fixed-branch-preferred
                    --require-field patch_device_elf branch_edge_slot_policy fixed-counter
                    --require-sum-field patch_device_elf edge_sites_patched 8
                    --require-sum-field patch_device_elf fixed_edge_sites 8
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_host
                    rocjitsu_afl_branchy_kernels_ccob
                    rocjitsu_afl_branchy_fatbin_host
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running rocfuzz AFL DBI safe-edge CCOB and KPACK report smoke target"
            USES_TERMINAL
            VERBATIM)

        set(ROCFUZZ_SCRATCH_POLICY_CCOB_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/ccob_scratch_spill_policy_report.jsonl)
        set(ROCFUZZ_SCRATCH_POLICY_KPACK_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/kpack_scratch_spill_policy_report.jsonl)
        add_custom_target(rocjitsu_afl_kpack_ccob_scratch_spill_policy_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_SCRATCH_POLICY_CCOB_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_VERBOSE=1
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_SCRATCH_POLICY_CCOB_REPORT}
                    ROCFUZZ_REPEAT_BRANCHY_KERNEL_B=1
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_branchy_host>
                    ${ROCFUZZ_SMOKE_CCOB}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_SCRATCH_POLICY_CCOB_REPORT}
                    --require-event patch_device_elf
                    --require-event lazy_ccob_launch
                    --require-event device_edge_delta
                    --require-count patch_device_elf 2
                    --require-field patch_device_elf vgpr_scratch_spills_requested true
                    --require-field patch_device_elf vgpr_scratch_spills_enabled false
                    --require-field patch_device_elf vgpr_scratch_spill_reason loader-context-scratch-spills-unproven
                    --require-field patch_device_elf scratch_spill_probe_points 0
                    --require-min-field device_edge_delta edge_slot_delta_count 1
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_SCRATCH_POLICY_KPACK_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_VERBOSE=1
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_DEBUG_FORCE_RUNTIME_SHADOW_MODULES=1
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_SCRATCH_POLICY_KPACK_REPORT}
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    ${ROCFUZZ_FATBIN_HOST}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_SCRATCH_POLICY_KPACK_REPORT}
                    --require-event patch_device_elf
                    --require-event hsa_reader_patch
                    --require-event runtime_shadow_launch
                    --require-event device_edge_delta
                    --require-count patch_device_elf 2
                    --require-field patch_device_elf vgpr_scratch_spills_requested true
                    --require-field patch_device_elf vgpr_scratch_spills_enabled false
                    --require-field patch_device_elf vgpr_scratch_spill_reason loader-context-scratch-spills-unproven
                    --require-field patch_device_elf scratch_spill_probe_points 0
                    --require-min-field device_edge_delta edge_slot_delta_count 1
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_host
                    rocjitsu_afl_branchy_kernels_ccob
                    rocjitsu_afl_branchy_fatbin_host
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running rocfuzz AFL DBI CCOB/KPACK scratch-spill policy report smoke target"
            USES_TERMINAL
            VERBATIM)

        set(ROCFUZZ_KPACK_METADATA_RESIZE_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/kpack_metadata_resize_report.jsonl)
        add_custom_target(rocjitsu_afl_kpack_metadata_resize_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_KPACK_METADATA_RESIZE_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_VERBOSE=1
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_DEBUG_FORCE_RUNTIME_SHADOW_MODULES=1
                    ROCJITSU_AFL_DEBUG_FORCE_RUNTIME_SHADOW_PRIVATE_SEGMENT_BYTES=256
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_KPACK_METADATA_RESIZE_REPORT}
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    ${ROCFUZZ_FATBIN_HOST}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_KPACK_METADATA_RESIZE_REPORT}
                    --require-event patch_device_elf
                    --require-event hsa_reader_patch
                    --require-event runtime_shadow_function
                    --require-event runtime_shadow_launch
                    --require-event device_edge_delta
                    --require-field patch_device_elf success true
                    --require-field patch_device_elf context "runtime KPACK shadow"
                    --require-list-field patch_device_elf descriptor_resources private_segment_metadata_patch rebuilt-note-section
                    --require-list-field patch_device_elf descriptor_resources patched_private_segment_enabled true
                    --require-list-field patch_device_elf descriptor_resources patched_private_segment_fixed_size 256
                    --require-min-field patch_device_elf probe_required_private_segment_bytes 256
                    --require-min-field device_edge_delta edge_slot_delta_count 1
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_fatbin_host
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Checking runtime-shadow loading of rebuilt AMDGPU metadata notes"
            USES_TERMINAL
            VERBATIM)
        set(ROCFUZZ_HIGH_EDGE_KPACK_A_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/kpack_high_edge_kernel_a_report.jsonl)
        set(ROCFUZZ_HIGH_EDGE_KPACK_B_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/kpack_high_edge_kernel_b_report.jsonl)
        add_custom_target(rocjitsu_afl_kpack_ccob_high_edge_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_HIGH_EDGE_KPACK_A_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCFUZZ_SKIP_FATBIN_KERNEL_B=1
                    ROCJITSU_AFL_VERBOSE=1
                    ROCJITSU_AFL_DEBUG_FORCE_RUNTIME_SHADOW_MODULES=1
                    ROCJITSU_AFL_DEBUG_FIXED_EDGE_SLOTS=1
                    ROCJITSU_AFL_DEBUG_EDGE_LIMIT=64
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_HIGH_EDGE_KPACK_A_REPORT}
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    ${ROCFUZZ_FATBIN_HOST}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_HIGH_EDGE_KPACK_A_REPORT}
                    --require-event patch_device_elf
                    --require-field patch_device_elf success false
                    --require-field patch_device_elf reason descriptor_resource_patch_failed
                    --require-field patch_device_elf coverage_strategy entry-previous-bb-block
                    --require-field patch_device_elf entries_patched 0
                    --require-field patch_device_elf edge_sites_selected 16
                    --require-field patch_device_elf edge_sites_patched 0
                    --require-list-item patch_device_elf kernel_summaries
                        kernel=_Z15fatbin_kernel_aPKhPj block_selected=16
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_HIGH_EDGE_KPACK_B_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCFUZZ_SKIP_FATBIN_KERNEL_A=1
                    ROCJITSU_AFL_VERBOSE=1
                    ROCJITSU_AFL_DEBUG_FORCE_RUNTIME_SHADOW_MODULES=1
                    ROCJITSU_AFL_DEBUG_FIXED_EDGE_SLOTS=1
                    ROCJITSU_AFL_DEBUG_EDGE_LIMIT=64
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_HIGH_EDGE_KPACK_B_REPORT}
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    ${ROCFUZZ_FATBIN_HOST}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_HIGH_EDGE_KPACK_B_REPORT}
                    --require-event patch_device_elf
                    --require-field patch_device_elf success false
                    --require-field patch_device_elf reason descriptor_resource_patch_failed
                    --require-field patch_device_elf coverage_strategy entry-previous-bb-block
                    --require-field patch_device_elf entries_patched 0
                    --require-field patch_device_elf edge_sites_selected 7
                    --require-field patch_device_elf edge_sites_patched 0
                    --require-list-item patch_device_elf kernel_summaries
                        kernel=_Z15fatbin_kernel_bPKjPj block_selected=7
            DEPENDS
                    rocjitsu_afl_kpack_ccob_safe_edge_report_smoke
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_fatbin_host
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Checking forced legacy entry-backed high-edge KPACK probes fail closed"
            USES_TERMINAL
            VERBATIM)
    else()
        message(STATUS "rocfuzz KPACK runtime-shadow report smoke disabled (python3 not found)")
    endif()

    add_custom_target(rocjitsu_afl_wave64_smoke
        COMMAND ${CMAKE_COMMAND} -E env
                ROCJITSU_AFL_VERBOSE=1
                ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                $<TARGET_FILE:rocjitsu_afl_branchy_host>
                ${ROCFUZZ_SMOKE_WAVE64_HSACO}
                ${ROCFUZZ_SMOKE_SEED}
        DEPENDS
                rocjitsu_afl_preload
                rocjitsu_afl_branchy_host
                rocjitsu_afl_branchy_kernels_wave64_hsaco
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running rocfuzz AFL DBI wave64 preload smoke target"
        USES_TERMINAL
        VERBATIM)

    if(ROCFUZZ_PYTHON3_EXECUTABLE)
        add_custom_target(rocjitsu_afl_wave64_default_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_WAVE64_DEFAULT_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_VERBOSE=1
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_WAVE64_DEFAULT_REPORT}
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_branchy_host>
                    ${ROCFUZZ_SMOKE_WAVE64_HSACO}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_WAVE64_DEFAULT_REPORT}
                    --require-event patch_device_elf
                    --require-event device_edge_delta
                    --require-field patch_device_elf success true
                    --require-field patch_device_elf coverage_strategy self-contained-hybrid-previous-bb-and-fixed-branch
                    --require-field patch_device_elf coverage_signal hybrid-previous-bb-and-fixed
                    --require-field patch_device_elf branch_edge_slot_policy previous-bb-hash
                    --require-min-field patch_device_elf edge_sites_patched 1
                    --require-min-field patch_device_elf hashed_edge_sites 1
                    --require-min-field patch_device_elf fixed_edge_sites 1
                    --require-min-field patch_device_elf previous_bb_branch_edges_selected 1
                    --require-min-field patch_device_elf branch_edges_degraded_to_fixed 1
                    --require-min-field device_edge_delta edge_slot_delta_count 1
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_host
                    rocjitsu_afl_branchy_kernels_wave64_hsaco
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Checking default wave64 previous-BB branch-edge coverage"
            USES_TERMINAL
            VERBATIM)

        add_custom_target(rocjitsu_afl_wave64_previous_bb_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_WAVE64_PREVIOUS_BB_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_VERBOSE=1
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE=1
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY=hashed
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_LIMIT=15
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_WAVE64_PREVIOUS_BB_REPORT}
                    ROCFUZZ_BRANCHY_EXPECT_RESULT=${ROCFUZZ_BRANCHY_EXPECTED_RESULT}
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_branchy_host>
                    ${ROCFUZZ_SMOKE_WAVE64_HSACO}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_WAVE64_PREVIOUS_BB_REPORT}
                    --require-event patch_device_elf
                    --require-event device_edge_delta
                    --require-field patch_device_elf success true
                    --require-field patch_device_elf skip_entry_probe true
                    --require-field patch_device_elf coverage_strategy self-contained-previous-bb-branch
                    --require-field patch_device_elf branch_edge_slot_policy previous-bb-hash
                    --require-field patch_device_elf require_liveness_registers true
                    --require-field patch_device_elf branch_edge_site_limit 15
                    --require-field patch_device_elf branch_edge_site_limit_auto false
                    --require-min-field patch_device_elf edge_sites_patched 10
                    --require-min-field patch_device_elf hashed_edge_sites 10
                    --require-min-field patch_device_elf previous_bb_branch_edges_selected 15
                    --require-min-field device_edge_delta edge_slot_delta_count 1
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_host
                    rocjitsu_afl_branchy_kernels_wave64_hsaco
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Checking bounded forced wave64 previous-BB branch-edge coverage"
            USES_TERMINAL
            VERBATIM)

        add_custom_target(rocjitsu_afl_rdna3_wave64_previous_bb_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_RDNA3_WAVE64_PREVIOUS_BB_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    HIP_VISIBLE_DEVICES=${ROCFUZZ_RDNA3_VISIBLE_DEVICES}
                    ROCJITSU_AFL_VERBOSE=1
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE=1
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY=hashed
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_LIMIT=15
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_RDNA3_WAVE64_PREVIOUS_BB_REPORT}
                    ROCFUZZ_BRANCHY_EXPECT_RESULT=${ROCFUZZ_BRANCHY_EXPECTED_RESULT}
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_branchy_host>
                    ${ROCFUZZ_SMOKE_RDNA3_WAVE64_HSACO}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_RDNA3_WAVE64_PREVIOUS_BB_REPORT}
                    --require-event patch_device_elf
                    --require-event device_edge_delta
                    --require-field patch_device_elf success true
                    --require-field patch_device_elf gfxip gfx1100
                    --require-field patch_device_elf skip_entry_probe true
                    --require-field patch_device_elf coverage_strategy self-contained-previous-bb-branch
                    --require-field patch_device_elf branch_edge_slot_policy previous-bb-hash
                    --require-field patch_device_elf require_liveness_registers true
                    --require-field patch_device_elf branch_edge_site_limit 15
                    --require-field patch_device_elf branch_edge_site_limit_auto false
                    --require-min-field patch_device_elf edge_sites_patched 10
                    --require-min-field patch_device_elf hashed_edge_sites 10
                    --require-min-field patch_device_elf previous_bb_branch_edges_selected 15
                    --require-min-field device_edge_delta edge_slot_delta_count 1
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_host
                    rocjitsu_afl_branchy_kernels_rdna3_wave64_hsaco
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Checking bounded forced gfx1100 wave64 previous-BB branch-edge coverage"
            USES_TERMINAL
            VERBATIM)
    endif()

    add_custom_target(rocjitsu_afl_high_lane_wave64_smoke
        COMMAND ${CMAKE_COMMAND} -E env
                ROCJITSU_AFL_VERBOSE=1
                ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                $<TARGET_FILE:rocjitsu_afl_high_lane_host>
                ${ROCFUZZ_HIGH_LANE_WAVE64_HSACO}
                ${ROCFUZZ_HIGH_LANE_SEED_B}
        DEPENDS
                rocjitsu_afl_preload
                rocjitsu_afl_high_lane_host
                rocjitsu_afl_high_lane_kernels_wave64_hsaco
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running rocfuzz AFL DBI high-lane wave64 preload smoke target"
        USES_TERMINAL
        VERBATIM)

    add_custom_target(rocjitsu_afl_persistent_smoke
        COMMAND ${CMAKE_COMMAND} -E env
                ROCJITSU_AFL_VERBOSE=1
                ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                ROCJITSU_AFL_PERSISTENT=1
                ROCJITSU_AFL_REQUIRE_PERSISTENT_HOOKS=1
                LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                $<TARGET_FILE:rocjitsu_afl_branchy_persistent_host>
                ${ROCFUZZ_SMOKE_HSACO}
                ${ROCFUZZ_PERSISTENT_SEED_A}
                ${ROCFUZZ_PERSISTENT_SEED_B}
        DEPENDS
                rocjitsu_afl_preload
                rocjitsu_afl_branchy_persistent_host
                rocjitsu_afl_branchy_kernels_hsaco
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running rocfuzz AFL DBI persistent-boundary smoke target"
        USES_TERMINAL
        VERBATIM)

    add_custom_target(rocjitsu_afl_rdna3_smoke
        COMMAND ${CMAKE_COMMAND} -E env
                HIP_VISIBLE_DEVICES=${ROCFUZZ_RDNA3_VISIBLE_DEVICES}
                ROCJITSU_AFL_VERBOSE=1
                ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                $<TARGET_FILE:rocjitsu_afl_branchy_host>
                ${ROCFUZZ_SMOKE_RDNA3_HSACO}
                ${ROCFUZZ_SMOKE_SEED}
        DEPENDS
                rocjitsu_afl_preload
                rocjitsu_afl_branchy_host
                rocjitsu_afl_branchy_kernels_rdna3_hsaco
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running rocfuzz AFL DBI preload smoke target on gfx1100"
        USES_TERMINAL
        VERBATIM)

    add_custom_target(rocjitsu_afl_rdna3_wave64_smoke
        COMMAND ${CMAKE_COMMAND} -E env
                HIP_VISIBLE_DEVICES=${ROCFUZZ_RDNA3_VISIBLE_DEVICES}
                ROCJITSU_AFL_VERBOSE=1
                ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                $<TARGET_FILE:rocjitsu_afl_branchy_host>
                ${ROCFUZZ_SMOKE_RDNA3_WAVE64_HSACO}
                ${ROCFUZZ_SMOKE_SEED}
        DEPENDS
                rocjitsu_afl_preload
                rocjitsu_afl_branchy_host
                rocjitsu_afl_branchy_kernels_rdna3_wave64_hsaco
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running rocfuzz AFL DBI wave64 preload smoke target on gfx1100"
        USES_TERMINAL
        VERBATIM)

    add_custom_target(rocjitsu_afl_rdna3_high_lane_wave64_smoke
        COMMAND ${CMAKE_COMMAND} -E env
                HIP_VISIBLE_DEVICES=${ROCFUZZ_RDNA3_VISIBLE_DEVICES}
                ROCJITSU_AFL_VERBOSE=1
                ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                $<TARGET_FILE:rocjitsu_afl_high_lane_host>
                ${ROCFUZZ_HIGH_LANE_RDNA3_WAVE64_HSACO}
                ${ROCFUZZ_HIGH_LANE_SEED_B}
        DEPENDS
                rocjitsu_afl_preload
                rocjitsu_afl_high_lane_host
                rocjitsu_afl_high_lane_kernels_rdna3_wave64_hsaco
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running rocfuzz AFL DBI high-lane wave64 preload smoke target on gfx1100"
        USES_TERMINAL
        VERBATIM)

    add_custom_target(rocjitsu_afl_branch_edge_smoke
        COMMAND ${CMAKE_COMMAND} -E env
                ROCJITSU_AFL_VERBOSE=1
                ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                $<TARGET_FILE:rocjitsu_afl_branchy_host>
                ${ROCFUZZ_SMOKE_HSACO}
                ${ROCFUZZ_SMOKE_SEED}
        DEPENDS
                rocjitsu_afl_preload
                rocjitsu_afl_branchy_host
                rocjitsu_afl_branchy_kernels_hsaco
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running rocfuzz AFL DBI branch-edge smoke target"
        USES_TERMINAL
        VERBATIM)

    add_custom_target(rocjitsu_afl_branch_edge_hashed_smoke
        COMMAND ${CMAKE_COMMAND} -E env
                ROCJITSU_AFL_VERBOSE=1
                ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE=1
                ROCJITSU_AFL_DEBUG_EDGE_LIMIT=0
                ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOTS=1
                ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY=hashed
                ROCFUZZ_BRANCHY_EXPECT_RESULT=${ROCFUZZ_BRANCHY_EXPECTED_RESULT}
                LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                $<TARGET_FILE:rocjitsu_afl_branchy_host>
                ${ROCFUZZ_SMOKE_HSACO}
                ${ROCFUZZ_SMOKE_SEED}
        DEPENDS
                rocjitsu_afl_preload
                rocjitsu_afl_branchy_host
                rocjitsu_afl_branchy_kernels_hsaco
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running rocfuzz AFL DBI hashed branch-edge smoke target"
        USES_TERMINAL
        VERBATIM)

    if(ROCFUZZ_PYTHON3_EXECUTABLE)
        set(ROCFUZZ_STRICT_HASHED_BRANCH_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/strict_hashed_branch_report.jsonl)
        add_custom_target(rocjitsu_afl_branch_edge_hashed_strict_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_STRICT_HASHED_BRANCH_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE=1
                    ROCJITSU_AFL_DEBUG_EDGE_LIMIT=0
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOTS=1
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY=hashed
                    ROCJITSU_AFL_DEBUG_REQUIRE_LIVENESS_REGISTERS=1
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_STRICT_HASHED_BRANCH_REPORT}
                    ROCFUZZ_BRANCHY_EXPECT_RESULT=${ROCFUZZ_BRANCHY_EXPECTED_RESULT}
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_branchy_host>
                    ${ROCFUZZ_SMOKE_HSACO}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_STRICT_HASHED_BRANCH_REPORT}
                    --require-event patch_device_elf
                    --require-event device_edge_delta
                    --require-field patch_device_elf skip_entry_probe true
                    --require-field patch_device_elf branch_edge_slot_policy previous-bb-hash
                    --require-field patch_device_elf require_liveness_registers true
                    --require-field patch_device_elf register_plan mixed-liveness-edge-registers
                    --require-min-field patch_device_elf edge_sites_patched 1
                    --require-min-field patch_device_elf hashed_edge_sites 1
                    --require-min-field patch_device_elf liveness_register_kernels 1
                    --require-min-field patch_device_elf liveness_probe_points 1
                    --require-min-field device_edge_delta edge_slot_delta_count 1
                    --require-list-field patch_device_elf kernel_summaries fresh_registers true
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_host
                    rocjitsu_afl_branchy_kernels_hsaco
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running rocfuzz AFL DBI strict hashed branch-edge smoke target"
            USES_TERMINAL
            VERBATIM)

        set(ROCFUZZ_SGPR_GROWTH_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/sgpr_growth_report.jsonl)
        add_custom_target(rocjitsu_afl_sgpr_growth_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_SGPR_GROWTH_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE=1
                    ROCJITSU_AFL_DEBUG_EDGE_LIMIT=0
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOTS=1
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY=hashed
                    ROCJITSU_AFL_DEBUG_REQUIRE_LIVENESS_REGISTERS=1
                    ROCJITSU_AFL_DEBUG_FORCE_FRESH_SGPRS=1
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_SGPR_GROWTH_REPORT}
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_sgpr_growth_host>
                    ${ROCFUZZ_SGPR_GROWTH_HSACO}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_SGPR_GROWTH_REPORT}
                    --require-event patch_device_elf
                    --require-event device_edge_delta
                    --require-field patch_device_elf success false
                    --require-field patch_device_elf reason no_patchable_sites
                    --require-field patch_device_elf skip_entry_probe true
                    --require-field patch_device_elf branch_edge_slot_policy previous-bb-hash
                    --require-field patch_device_elf require_liveness_registers true
                    --require-field patch_device_elf register_plan fixed-high-registers
                    --require-field patch_device_elf edge_sites_patched 0
                    --require-field patch_device_elf hashed_edge_sites 0
                    --require-list-item patch_device_elf kernel_summaries
                        kernel=sgpr_growth_kernel
                        fresh_registers=false
                    --require-list-item patch_device_elf kernel_summaries.skip_reason_counts
                        kind=branch
                        "reason=fresh SGPR growth disabled by target resource model"
                    --require-field device_edge_delta edge_slot_delta_count 0
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_sgpr_growth_host
                    rocjitsu_afl_sgpr_growth_kernels_hsaco
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Checking gfx10+ SGPR fresh-growth fail-closed report"
            USES_TERMINAL
            VERBATIM)

        set(ROCFUZZ_SCRATCH_SPILL_BRANCH_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/scratch_spill_branch_report.jsonl)
        add_custom_target(rocjitsu_afl_scratch_spill_branch_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_SCRATCH_SPILL_BRANCH_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE=1
                    ROCJITSU_AFL_DEBUG_EDGE_LIMIT=0
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOTS=1
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY=hashed
                    ROCJITSU_AFL_DEBUG_REQUIRE_LIVENESS_REGISTERS=1
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_SCRATCH_SPILL_BRANCH_REPORT}
                    ROCFUZZ_BRANCHY_EXPECT_RESULT=${ROCFUZZ_BRANCHY_EXPECTED_RESULT}
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_branchy_host>
                    ${ROCFUZZ_SMOKE_HSACO}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_SCRATCH_SPILL_BRANCH_REPORT}
                    --require-event patch_device_elf
                    --require-event device_edge_delta
                    --require-field patch_device_elf gfxip ${ROCFUZZ_GPU_ARCH}
                    --require-field patch_device_elf skip_entry_probe true
                    --require-field patch_device_elf branch_edge_slot_policy previous-bb-hash
                    --require-field patch_device_elf require_liveness_registers true
                    --require-field patch_device_elf vgpr_scratch_spills_requested true
                    --require-field patch_device_elf vgpr_scratch_spills_enabled true
                    --require-field patch_device_elf vgpr_scratch_spill_reason loader-context-scratch-spills-supported
                    --require-field patch_device_elf register_plan mixed-liveness-edge-registers
                    --require-min-field patch_device_elf edge_sites_patched 1
                    --require-min-field patch_device_elf scratch_spill_probe_points 1
                    --require-min-field patch_device_elf probe_required_private_segment_bytes 1
                    --require-min-field patch_device_elf spill_bytes 1
                    --require-list-field patch_device_elf descriptor_resources patched_private_segment_enabled true
                    --require-list-field patch_device_elf descriptor_resources private_segment_metadata_patch in-place-bytes
                    --require-list-field patch_device_elf descriptor_resources resource_fields_changed true
                    --require-min-field device_edge_delta edge_slot_delta_count 1
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_host
                    rocjitsu_afl_branchy_kernels_hsaco
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running rocfuzz AFL DBI scratch-spill branch-edge report smoke target"
            USES_TERMINAL
            VERBATIM)

        set(ROCFUZZ_RDNA3_SCRATCH_SPILL_BRANCH_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/scratch_spill_branch_gfx1100_report.jsonl)
        add_custom_target(rocjitsu_afl_rdna3_scratch_spill_branch_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_RDNA3_SCRATCH_SPILL_BRANCH_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    HIP_VISIBLE_DEVICES=${ROCFUZZ_RDNA3_VISIBLE_DEVICES}
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE=1
                    ROCJITSU_AFL_DEBUG_EDGE_LIMIT=0
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOTS=1
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY=hashed
                    ROCJITSU_AFL_DEBUG_REQUIRE_LIVENESS_REGISTERS=1
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_RDNA3_SCRATCH_SPILL_BRANCH_REPORT}
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_branchy_host>
                    ${ROCFUZZ_SMOKE_RDNA3_HSACO}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_RDNA3_SCRATCH_SPILL_BRANCH_REPORT}
                    --require-event patch_device_elf
                    --require-event device_edge_delta
                    --require-field patch_device_elf gfxip gfx1100
                    --require-field patch_device_elf skip_entry_probe true
                    --require-field patch_device_elf branch_edge_slot_policy previous-bb-hash
                    --require-field patch_device_elf require_liveness_registers true
                    --require-field patch_device_elf vgpr_scratch_spills_requested true
                    --require-field patch_device_elf vgpr_scratch_spills_enabled true
                    --require-field patch_device_elf vgpr_scratch_spill_reason loader-context-scratch-spills-supported
                    --require-field patch_device_elf register_plan mixed-liveness-edge-registers
                    --require-min-field patch_device_elf edge_sites_patched 1
                    --require-min-field patch_device_elf scratch_spill_probe_points 1
                    --require-min-field patch_device_elf probe_required_private_segment_bytes 1
                    --require-min-field patch_device_elf spill_bytes 1
                    --require-list-field patch_device_elf descriptor_resources patched_private_segment_enabled true
                    --require-list-field patch_device_elf descriptor_resources private_segment_metadata_patch in-place-bytes
                    --require-list-field patch_device_elf descriptor_resources resource_fields_changed true
                    --require-min-field device_edge_delta edge_slot_delta_count 1
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_host
                    rocjitsu_afl_branchy_kernels_rdna3_hsaco
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running rocfuzz AFL DBI scratch-spill branch-edge report smoke target on gfx1100"
            USES_TERMINAL
            VERBATIM)

        set(ROCFUZZ_LIVENESS_REGISTER_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/liveness_register_report.jsonl)
        add_custom_target(rocjitsu_afl_liveness_register_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_LIVENESS_REGISTER_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                    ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE=1
                    ROCJITSU_AFL_DEBUG_EDGE_LIMIT=0
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOTS=1
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_LIVENESS_REGISTER_REPORT}
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_branchy_host>
                    ${ROCFUZZ_SMOKE_HSACO}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_LIVENESS_REGISTER_REPORT}
                    --require-event patch_device_elf
                    --require-event device_edge_delta
                    --require-field patch_device_elf register_plan mixed-liveness-edge-registers
                    --require-min-field patch_device_elf liveness_register_kernels 1
                    --require-min-field patch_device_elf liveness_probe_points 1
                    --require-list-field patch_device_elf descriptor_resources patched_vgpr_count 8
                    --require-list-field patch_device_elf descriptor_resources has_metadata_sgpr_count true
                    --require-list-field patch_device_elf descriptor_resources descriptor_sgpr_count_effective false
                    --require-min-field device_edge_delta edge_slot_delta_count 1
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_host
                    rocjitsu_afl_branchy_kernels_hsaco
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running rocfuzz AFL DBI liveness-register report smoke target"
            USES_TERMINAL
            VERBATIM)

        set(ROCFUZZ_STATE_LIVENESS_REGISTER_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/state_liveness_register_report.jsonl)
        add_custom_target(rocjitsu_afl_state_liveness_register_fail_closed_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_STATE_LIVENESS_REGISTER_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_DEBUG_EDGE_LIMIT=0
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOTS=1
                    ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY=fixed
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_STATE_LIVENESS_REGISTER_REPORT}
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_branchy_host>
                    ${ROCFUZZ_SMOKE_HSACO}
                    ${ROCFUZZ_SMOKE_SEED}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_STATE_LIVENESS_REGISTER_REPORT}
                    --require-event patch_device_elf
                    --require-event device_edge_delta
                    --require-field patch_device_elf success false
                    --require-field patch_device_elf reason descriptor_resource_patch_failed
                    --require-field patch_device_elf failure_phase plan
                    --require-field patch_device_elf descriptor_resource_failure_reason
                        "SGPR growth unsupported because SGPR count comes from AMDGPU metadata"
                    --require-field patch_device_elf coverage_strategy branch-edge-diagnostic
                    --require-field patch_device_elf coverage_signal none
                    --require-field patch_device_elf branch_edge_slot_policy fixed-counter
                    --require-field patch_device_elf register_plan fixed-high-registers
                    --require-field patch_device_elf entries_patched 0
                    --require-field patch_device_elf edge_sites_patched 0
                    --require-field patch_device_elf fixed_edge_sites 7
                    --require-list-field patch_device_elf kernel_summaries
                        coverage_strategy entry-backed-fixed-branch
                    --require-field device_edge_delta edge_slot_delta_count 0
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_host
                    rocjitsu_afl_branchy_kernels_hsaco
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Checking legacy entry-backed branch-state diagnostic fails closed"
            USES_TERMINAL
            VERBATIM)
        add_custom_target(rocjitsu_afl_state_liveness_register_report_smoke
            DEPENDS rocjitsu_afl_state_liveness_register_fail_closed_report_smoke)

        set(ROCFUZZ_ENTRY_LIVENESS_REGISTER_REPORT
            ${CMAKE_CURRENT_BINARY_DIR}/entry_liveness_register_report.jsonl)
        add_custom_target(rocjitsu_afl_entry_liveness_register_report_smoke
            COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_ENTRY_LIVENESS_REGISTER_REPORT}
            COMMAND ${CMAKE_COMMAND} -E env
                    ROCJITSU_AFL_DEBUG_DISABLE_EDGES=1
                    ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_ENTRY_LIVENESS_REGISTER_REPORT}
                    LD_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    $<TARGET_FILE:rocjitsu_afl_entry_liveness_host>
                    ${ROCFUZZ_ENTRY_LIVENESS_HSACO}
            COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_patch_report_events.py
                    ${ROCFUZZ_ENTRY_LIVENESS_REGISTER_REPORT}
                    --require-event patch_device_elf
                    --require-field patch_device_elf success true
                    --require-field patch_device_elf register_plan mixed-liveness-entry-registers
                    --require-min-field patch_device_elf entry_liveness_register_kernels 1
                    --require-min-field patch_device_elf entry_liveness_probe_points 1
                    --require-field patch_device_elf probe_required_sgprs 4
                    --require-field patch_device_elf probe_required_vgprs 3
                    --require-nonempty-list patch_device_elf descriptor_resources
                    --require-list-field patch_device_elf descriptor_resources has_metadata_sgpr_count true
                    --require-list-field patch_device_elf descriptor_resources descriptor_sgpr_count_effective false
                    --require-list-field patch_device_elf descriptor_resources patched_vgpr_count 8
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_entry_liveness_host
                    rocjitsu_afl_entry_liveness_kernels_hsaco
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running rocfuzz AFL DBI entry-liveness-register report smoke target"
            USES_TERMINAL
            VERBATIM)
    endif()

    if(ROCFUZZ_AFL_CXX_EXECUTABLE AND ROCFUZZ_AFL_SHOWMAP_EXECUTABLE)
        get_filename_component(ROCFUZZ_HIP_RUNTIME_DIR ${ROCFUZZ_HIP_RUNTIME} DIRECTORY)
        set(ROCFUZZ_AFL_BRANCHY_HOST
            ${CMAKE_CURRENT_BINARY_DIR}/branchy_host_afl)
        set(ROCFUZZ_AFL_HIGH_LANE_HOST
            ${CMAKE_CURRENT_BINARY_DIR}/high_lane_host_afl)
        set(ROCFUZZ_AFL_SEED_A ${CMAKE_CURRENT_BINARY_DIR}/seed.edge.a)
        set(ROCFUZZ_AFL_SEED_B ${CMAKE_CURRENT_BINARY_DIR}/seed.edge.b)
        set(ROCFUZZ_AFL_SHOWMAP_A ${CMAKE_CURRENT_BINARY_DIR}/showmap.edge.a)
        set(ROCFUZZ_AFL_SHOWMAP_B ${CMAKE_CURRENT_BINARY_DIR}/showmap.edge.b)
        set(ROCFUZZ_AFL_DEVICE_A ${CMAKE_CURRENT_BINARY_DIR}/showmap.edge.a.device)
        set(ROCFUZZ_AFL_DEVICE_B ${CMAKE_CURRENT_BINARY_DIR}/showmap.edge.b.device)
        set(ROCFUZZ_AFL_HIGH_LANE_SHOWMAP_A
            ${CMAKE_CURRENT_BINARY_DIR}/showmap.high_lane.wave64.a)
        set(ROCFUZZ_AFL_HIGH_LANE_SHOWMAP_B
            ${CMAKE_CURRENT_BINARY_DIR}/showmap.high_lane.wave64.b)
        set(ROCFUZZ_AFL_HIGH_LANE_DEVICE_A
            ${CMAKE_CURRENT_BINARY_DIR}/showmap.high_lane.wave64.a.device)
        set(ROCFUZZ_AFL_HIGH_LANE_DEVICE_B
            ${CMAKE_CURRENT_BINARY_DIR}/showmap.high_lane.wave64.b.device)
        set(ROCFUZZ_AFL_RDNA3_HIGH_LANE_SHOWMAP_A
            ${CMAKE_CURRENT_BINARY_DIR}/showmap.high_lane.gfx1100.wave64.a)
        set(ROCFUZZ_AFL_RDNA3_HIGH_LANE_SHOWMAP_B
            ${CMAKE_CURRENT_BINARY_DIR}/showmap.high_lane.gfx1100.wave64.b)
        set(ROCFUZZ_AFL_RDNA3_HIGH_LANE_DEVICE_A
            ${CMAKE_CURRENT_BINARY_DIR}/showmap.high_lane.gfx1100.wave64.a.device)
        set(ROCFUZZ_AFL_RDNA3_HIGH_LANE_DEVICE_B
            ${CMAKE_CURRENT_BINARY_DIR}/showmap.high_lane.gfx1100.wave64.b.device)
        file(WRITE ${ROCFUZZ_AFL_SEED_A} "00000000")
        file(WRITE ${ROCFUZZ_AFL_SEED_B} "11111111")

        add_custom_command(
            OUTPUT ${ROCFUZZ_AFL_BRANCHY_HOST}
            COMMAND ${CMAKE_COMMAND} -E env
                    --unset=AFL_CXX
                    --unset=MAKEFLAGS
                    --unset=MFLAGS
                    ${ROCFUZZ_AFL_CXX_EXECUTABLE}
                    -O2 -g -std=c++17
                    -D__HIP_PLATFORM_AMD__
                    -I${ROCFUZZ_ROCM_PATH}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/smoke/branchy_host.cpp
                    ${ROCFUZZ_HIP_RUNTIME}
                    -Wl,-rpath,${ROCFUZZ_HIP_RUNTIME_DIR}
                    -o ${ROCFUZZ_AFL_BRANCHY_HOST}
            DEPENDS smoke/branchy_host.cpp
            COMMENT "Building AFL-instrumented rocfuzz branchy host"
            VERBATIM)
        add_custom_target(rocjitsu_afl_branchy_host_afl
            DEPENDS ${ROCFUZZ_AFL_BRANCHY_HOST})

        add_custom_command(
            OUTPUT ${ROCFUZZ_AFL_HIGH_LANE_HOST}
            COMMAND ${CMAKE_COMMAND} -E env
                    --unset=AFL_CXX
                    --unset=MAKEFLAGS
                    --unset=MFLAGS
                    ${ROCFUZZ_AFL_CXX_EXECUTABLE}
                    -O2 -g -std=c++17
                    -D__HIP_PLATFORM_AMD__
                    -I${ROCFUZZ_ROCM_PATH}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/smoke/high_lane_host.cpp
                    ${ROCFUZZ_HIP_RUNTIME}
                    -Wl,-rpath,${ROCFUZZ_HIP_RUNTIME_DIR}
                    -o ${ROCFUZZ_AFL_HIGH_LANE_HOST}
            DEPENDS smoke/high_lane_host.cpp
            COMMENT "Building AFL-instrumented rocfuzz high-lane host"
            VERBATIM)
        add_custom_target(rocjitsu_afl_high_lane_host_afl
            DEPENDS ${ROCFUZZ_AFL_HIGH_LANE_HOST})

        add_custom_target(rocjitsu_afl_showmap_edge
            COMMAND ${CMAKE_COMMAND} -E env
                    AFL_SKIP_BIN_CHECK=1
                    AFL_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    ${ROCFUZZ_AFL_SHOWMAP_EXECUTABLE}
                    -m none -t 10000
                    -o ${ROCFUZZ_AFL_SHOWMAP_A}
                    --
                    ${ROCFUZZ_AFL_BRANCHY_HOST}
                    ${ROCFUZZ_SMOKE_HSACO}
                    ${ROCFUZZ_AFL_SEED_A}
            COMMAND ${CMAKE_COMMAND} -E env
                    AFL_SKIP_BIN_CHECK=1
                    AFL_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    ${ROCFUZZ_AFL_SHOWMAP_EXECUTABLE}
                    -m none -t 10000
                    -o ${ROCFUZZ_AFL_SHOWMAP_B}
                    --
                    ${ROCFUZZ_AFL_BRANCHY_HOST}
                    ${ROCFUZZ_SMOKE_HSACO}
                    ${ROCFUZZ_AFL_SEED_B}
            COMMAND /bin/sh -c
                    "awk -F: '$1 + 0 >= 32769 { print }' '${ROCFUZZ_AFL_SHOWMAP_A}' > '${ROCFUZZ_AFL_DEVICE_A}'"
            COMMAND /bin/sh -c
                    "awk -F: '$1 + 0 >= 32769 { print }' '${ROCFUZZ_AFL_SHOWMAP_B}' > '${ROCFUZZ_AFL_DEVICE_B}'"
            COMMAND /bin/sh -c
                    "test -s '${ROCFUZZ_AFL_DEVICE_A}' && test -s '${ROCFUZZ_AFL_DEVICE_B}' && ! cmp -s '${ROCFUZZ_AFL_DEVICE_A}' '${ROCFUZZ_AFL_DEVICE_B}'"
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_branchy_host_afl
                    rocjitsu_afl_branchy_kernels_hsaco
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Checking AFL-visible rocfuzz device edge coverage"
            USES_TERMINAL
            VERBATIM)

        add_custom_target(rocjitsu_afl_high_lane_wave64_showmap
            COMMAND ${CMAKE_COMMAND} -E env
                    AFL_SKIP_BIN_CHECK=1
                    AFL_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    ${ROCFUZZ_AFL_SHOWMAP_EXECUTABLE}
                    -m none -t 10000
                    -o ${ROCFUZZ_AFL_HIGH_LANE_SHOWMAP_A}
                    --
                    ${ROCFUZZ_AFL_HIGH_LANE_HOST}
                    ${ROCFUZZ_HIGH_LANE_WAVE64_HSACO}
                    ${ROCFUZZ_HIGH_LANE_SEED_A}
            COMMAND ${CMAKE_COMMAND} -E env
                    AFL_SKIP_BIN_CHECK=1
                    AFL_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    ${ROCFUZZ_AFL_SHOWMAP_EXECUTABLE}
                    -m none -t 10000
                    -o ${ROCFUZZ_AFL_HIGH_LANE_SHOWMAP_B}
                    --
                    ${ROCFUZZ_AFL_HIGH_LANE_HOST}
                    ${ROCFUZZ_HIGH_LANE_WAVE64_HSACO}
                    ${ROCFUZZ_HIGH_LANE_SEED_B}
            COMMAND /bin/sh -c
                    "awk -F: '$1 + 0 >= 32769 { print }' '${ROCFUZZ_AFL_HIGH_LANE_SHOWMAP_A}' > '${ROCFUZZ_AFL_HIGH_LANE_DEVICE_A}'"
            COMMAND /bin/sh -c
                    "awk -F: '$1 + 0 >= 32769 { print }' '${ROCFUZZ_AFL_HIGH_LANE_SHOWMAP_B}' > '${ROCFUZZ_AFL_HIGH_LANE_DEVICE_B}'"
            COMMAND /bin/sh -c
                    "test -s '${ROCFUZZ_AFL_HIGH_LANE_DEVICE_A}' && test -s '${ROCFUZZ_AFL_HIGH_LANE_DEVICE_B}' && ! cmp -s '${ROCFUZZ_AFL_HIGH_LANE_DEVICE_A}' '${ROCFUZZ_AFL_HIGH_LANE_DEVICE_B}'"
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_high_lane_host_afl
                    rocjitsu_afl_high_lane_kernels_wave64_hsaco
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Checking AFL-visible high-lane-only wave64 device edge coverage"
            USES_TERMINAL
            VERBATIM)

        add_custom_target(rocjitsu_afl_rdna3_high_lane_wave64_showmap
            COMMAND ${CMAKE_COMMAND} -E env
                    HIP_VISIBLE_DEVICES=${ROCFUZZ_RDNA3_VISIBLE_DEVICES}
                    AFL_SKIP_BIN_CHECK=1
                    AFL_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    ${ROCFUZZ_AFL_SHOWMAP_EXECUTABLE}
                    -m none -t 10000
                    -o ${ROCFUZZ_AFL_RDNA3_HIGH_LANE_SHOWMAP_A}
                    --
                    ${ROCFUZZ_AFL_HIGH_LANE_HOST}
                    ${ROCFUZZ_HIGH_LANE_RDNA3_WAVE64_HSACO}
                    ${ROCFUZZ_HIGH_LANE_SEED_A}
            COMMAND ${CMAKE_COMMAND} -E env
                    HIP_VISIBLE_DEVICES=${ROCFUZZ_RDNA3_VISIBLE_DEVICES}
                    AFL_SKIP_BIN_CHECK=1
                    AFL_PRELOAD=$<TARGET_FILE:rocjitsu_afl_preload>
                    ${ROCFUZZ_AFL_SHOWMAP_EXECUTABLE}
                    -m none -t 10000
                    -o ${ROCFUZZ_AFL_RDNA3_HIGH_LANE_SHOWMAP_B}
                    --
                    ${ROCFUZZ_AFL_HIGH_LANE_HOST}
                    ${ROCFUZZ_HIGH_LANE_RDNA3_WAVE64_HSACO}
                    ${ROCFUZZ_HIGH_LANE_SEED_B}
            COMMAND /bin/sh -c
                    "awk -F: '$1 + 0 >= 32769 { print }' '${ROCFUZZ_AFL_RDNA3_HIGH_LANE_SHOWMAP_A}' > '${ROCFUZZ_AFL_RDNA3_HIGH_LANE_DEVICE_A}'"
            COMMAND /bin/sh -c
                    "awk -F: '$1 + 0 >= 32769 { print }' '${ROCFUZZ_AFL_RDNA3_HIGH_LANE_SHOWMAP_B}' > '${ROCFUZZ_AFL_RDNA3_HIGH_LANE_DEVICE_B}'"
            COMMAND /bin/sh -c
                    "test -s '${ROCFUZZ_AFL_RDNA3_HIGH_LANE_DEVICE_A}' && test -s '${ROCFUZZ_AFL_RDNA3_HIGH_LANE_DEVICE_B}' && ! cmp -s '${ROCFUZZ_AFL_RDNA3_HIGH_LANE_DEVICE_A}' '${ROCFUZZ_AFL_RDNA3_HIGH_LANE_DEVICE_B}'"
            DEPENDS
                    rocjitsu_afl_preload
                    rocjitsu_afl_high_lane_host_afl
                    rocjitsu_afl_high_lane_kernels_rdna3_wave64_hsaco
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Checking AFL-visible high-lane-only wave64 device edge coverage on gfx1100"
            USES_TERMINAL
            VERBATIM)
    else()
        message(STATUS "rocfuzz AFL showmap target disabled (AFL++ tools not found)")
    endif()

    if(ROCFUZZ_PYTHON3_EXECUTABLE AND ROCFUZZ_HSA_RUNTIME AND
            ROCFUZZ_AFL_CXX_EXECUTABLE AND ROCFUZZ_AFL_SHOWMAP_EXECUTABLE)
        add_custom_target(rocjitsu_afl_gfx11_gfx12_coverage_matrix
            DEPENDS
                    # Direct-module default adaptive coverage.
                    rocjitsu_afl_hybrid_default_report_smoke
                    rocjitsu_afl_runtime_overhead_smoke
                    rocjitsu_afl_opaque_scratch_guard_report_smoke
                    # CCOB and loader-scoped HSA reader coverage.
                    rocjitsu_afl_ccob_report_smoke
                    rocjitsu_afl_hsa_multi_ccob_reader_report_smoke
                    # KPACK/runtime-shadow coverage. The forced high-edge
                    # entry-backed diagnostic intentionally stays outside this
                    # safe matrix until persistent-state SGPR growth is proven
                    # on gfx10+/RDNA.
                    rocjitsu_afl_kpack_ccob_safe_edge_report_smoke
                    rocjitsu_afl_kpack_ccob_scratch_spill_policy_report_smoke
                    rocjitsu_afl_kpack_metadata_resize_report_smoke
                    # Descriptor plus loader-visible metadata growth for
                    # scratch-backed branch probes.
                    rocjitsu_afl_metadata_resize_loader_smoke
                    rocjitsu_afl_scratch_spill_branch_report_smoke
                    rocjitsu_afl_rdna3_scratch_spill_branch_report_smoke
                    # Wave64 execution for both current gfx11/gfx12 targets.
                    rocjitsu_afl_wave64_default_report_smoke
                    rocjitsu_afl_wave64_previous_bb_report_smoke
                    rocjitsu_afl_rdna3_wave64_previous_bb_report_smoke
                    rocjitsu_afl_rdna3_wave64_smoke
                    rocjitsu_afl_high_lane_wave64_showmap
                    rocjitsu_afl_rdna3_high_lane_wave64_showmap
            COMMENT "Running rocfuzz gfx11/gfx12 coverage regression matrix")
    else()
        message(STATUS "rocfuzz gfx11/gfx12 coverage matrix disabled (python3, hsa-runtime64, or AFL++ tools not found)")
    endif()
else()
    message(STATUS "rocfuzz AFL DBI smoke target disabled (hipcc, amdhip64, or zstd not found)")
endif()
