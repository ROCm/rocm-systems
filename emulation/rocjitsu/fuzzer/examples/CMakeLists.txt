# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.20)

project(RocjitsuFuzzerLibraryExamples LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(RocfuzzTheRock)

rocfuzz_discover_therock()

find_package(hip REQUIRED CONFIG)
find_package(rocblas REQUIRED CONFIG)
find_package(rocfft REQUIRED CONFIG)
find_package(rocrand REQUIRED CONFIG)
find_package(hipblaslt REQUIRED CONFIG)
find_package(rocsparse REQUIRED CONFIG)
find_package(rocsolver REQUIRED CONFIG)
find_package(miopen REQUIRED CONFIG)
find_program(ROCFUZZ_PYTHON3_EXECUTABLE python3)
set(ROCFUZZ_AFL_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/AFLplusplus"
    CACHE PATH "Optional AFL++ checkout used by rocfuzz example showmap smoke targets")
find_program(ROCFUZZ_AFL_SHOWMAP_EXECUTABLE afl-showmap
    HINTS "${ROCFUZZ_AFL_ROOT}")

set(ROCFUZZ_ROCJITSU_BUILD
    "${CMAKE_CURRENT_SOURCE_DIR}/../../build"
    CACHE PATH "rocjitsu build directory containing the rocjitsu_afl_preload target")
set(ROCFUZZ_AFL_DBI_BUILD
    "${ROCFUZZ_ROCJITSU_BUILD}/fuzzer/afl-dbi"
    CACHE PATH "rocjitsu build directory containing the AFL DBI preload library")
set(ROCFUZZ_AFL_PRELOAD
    "${ROCFUZZ_AFL_DBI_BUILD}/librocjitsu_afl_preload.so"
    CACHE FILEPATH "rocjitsu AFL DBI preload library")
option(ROCFUZZ_BUILD_AFL_PRELOAD
    "Build the parent rocjitsu_afl_preload target before example smoke targets"
    ON)

set(ROCFUZZ_AFL_PRELOAD_DEPENDS "${ROCFUZZ_AFL_PRELOAD}")
if(ROCFUZZ_BUILD_AFL_PRELOAD AND
   EXISTS "${ROCFUZZ_ROCJITSU_BUILD}/CMakeCache.txt")
    add_custom_target(rocfuzz_afl_preload_current
        COMMAND ${CMAKE_COMMAND}
                --build "${ROCFUZZ_ROCJITSU_BUILD}"
                --target rocjitsu_afl_preload
        COMMENT "Building current rocjitsu AFL DBI preload"
        VERBATIM)
    set(ROCFUZZ_AFL_PRELOAD_DEPENDS rocfuzz_afl_preload_current)
elseif(ROCFUZZ_BUILD_AFL_PRELOAD)
    message(WARNING
        "ROCFUZZ_ROCJITSU_BUILD is not configured: ${ROCFUZZ_ROCJITSU_BUILD}. "
        "Example smoke targets will use ROCFUZZ_AFL_PRELOAD as an external file.")
endif()

if(NOT TARGET rocfuzz_afl_preload_current AND
   NOT EXISTS "${ROCFUZZ_AFL_PRELOAD}")
    message(WARNING
        "ROCFUZZ_AFL_PRELOAD does not exist yet: ${ROCFUZZ_AFL_PRELOAD}. "
        "Build the rocjitsu_afl_preload target or set ROCFUZZ_AFL_PRELOAD before "
        "running example smoke targets.")
endif()

set(ROCFUZZ_COMMON_ENV
    "ROCM_PATH=${ROCFUZZ_THEROCK_ROOT}"
    "LD_LIBRARY_PATH=${ROCFUZZ_RUNTIME_LIBRARY_PATH_ENV}:$ENV{LD_LIBRARY_PATH}")
set(ROCFUZZ_ROCBLAS_ENV
    ${ROCFUZZ_COMMON_ENV}
    "ROCBLAS_USE_HIPBLASLT=0")

if(EXISTS "${ROCFUZZ_THEROCK_LIBRARIES_ROOT}/lib/rocblas/library")
    list(APPEND ROCFUZZ_COMMON_ENV
        "ROCBLAS_TENSILE_LIBPATH=${ROCFUZZ_THEROCK_LIBRARIES_ROOT}/lib/rocblas/library")
    list(APPEND ROCFUZZ_ROCBLAS_ENV
        "ROCBLAS_TENSILE_LIBPATH=${ROCFUZZ_THEROCK_LIBRARIES_ROOT}/lib/rocblas/library")
endif()
if(EXISTS "${ROCFUZZ_THEROCK_LIBRARIES_ROOT}/lib/hipblaslt/library")
    list(APPEND ROCFUZZ_COMMON_ENV
        "HIPBLASLT_TENSILE_LIBPATH=${ROCFUZZ_THEROCK_LIBRARIES_ROOT}/lib/hipblaslt/library")
    list(APPEND ROCFUZZ_ROCBLAS_ENV
        "HIPBLASLT_TENSILE_LIBPATH=${ROCFUZZ_THEROCK_LIBRARIES_ROOT}/lib/hipblaslt/library")
endif()

function(rocfuzz_configure_example target)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_definitions(${target} PRIVATE __HIP_PLATFORM_AMD__)
    target_include_directories(${target} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
    set_target_properties(${target} PROPERTIES
        BUILD_RPATH "${ROCFUZZ_RUNTIME_RPATH}")
endfunction()

function(rocfuzz_add_device_showmap_check target_name binary_target)
    set(options)
    set(one_value_args SEED_A SEED_B)
    set(multi_value_args ENV)
    cmake_parse_arguments(SHOWMAP
        "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
    if(NOT SHOWMAP_SEED_A OR NOT SHOWMAP_SEED_B)
        message(FATAL_ERROR
            "rocfuzz_add_device_showmap_check requires SEED_A and SEED_B")
    endif()

    set(seed_a ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.a.seed)
    set(seed_b ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.b.seed)
    set(showmap_a ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.a.map)
    set(showmap_b ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.b.map)
    set(device_a ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.a.device)
    set(device_b ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.b.device)
    set(report_a ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.a.jsonl)
    set(report_b ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.b.jsonl)

    file(WRITE ${seed_a} "${SHOWMAP_SEED_A}")
    file(WRITE ${seed_b} "${SHOWMAP_SEED_B}")

    add_custom_target(${target_name}
        COMMAND ${CMAKE_COMMAND} -E rm -f
                ${showmap_a} ${showmap_b}
                ${device_a} ${device_b}
                ${report_a} ${report_b}
        COMMAND ${CMAKE_COMMAND} -E env
                ${SHOWMAP_ENV}
                AFL_SKIP_BIN_CHECK=1
                AFL_NO_FORKSRV=1
                AFL_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
                ROCJITSU_AFL_PATCH_REPORT=${report_a}
                ${ROCFUZZ_AFL_SHOWMAP_EXECUTABLE}
                -m none -t 30000
                -o ${showmap_a}
                --
                $<TARGET_FILE:${binary_target}>
                ${seed_a}
        COMMAND ${CMAKE_COMMAND} -E env
                ${SHOWMAP_ENV}
                AFL_SKIP_BIN_CHECK=1
                AFL_NO_FORKSRV=1
                AFL_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
                ROCJITSU_AFL_PATCH_REPORT=${report_b}
                ${ROCFUZZ_AFL_SHOWMAP_EXECUTABLE}
                -m none -t 30000
                -o ${showmap_b}
                --
                $<TARGET_FILE:${binary_target}>
                ${seed_b}
        COMMAND /bin/sh -c
                "awk -F: '$1 + 0 >= 32769 { print }' '${showmap_a}' > '${device_a}'"
        COMMAND /bin/sh -c
                "awk -F: '$1 + 0 >= 32769 { print }' '${showmap_b}' > '${device_b}'"
        COMMAND /bin/sh -c
                "test -s '${device_a}' && test -s '${device_b}' || { echo 'missing device showmap entries; inspect ${report_a} and ${report_b}' >&2; exit 1; }"
        COMMAND /bin/sh -c
                "! cmp -s '${device_a}' '${device_b}' || { echo 'device showmaps are identical; inspect ${report_a} and ${report_b}' >&2; exit 1; }"
        DEPENDS
                ${binary_target}
                ${ROCFUZZ_AFL_PRELOAD_DEPENDS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Checking AFL-visible device coverage changes for ${binary_target}"
        VERBATIM)
endfunction()

function(rocfuzz_add_patch_report_check target_name report_path producer_target)
    set(options)
    set(one_value_args)
    set(multi_value_args ARGS)
    cmake_parse_arguments(REPORT_CHECK
        "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    add_custom_target(${target_name}
        COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                ${ROCFUZZ_CHECK_PATCH_REPORT_EVENTS}
                ${report_path}
                ${REPORT_CHECK_ARGS}
        DEPENDS
                ${producer_target}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Checking rocfuzz patch-report policy for ${producer_target}"
        VERBATIM)
endfunction()

add_executable(rocfuzz_example_hipblaslt_transform
    src/hipblaslt_transform_fuzz.cpp)
rocfuzz_configure_example(rocfuzz_example_hipblaslt_transform)
target_link_libraries(rocfuzz_example_hipblaslt_transform PRIVATE
    hip::host
    roc::hipblaslt)

add_executable(rocfuzz_example_hipblaslt_transform_persistent
    src/hipblaslt_transform_fuzz.cpp)
rocfuzz_configure_example(rocfuzz_example_hipblaslt_transform_persistent)
target_compile_definitions(rocfuzz_example_hipblaslt_transform_persistent PRIVATE
    ROCFUZZ_EXAMPLE_PERSISTENT=1)
target_link_libraries(rocfuzz_example_hipblaslt_transform_persistent PRIVATE
    hip::host
    roc::hipblaslt
    ${CMAKE_DL_LIBS})

add_executable(rocfuzz_example_rocblas_saxpy
    src/rocblas_saxpy_fuzz.cpp)
rocfuzz_configure_example(rocfuzz_example_rocblas_saxpy)
target_link_libraries(rocfuzz_example_rocblas_saxpy PRIVATE
    hip::host
    roc::rocblas)

add_executable(rocfuzz_example_rocblas_saxpy_persistent
    src/rocblas_saxpy_fuzz.cpp)
rocfuzz_configure_example(rocfuzz_example_rocblas_saxpy_persistent)
target_compile_definitions(rocfuzz_example_rocblas_saxpy_persistent PRIVATE
    ROCFUZZ_EXAMPLE_PERSISTENT=1)
target_link_libraries(rocfuzz_example_rocblas_saxpy_persistent PRIVATE
    hip::host
    roc::rocblas
    ${CMAKE_DL_LIBS})

add_executable(rocfuzz_example_rocblas_sgemm
    src/rocblas_sgemm_fuzz.cpp)
rocfuzz_configure_example(rocfuzz_example_rocblas_sgemm)
target_link_libraries(rocfuzz_example_rocblas_sgemm PRIVATE
    hip::host
    roc::rocblas)

add_executable(rocfuzz_example_rocblas_sgemm_persistent
    src/rocblas_sgemm_fuzz.cpp)
rocfuzz_configure_example(rocfuzz_example_rocblas_sgemm_persistent)
target_compile_definitions(rocfuzz_example_rocblas_sgemm_persistent PRIVATE
    ROCFUZZ_EXAMPLE_PERSISTENT=1)
target_link_libraries(rocfuzz_example_rocblas_sgemm_persistent PRIVATE
    hip::host
    roc::rocblas
    ${CMAKE_DL_LIBS})

add_executable(rocfuzz_example_rocfft_c2c
    src/rocfft_c2c_fuzz.cpp)
rocfuzz_configure_example(rocfuzz_example_rocfft_c2c)
target_link_libraries(rocfuzz_example_rocfft_c2c PRIVATE
    hip::host
    roc::rocfft)

add_executable(rocfuzz_example_rocfft_c2c_persistent
    src/rocfft_c2c_fuzz.cpp)
rocfuzz_configure_example(rocfuzz_example_rocfft_c2c_persistent)
target_compile_definitions(rocfuzz_example_rocfft_c2c_persistent PRIVATE
    ROCFUZZ_EXAMPLE_PERSISTENT=1)
target_link_libraries(rocfuzz_example_rocfft_c2c_persistent PRIVATE
    hip::host
    roc::rocfft
    ${CMAKE_DL_LIBS})

add_executable(rocfuzz_example_rocrand_uniform
    src/rocrand_uniform_fuzz.cpp)
rocfuzz_configure_example(rocfuzz_example_rocrand_uniform)
target_link_libraries(rocfuzz_example_rocrand_uniform PRIVATE
    hip::host
    roc::rocrand)

add_executable(rocfuzz_example_rocrand_uniform_persistent
    src/rocrand_uniform_fuzz.cpp)
rocfuzz_configure_example(rocfuzz_example_rocrand_uniform_persistent)
target_compile_definitions(rocfuzz_example_rocrand_uniform_persistent PRIVATE
    ROCFUZZ_EXAMPLE_PERSISTENT=1)
target_link_libraries(rocfuzz_example_rocrand_uniform_persistent PRIVATE
    hip::host
    roc::rocrand
    ${CMAKE_DL_LIBS})

add_executable(rocfuzz_example_rocsparse_spmv
    src/rocsparse_spmv_fuzz.cpp)
rocfuzz_configure_example(rocfuzz_example_rocsparse_spmv)
target_link_libraries(rocfuzz_example_rocsparse_spmv PRIVATE
    hip::host
    roc::rocsparse)

add_executable(rocfuzz_example_rocsparse_spmv_persistent
    src/rocsparse_spmv_fuzz.cpp)
rocfuzz_configure_example(rocfuzz_example_rocsparse_spmv_persistent)
target_compile_definitions(rocfuzz_example_rocsparse_spmv_persistent PRIVATE
    ROCFUZZ_EXAMPLE_PERSISTENT=1
    ROCFUZZ_EXAMPLE_AFL_LOOP_COUNT=1)
target_link_libraries(rocfuzz_example_rocsparse_spmv_persistent PRIVATE
    hip::host
    roc::rocsparse
    ${CMAKE_DL_LIBS})

add_executable(rocfuzz_example_rocsolver_getrf
    src/rocsolver_getrf_fuzz.cpp)
rocfuzz_configure_example(rocfuzz_example_rocsolver_getrf)
target_link_libraries(rocfuzz_example_rocsolver_getrf PRIVATE
    hip::host
    roc::rocsolver)

add_executable(rocfuzz_example_rocsolver_getrf_persistent
    src/rocsolver_getrf_fuzz.cpp)
rocfuzz_configure_example(rocfuzz_example_rocsolver_getrf_persistent)
target_compile_definitions(rocfuzz_example_rocsolver_getrf_persistent PRIVATE
    ROCFUZZ_EXAMPLE_PERSISTENT=1)
target_link_libraries(rocfuzz_example_rocsolver_getrf_persistent PRIVATE
    hip::host
    roc::rocsolver
    ${CMAKE_DL_LIBS})

add_executable(rocfuzz_example_miopen_activation
    src/miopen_activation_fuzz.cpp)
rocfuzz_configure_example(rocfuzz_example_miopen_activation)
target_link_libraries(rocfuzz_example_miopen_activation PRIVATE
    hip::host
    MIOpen)

add_executable(rocfuzz_example_miopen_activation_persistent
    src/miopen_activation_fuzz.cpp)
rocfuzz_configure_example(rocfuzz_example_miopen_activation_persistent)
target_compile_definitions(rocfuzz_example_miopen_activation_persistent PRIVATE
    ROCFUZZ_EXAMPLE_PERSISTENT=1)
target_link_libraries(rocfuzz_example_miopen_activation_persistent PRIVATE
    hip::host
    MIOpen
    ${CMAKE_DL_LIBS})

add_custom_target(rocfuzz_example_rocfft_c2c_smoke
    COMMAND ${CMAKE_COMMAND} -E env
            ${ROCFUZZ_COMMON_ENV}
            ROCJITSU_AFL_VERBOSE=1
            ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
            LD_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
            $<TARGET_FILE:rocfuzz_example_rocfft_c2c>
            ${CMAKE_CURRENT_SOURCE_DIR}/seeds/rocfft_c2c/seed0
    DEPENDS
            rocfuzz_example_rocfft_c2c
            ${ROCFUZZ_AFL_PRELOAD_DEPENDS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running rocFFT C2C non-GEMM branch-coverage smoke target under rocjitsu AFL DBI"
    VERBATIM)

add_custom_target(rocfuzz_example_rocfft_c2c_persistent_smoke
    COMMAND ${CMAKE_COMMAND} -E env
            ${ROCFUZZ_COMMON_ENV}
            ROCJITSU_AFL_VERBOSE=1
            ROCJITSU_AFL_PERSISTENT=1
            ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
            ROCJITSU_AFL_REQUIRE_PERSISTENT_HOOKS=1
            LD_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
            $<TARGET_FILE:rocfuzz_example_rocfft_c2c_persistent>
            ${CMAKE_CURRENT_SOURCE_DIR}/seeds/rocfft_c2c/seed0
    DEPENDS
            rocfuzz_example_rocfft_c2c_persistent
            ${ROCFUZZ_AFL_PRELOAD_DEPENDS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running persistent rocFFT C2C non-GEMM branch-coverage smoke target under rocjitsu AFL DBI"
    VERBATIM)

set(ROCFUZZ_ROCRAND_UNIFORM_REPORT
    ${CMAKE_CURRENT_BINARY_DIR}/rocrand_uniform_report.jsonl)
set(ROCFUZZ_ROCRAND_UNIFORM_PERSISTENT_REPORT
    ${CMAKE_CURRENT_BINARY_DIR}/rocrand_uniform_persistent_report.jsonl)
set(ROCFUZZ_ROCSPARSE_SPMV_REPORT
    ${CMAKE_CURRENT_BINARY_DIR}/rocsparse_spmv_report.jsonl)
set(ROCFUZZ_ROCSOLVER_GETRF_REPORT
    ${CMAKE_CURRENT_BINARY_DIR}/rocsolver_getrf_report.jsonl)
set(ROCFUZZ_MIOPEN_ACTIVATION_REPORT
    ${CMAKE_CURRENT_BINARY_DIR}/miopen_activation_report.jsonl)

add_custom_target(rocfuzz_example_rocrand_uniform_smoke
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_ROCRAND_UNIFORM_REPORT}
    COMMAND ${CMAKE_COMMAND} -E env
            ${ROCFUZZ_COMMON_ENV}
            ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_ROCRAND_UNIFORM_REPORT}
            LD_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
            $<TARGET_FILE:rocfuzz_example_rocrand_uniform>
            ${CMAKE_CURRENT_SOURCE_DIR}/seeds/rocrand_uniform/seed0
    DEPENDS
            rocfuzz_example_rocrand_uniform
            ${ROCFUZZ_AFL_PRELOAD_DEPENDS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running rocRAND uniform branch-coverage report target under rocjitsu AFL DBI"
    VERBATIM)

add_custom_target(rocfuzz_example_rocrand_uniform_persistent_smoke
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_ROCRAND_UNIFORM_PERSISTENT_REPORT}
    COMMAND ${CMAKE_COMMAND} -E env
            ${ROCFUZZ_COMMON_ENV}
            ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_ROCRAND_UNIFORM_PERSISTENT_REPORT}
            ROCJITSU_AFL_PERSISTENT=1
            ROCJITSU_AFL_REQUIRE_PERSISTENT_HOOKS=1
            LD_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
            $<TARGET_FILE:rocfuzz_example_rocrand_uniform_persistent>
            ${CMAKE_CURRENT_SOURCE_DIR}/seeds/rocrand_uniform/seed0
    DEPENDS
            rocfuzz_example_rocrand_uniform_persistent
            ${ROCFUZZ_AFL_PRELOAD_DEPENDS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running persistent rocRAND uniform branch-coverage report target under rocjitsu AFL DBI"
    VERBATIM)

add_custom_target(rocfuzz_example_rocblas_saxpy_smoke
    COMMAND ${CMAKE_COMMAND} -E env
            ${ROCFUZZ_ROCBLAS_ENV}
            ROCJITSU_AFL_VERBOSE=1
            LD_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
            $<TARGET_FILE:rocfuzz_example_rocblas_saxpy>
            ${CMAKE_CURRENT_SOURCE_DIR}/seeds/rocblas_saxpy/seed0
    DEPENDS
            rocfuzz_example_rocblas_saxpy
            ${ROCFUZZ_AFL_PRELOAD_DEPENDS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running rocBLAS SAXPY non-GEMM exploratory target under rocjitsu AFL DBI"
    VERBATIM)

add_custom_target(rocfuzz_example_rocblas_saxpy_persistent_smoke
    COMMAND ${CMAKE_COMMAND} -E env
            ${ROCFUZZ_ROCBLAS_ENV}
            ROCJITSU_AFL_VERBOSE=1
            ROCJITSU_AFL_PERSISTENT=1
            ROCJITSU_AFL_REQUIRE_PERSISTENT_HOOKS=1
            LD_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
            $<TARGET_FILE:rocfuzz_example_rocblas_saxpy_persistent>
            ${CMAKE_CURRENT_SOURCE_DIR}/seeds/rocblas_saxpy/seed0
    DEPENDS
            rocfuzz_example_rocblas_saxpy_persistent
            ${ROCFUZZ_AFL_PRELOAD_DEPENDS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running persistent rocBLAS SAXPY non-GEMM exploratory target under rocjitsu AFL DBI"
    VERBATIM)

add_custom_target(rocfuzz_example_rocblas_sgemm_smoke
    COMMAND ${CMAKE_COMMAND} -E env
            ${ROCFUZZ_ROCBLAS_ENV}
            ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
            LD_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
            $<TARGET_FILE:rocfuzz_example_rocblas_sgemm>
            ${CMAKE_CURRENT_SOURCE_DIR}/seeds/rocblas_sgemm/seed0
    DEPENDS
            rocfuzz_example_rocblas_sgemm
            ${ROCFUZZ_AFL_PRELOAD_DEPENDS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running rocBLAS SGEMM branch-coverage smoke target under rocjitsu AFL DBI"
    VERBATIM)

add_custom_target(rocfuzz_example_rocblas_sgemm_persistent_smoke
    COMMAND ${CMAKE_COMMAND} -E env
            ${ROCFUZZ_ROCBLAS_ENV}
            ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
            ROCJITSU_AFL_PERSISTENT=1
            ROCJITSU_AFL_REQUIRE_PERSISTENT_HOOKS=1
            LD_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
            $<TARGET_FILE:rocfuzz_example_rocblas_sgemm_persistent>
            ${CMAKE_CURRENT_SOURCE_DIR}/seeds/rocblas_sgemm/seed0
    DEPENDS
            rocfuzz_example_rocblas_sgemm_persistent
            ${ROCFUZZ_AFL_PRELOAD_DEPENDS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running persistent rocBLAS SGEMM branch-coverage smoke target under rocjitsu AFL DBI"
    VERBATIM)

add_custom_target(rocfuzz_example_rocsparse_spmv_smoke
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_ROCSPARSE_SPMV_REPORT}
    COMMAND ${CMAKE_COMMAND} -E env
            ${ROCFUZZ_COMMON_ENV}
            ROCJITSU_AFL_VERBOSE=1
            ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
            ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_ROCSPARSE_SPMV_REPORT}
            LD_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
            $<TARGET_FILE:rocfuzz_example_rocsparse_spmv>
            ${CMAKE_CURRENT_SOURCE_DIR}/seeds/rocsparse_spmv/seed0
    DEPENDS
            rocfuzz_example_rocsparse_spmv
            ${ROCFUZZ_AFL_PRELOAD_DEPENDS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running rocSPARSE SpMV branch-coverage smoke target under rocjitsu AFL DBI"
    VERBATIM)

add_custom_target(rocfuzz_example_rocsparse_spmv_persistent_smoke
    COMMAND ${CMAKE_COMMAND} -E env
            ${ROCFUZZ_COMMON_ENV}
            ROCJITSU_AFL_VERBOSE=1
            ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
            ROCJITSU_AFL_PERSISTENT=1
            ROCJITSU_AFL_REQUIRE_PERSISTENT_HOOKS=1
            LD_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
            $<TARGET_FILE:rocfuzz_example_rocsparse_spmv_persistent>
            ${CMAKE_CURRENT_SOURCE_DIR}/seeds/rocsparse_spmv/seed0
    DEPENDS
            rocfuzz_example_rocsparse_spmv_persistent
            ${ROCFUZZ_AFL_PRELOAD_DEPENDS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running persistent rocSPARSE SpMV branch-coverage smoke target under rocjitsu AFL DBI"
    VERBATIM)

add_custom_target(rocfuzz_example_rocsolver_getrf_smoke
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_ROCSOLVER_GETRF_REPORT}
    COMMAND ${CMAKE_COMMAND} -E env
            ${ROCFUZZ_ROCBLAS_ENV}
            ROCJITSU_AFL_VERBOSE=1
            ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
            ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_ROCSOLVER_GETRF_REPORT}
            LD_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
            $<TARGET_FILE:rocfuzz_example_rocsolver_getrf>
            ${CMAKE_CURRENT_SOURCE_DIR}/seeds/rocsolver_getrf/seed0
    DEPENDS
            rocfuzz_example_rocsolver_getrf
            ${ROCFUZZ_AFL_PRELOAD_DEPENDS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running rocSOLVER GETRF branch-coverage smoke target under rocjitsu AFL DBI"
    VERBATIM)

add_custom_target(rocfuzz_example_rocsolver_getrf_persistent_smoke
    COMMAND ${CMAKE_COMMAND} -E env
            ${ROCFUZZ_ROCBLAS_ENV}
            ROCJITSU_AFL_VERBOSE=1
            ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
            ROCJITSU_AFL_PERSISTENT=1
            ROCJITSU_AFL_REQUIRE_PERSISTENT_HOOKS=1
            LD_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
            $<TARGET_FILE:rocfuzz_example_rocsolver_getrf_persistent>
            ${CMAKE_CURRENT_SOURCE_DIR}/seeds/rocsolver_getrf/seed0
    DEPENDS
            rocfuzz_example_rocsolver_getrf_persistent
            ${ROCFUZZ_AFL_PRELOAD_DEPENDS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running persistent rocSOLVER GETRF branch-coverage smoke target under rocjitsu AFL DBI"
    VERBATIM)

add_custom_target(rocfuzz_example_miopen_activation_smoke
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_MIOPEN_ACTIVATION_REPORT}
    COMMAND ${CMAKE_COMMAND} -E env
            ${ROCFUZZ_COMMON_ENV}
            ROCJITSU_AFL_VERBOSE=1
            ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
            ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_MIOPEN_ACTIVATION_REPORT}
            LD_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
            $<TARGET_FILE:rocfuzz_example_miopen_activation>
            ${CMAKE_CURRENT_SOURCE_DIR}/seeds/miopen_activation/seed0
    DEPENDS
            rocfuzz_example_miopen_activation
            ${ROCFUZZ_AFL_PRELOAD_DEPENDS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running MIOpen activation branch-coverage smoke target under rocjitsu AFL DBI"
    VERBATIM)

add_custom_target(rocfuzz_example_miopen_activation_persistent_smoke
    COMMAND ${CMAKE_COMMAND} -E env
            ${ROCFUZZ_COMMON_ENV}
            ROCJITSU_AFL_VERBOSE=1
            ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
            ROCJITSU_AFL_PERSISTENT=1
            ROCJITSU_AFL_REQUIRE_PERSISTENT_HOOKS=1
            LD_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
            $<TARGET_FILE:rocfuzz_example_miopen_activation_persistent>
            ${CMAKE_CURRENT_SOURCE_DIR}/seeds/miopen_activation/seed0
    DEPENDS
            rocfuzz_example_miopen_activation_persistent
            ${ROCFUZZ_AFL_PRELOAD_DEPENDS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running persistent MIOpen activation branch-coverage smoke target under rocjitsu AFL DBI"
    VERBATIM)

set(ROCFUZZ_EXAMPLE_SMOKE_POLICY_TARGETS
    rocfuzz_example_rocfft_c2c_smoke
    rocfuzz_example_rocfft_c2c_persistent_smoke
    rocfuzz_example_rocblas_sgemm_smoke
    rocfuzz_example_rocblas_sgemm_persistent_smoke
    rocfuzz_example_rocrand_uniform_persistent_smoke
    rocfuzz_example_rocsparse_spmv_smoke
    rocfuzz_example_rocsparse_spmv_persistent_smoke
    rocfuzz_example_rocsolver_getrf_smoke
    rocfuzz_example_rocsolver_getrf_persistent_smoke
    rocfuzz_example_miopen_activation_smoke
    rocfuzz_example_miopen_activation_persistent_smoke)

if(ROCFUZZ_AFL_SHOWMAP_EXECUTABLE)
    # Keep both rocRAND showmap seeds on XORWOW launched kernels. ASCII '2'
    # currently selects the MRG32K3A path, whose launched kernel has no safe
    # patchable sites and makes afl-showmap report no instrumentation. ASCII
    # '5' still selects XORWOW but changes seed/count enough to alter counters.
    string(REPEAT "1" 32 ROCFUZZ_ROCRAND_SHOWMAP_SEED_A)
    string(REPEAT "5" 32 ROCFUZZ_ROCRAND_SHOWMAP_SEED_B)
    rocfuzz_add_device_showmap_check(rocfuzz_example_rocrand_uniform_showmap
        rocfuzz_example_rocrand_uniform
        ENV ${ROCFUZZ_COMMON_ENV}
        SEED_A "${ROCFUZZ_ROCRAND_SHOWMAP_SEED_A}"
        SEED_B "${ROCFUZZ_ROCRAND_SHOWMAP_SEED_B}")

    string(REPEAT "1" 80 ROCFUZZ_ROCSPARSE_SHOWMAP_SEED_A)
    string(REPEAT "3" 80 ROCFUZZ_ROCSPARSE_SHOWMAP_SEED_B)
    rocfuzz_add_device_showmap_check(rocfuzz_example_rocsparse_spmv_showmap
        rocfuzz_example_rocsparse_spmv
        ENV ${ROCFUZZ_COMMON_ENV}
        SEED_A "${ROCFUZZ_ROCSPARSE_SHOWMAP_SEED_A}"
        SEED_B "${ROCFUZZ_ROCSPARSE_SHOWMAP_SEED_B}")

    add_custom_target(rocfuzz_example_device_showmap_checks
        DEPENDS
            rocfuzz_example_rocrand_uniform_showmap
            rocfuzz_example_rocsparse_spmv_showmap)
    list(APPEND ROCFUZZ_EXAMPLE_SMOKE_POLICY_TARGETS
        rocfuzz_example_device_showmap_checks)
else()
    message(STATUS "rocfuzz example showmap checks disabled (afl-showmap not found)")
endif()

if(ROCFUZZ_PYTHON3_EXECUTABLE)
    set(ROCFUZZ_HIGH_EDGE_REPORT_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/high-edge-reports)
    set(ROCFUZZ_ROCBLAS_SGEMM_HIGH_EDGE_REPORT
        ${ROCFUZZ_HIGH_EDGE_REPORT_DIR}/rocblas_sgemm_high_edge.jsonl)
    set(ROCFUZZ_ROCBLAS_SGEMM_DEFAULT_HYBRID_REPORT
        ${ROCFUZZ_HIGH_EDGE_REPORT_DIR}/rocblas_sgemm_default_hybrid_high_edge.jsonl)
    set(ROCFUZZ_ROCBLAS_SGEMM_VOPD_FRESH_GROWTH_REPORT
        ${ROCFUZZ_HIGH_EDGE_REPORT_DIR}/rocblas_sgemm_vopd_fresh_growth.jsonl)
    set(ROCFUZZ_ROCFFT_C2C_HIGH_EDGE_REPORT
        ${ROCFUZZ_HIGH_EDGE_REPORT_DIR}/rocfft_c2c_high_edge.jsonl)
    set(ROCFUZZ_MIOPEN_ACTIVATION_HIGH_EDGE_REPORT
        ${ROCFUZZ_HIGH_EDGE_REPORT_DIR}/miopen_activation_high_edge.jsonl)
    set(ROCFUZZ_CHECK_PATCH_REPORT_EVENTS
        ${CMAKE_CURRENT_SOURCE_DIR}/../afl-dbi/tools/check_patch_report_events.py)

    rocfuzz_add_patch_report_check(rocfuzz_example_rocrand_uniform_report_check
        ${ROCFUZZ_ROCRAND_UNIFORM_REPORT}
        rocfuzz_example_rocrand_uniform_smoke
        ARGS
                --require-event patch_device_elf
                --require-event device_edge_delta
                --require-field patch_device_elf success true
                --require-field patch_device_elf reason patched
                --require-field patch_device_elf fixed_edge_slot_scope stable-site-hash
                --require-field patch_device_elf fixed_slot_collisions 0
                --require-list-field patch_device_elf kernel_summaries coverage_strategy self-contained-fixed-branch
                --require-sum-field patch_device_elf edge_sites_patched 2
                --require-sum-field patch_device_elf fixed_edge_sites 2
                --require-min-field device_edge_delta edge_slot_delta_count 1)

    rocfuzz_add_patch_report_check(rocfuzz_example_rocsparse_spmv_report_check
        ${ROCFUZZ_ROCSPARSE_SPMV_REPORT}
        rocfuzz_example_rocsparse_spmv_smoke
        ARGS
                --require-event patch_device_elf
                --require-event device_edge_delta
                --require-field patch_device_elf success true
                --require-field patch_device_elf reason patched
                --require-list-field patch_device_elf kernel_summaries coverage_strategy self-contained-fixed-branch
                --require-sum-field patch_device_elf edge_sites_patched 1
                --require-sum-field patch_device_elf fixed_edge_sites 1
                --require-min-field device_edge_delta edge_slot_delta_count 1)

    rocfuzz_add_patch_report_check(rocfuzz_example_rocsolver_getrf_report_check
        ${ROCFUZZ_ROCSOLVER_GETRF_REPORT}
        rocfuzz_example_rocsolver_getrf_smoke
        ARGS
                --require-event patch_device_elf
                --require-event device_edge_delta
                --require-field patch_device_elf success true
                --require-field patch_device_elf reason patched
                --require-list-field patch_device_elf kernel_summaries coverage_strategy self-contained-fixed-branch
                --require-sum-field patch_device_elf edge_sites_patched 1
                --require-sum-field patch_device_elf fixed_edge_sites 1
                --require-min-field device_edge_delta edge_slot_delta_count 1)

    rocfuzz_add_patch_report_check(rocfuzz_example_miopen_activation_report_check
        ${ROCFUZZ_MIOPEN_ACTIVATION_REPORT}
        rocfuzz_example_miopen_activation_smoke
        ARGS
                --require-event patch_device_elf
                --require-event device_edge_delta
                --require-field patch_device_elf success true
                --require-field patch_device_elf reason patched
                --require-field patch_device_elf coverage_strategy self-contained-fixed-branch
                --require-field patch_device_elf coverage_strategy_reason classifier-entry-redirection-unsafe
                --require-field patch_device_elf skip_entry_probe true
                --require-field patch_device_elf branch_edge_slot_policy fixed-counter
                --require-field patch_device_elf entries_patched 0
                --require-list-item patch_device_elf kernel_summaries
                    kernel=MIOpenActiveFwdLite
                    self_contained_probe=true
                    coverage_strategy=self-contained-fixed-branch
                    fixed_edge_sites=1
                --require-list-item patch_device_elf descriptor_resources
                    kernel=MIOpenActiveFwdLite
                    wave32=true
                    resource_fields_changed=false
                --require-sum-field patch_device_elf edge_sites_patched 4
                --require-sum-field patch_device_elf fixed_edge_sites 4
                --require-min-field device_edge_delta edge_slot_delta_count 1)

    add_custom_target(rocfuzz_example_high_edge_report_smoke
        COMMAND ${CMAKE_COMMAND} -E make_directory ${ROCFUZZ_HIGH_EDGE_REPORT_DIR}

        COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_ROCBLAS_SGEMM_HIGH_EDGE_REPORT}
        COMMAND ${CMAKE_COMMAND} -E env
                ${ROCFUZZ_ROCBLAS_ENV}
                ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE=1
                ROCJITSU_AFL_DEBUG_EDGE_LIMIT=0
                ROCJITSU_AFL_DEBUG_BRANCH_EDGE_LIMIT=64
                ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOTS=1
                ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY=fixed
                ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_ROCBLAS_SGEMM_HIGH_EDGE_REPORT}
                LD_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
                $<TARGET_FILE:rocfuzz_example_rocblas_sgemm>
                ${CMAKE_CURRENT_SOURCE_DIR}/seeds/rocblas_sgemm/seed0
        COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                ${ROCFUZZ_CHECK_PATCH_REPORT_EVENTS}
                ${ROCFUZZ_ROCBLAS_SGEMM_HIGH_EDGE_REPORT}
                --require-event patch_device_elf
                --require-event device_edge_delta
                --require-field patch_device_elf success true
                --require-field patch_device_elf branch_edge_slot_policy fixed-counter
                --require-field patch_device_elf fixed_edge_slot_scope stable-site-hash
                --require-field patch_device_elf fixed_edge_slot_collision_policy stable-site-hash-afl-bitmap-collisions-accepted
                --require-field patch_device_elf fixed_slot_exhaustions 0
                --require-field patch_device_elf branch_edge_site_limit 64
                --require-field patch_device_elf branch_edge_site_limit_auto false
                --require-field patch_device_elf edge_site_limit 0
                --require-min-field patch_device_elf edge_sites_patched 64
                --require-min-field device_edge_delta edge_slot_delta_count 1

        COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_ROCFFT_C2C_HIGH_EDGE_REPORT}
        COMMAND ${CMAKE_COMMAND} -E env
                ${ROCFUZZ_COMMON_ENV}
                ROCJITSU_AFL_DEBUG_EDGE_LIMIT=64
                ROCJITSU_AFL_DEBUG_BRANCH_EDGE_LIMIT=64
                ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_ROCFFT_C2C_HIGH_EDGE_REPORT}
                LD_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
                $<TARGET_FILE:rocfuzz_example_rocfft_c2c>
                ${CMAKE_CURRENT_SOURCE_DIR}/seeds/rocfft_c2c/seed0
        COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                ${ROCFUZZ_CHECK_PATCH_REPORT_EVENTS}
                ${ROCFUZZ_ROCFFT_C2C_HIGH_EDGE_REPORT}
                --require-event patch_device_elf
                --require-event device_edge_delta
                --require-field patch_device_elf success true
                --require-field patch_device_elf reason patched
                --require-field patch_device_elf success false
                --require-field patch_device_elf reason no_patchable_sites
                --require-field patch_device_elf coverage_strategy self-contained-previous-bb-branch
                --require-field patch_device_elf coverage_signal previous-bb-edges
                --require-field patch_device_elf edge_site_limit 64
                --require-field patch_device_elf branch_edge_site_limit 64
                --require-field patch_device_elf branch_edge_site_limit_auto false
                --require-list-item patch_device_elf skipped_kernels
                    kernel=twiddle_gen_radices_sp
                    reason=prior-invalid-dispatch-kernel
                --require-list-field patch_device_elf kernel_summaries coverage_strategy self-contained-previous-bb-branch
                --require-list-field patch_device_elf kernel_summaries.skip_reason_counts
                    reason "EXEC-conditioned previous-BB branch requires fixed-counter fallback budget"
                --require-sum-field patch_device_elf edge_sites_patched 8
                --require-sum-field patch_device_elf hashed_edge_sites 8
                --require-min-field device_edge_delta edge_slot_delta_count 1

        COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_MIOPEN_ACTIVATION_HIGH_EDGE_REPORT}
        COMMAND ${CMAKE_COMMAND} -E env
                ${ROCFUZZ_COMMON_ENV}
                ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE=1
                ROCJITSU_AFL_DEBUG_EDGE_LIMIT=0
                ROCJITSU_AFL_DEBUG_BRANCH_EDGE_LIMIT=64
                ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOTS=1
                ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY=fixed
                ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_MIOPEN_ACTIVATION_HIGH_EDGE_REPORT}
                LD_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
                $<TARGET_FILE:rocfuzz_example_miopen_activation>
                ${CMAKE_CURRENT_SOURCE_DIR}/seeds/miopen_activation/seed0
        COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                ${ROCFUZZ_CHECK_PATCH_REPORT_EVENTS}
                ${ROCFUZZ_MIOPEN_ACTIVATION_HIGH_EDGE_REPORT}
                --require-event patch_device_elf
                --require-event device_edge_delta
                --require-field patch_device_elf success true
                --require-field patch_device_elf branch_edge_slot_policy fixed-counter
                --require-field patch_device_elf fixed_edge_slot_scope stable-site-hash
                --require-field patch_device_elf fixed_slot_collisions 0
                --require-field patch_device_elf branch_edge_site_limit 64
                --require-field patch_device_elf branch_edge_site_limit_auto false
                --require-field patch_device_elf edge_site_limit 0
                --require-min-field patch_device_elf edge_sites_patched 4
                --require-min-field device_edge_delta edge_slot_delta_count 1
        DEPENDS
                rocfuzz_example_rocblas_sgemm
                rocfuzz_example_rocfft_c2c
                rocfuzz_example_miopen_activation
                ${ROCFUZZ_AFL_PRELOAD_DEPENDS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running high edge-limit report smokes for rocBLAS, rocFFT, and MIOpen examples"
        VERBATIM)

    add_custom_target(rocfuzz_example_rocblas_sgemm_default_hybrid_report_smoke
        COMMAND ${CMAKE_COMMAND} -E make_directory ${ROCFUZZ_HIGH_EDGE_REPORT_DIR}
        COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_ROCBLAS_SGEMM_DEFAULT_HYBRID_REPORT}
        COMMAND ${CMAKE_COMMAND} -E env
                ${ROCFUZZ_ROCBLAS_ENV}
                ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_ROCBLAS_SGEMM_DEFAULT_HYBRID_REPORT}
                LD_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
                $<TARGET_FILE:rocfuzz_example_rocblas_sgemm>
                ${CMAKE_CURRENT_SOURCE_DIR}/seeds/rocblas_sgemm/seed0
        COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                ${ROCFUZZ_CHECK_PATCH_REPORT_EVENTS}
                ${ROCFUZZ_ROCBLAS_SGEMM_DEFAULT_HYBRID_REPORT}
                --require-event patch_device_elf
                --require-event device_edge_delta
                --require-field patch_device_elf success true
                --require-field patch_device_elf coverage_strategy self-contained-hybrid-previous-bb-and-fixed-branch
                --require-field patch_device_elf coverage_signal hybrid-previous-bb-and-fixed
                --require-field patch_device_elf coverage_strategy_reason entry-liveness-preflight-needs-fixed-registers
                --require-field patch_device_elf skip_entry_probe true
                --require-field patch_device_elf branch_edge_slot_policy previous-bb-hash
                --require-field patch_device_elf fixed_edge_slot_scope stable-site-hash
                --require-field patch_device_elf fixed_edge_slot_collision_policy stable-site-hash-afl-bitmap-collisions-accepted
                --require-field patch_device_elf fixed_slot_exhaustions 0
                --require-field patch_device_elf require_liveness_registers true
                --require-field patch_device_elf edge_site_limit 0
                --require-field patch_device_elf branch_edge_site_limit 100
                --require-field patch_device_elf branch_edge_site_limit_auto true
                --require-field patch_device_elf previous_bb_branch_site_limit 4294967295
                --require-field patch_device_elf previous_bb_branch_site_limit_auto true
                --require-field patch_device_elf entries_patched 0
                # The exact total depends on the rocBLAS/Tensile payload selected by
                # the local library package. Keep this broad count high, but let the
                # semantic hybrid checks below carry the signal.
                --require-sum-field patch_device_elf edge_sites_patched 500
                --require-sum-field patch_device_elf hashed_edge_sites 260
                --require-sum-field patch_device_elf fixed_edge_sites 180
                --require-sum-field patch_device_elf fixed_counter_branch_edge_safety_fallback_used 300
                --require-list-field patch_device_elf kernel_summaries coverage_strategy self-contained-hybrid-previous-bb-and-fixed-branch
                --require-list-field patch_device_elf kernel_summaries.degradation_reason_counts
                    reason "EXEC-conditioned previous-BB branch probe is not yet proven safe"
                --require-list-item-count patch_device_elf kernel_summaries 2
                    branch_edge_budget_reason=auto-previous-bb-site-derived-candidate-count
                    previous_bb_branch_site_budget_reason=auto-candidate-count
                    previous_bb_branch_edge_over_budget=0
                    previous_bb_branch_site_over_budget=0
                    previous_bb_branch_aggregate_limit_kind=none
                    fixed_counter_branch_edge_fallback_budget_reason=auto-candidate-branch-edge-count
                    skipped_branch_limit=0
                --require-min-field patch_device_elf liveness_register_kernels 1
                --require-min-field patch_device_elf liveness_probe_points 1
                --require-min-field device_edge_delta edge_slot_delta_count 1
        DEPENDS
                rocfuzz_example_rocblas_sgemm
                ${ROCFUZZ_AFL_PRELOAD_DEPENDS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running default hybrid branch-edge report smoke for rocBLAS SGEMM"
        VERBATIM)

    add_custom_target(rocfuzz_example_rocblas_sgemm_vopd_fresh_growth_report_smoke
        COMMAND ${CMAKE_COMMAND} -E make_directory ${ROCFUZZ_HIGH_EDGE_REPORT_DIR}
        COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCFUZZ_ROCBLAS_SGEMM_VOPD_FRESH_GROWTH_REPORT}
        COMMAND ${CMAKE_COMMAND} -E env
                ${ROCFUZZ_ROCBLAS_ENV}
                ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE=1
                ROCJITSU_AFL_DEBUG_EDGE_LIMIT=0
                ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOTS=1
                ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY=hashed
                ROCJITSU_AFL_DEBUG_REQUIRE_LIVENESS_REGISTERS=1
                ROCJITSU_AFL_DEBUG_ALLOW_OPAQUE_FRESH_REGISTERS=1
                ROCJITSU_AFL_DEBUG_FORCE_FRESH_SGPRS=1
                ROCJITSU_AFL_DEBUG_FORCE_FRESH_VGPRS=1
                ROCJITSU_AFL_DEBUG_DISABLE_VGPR_SCRATCH_SPILLS=1
                ROCJITSU_AFL_DEBUG_BRANCH_EDGE_LIMIT=128
                ROCJITSU_AFL_KERNEL_INCLUDE=MT128x64
                ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
                ROCJITSU_AFL_PATCH_REPORT=${ROCFUZZ_ROCBLAS_SGEMM_VOPD_FRESH_GROWTH_REPORT}
                LD_PRELOAD=${ROCFUZZ_AFL_PRELOAD}
                $<TARGET_FILE:rocfuzz_example_rocblas_sgemm>
                ${CMAKE_CURRENT_SOURCE_DIR}/seeds/rocblas_sgemm/seed0
        COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                ${ROCFUZZ_CHECK_PATCH_REPORT_EVENTS}
                ${ROCFUZZ_ROCBLAS_SGEMM_VOPD_FRESH_GROWTH_REPORT}
                --require-event patch_device_elf
                --require-event device_edge_delta
                --require-field patch_device_elf success true
                --require-field patch_device_elf reason patched
                --require-field patch_device_elf coverage_strategy self-contained-previous-bb-branch
                --require-field patch_device_elf coverage_signal previous-bb-edges
                --require-field patch_device_elf branch_edge_slot_policy previous-bb-hash
                --require-field patch_device_elf allow_opaque_fresh_registers true
                --require-field patch_device_elf require_liveness_registers true
                --require-field patch_device_elf vgpr_scratch_spills_enabled false
                --require-field patch_device_elf edge_patch_failures 0
                --require-field patch_device_elf edge_site_limit 0
                --require-field patch_device_elf branch_edge_site_limit 128
                --require-field patch_device_elf branch_edge_site_limit_auto false
                --require-sum-field patch_device_elf edge_sites_patched 128
                --require-sum-field patch_device_elf hashed_edge_sites 128
                --require-min-field patch_device_elf probe_required_sgprs 44
                --require-min-field patch_device_elf probe_required_vgprs 108
                --require-list-item-count patch_device_elf kernel_summaries 2
                    fresh_registers=true
                    fresh_register_probe_points=69
                    unmodeled_opaque_instruction_count=0
                --require-list-field patch_device_elf kernel_summaries.sampled_opaque_instructions
                    mnemonic vopd_opaque
                --require-list-field patch_device_elf descriptor_resources
                    sgpr_count_metadata_patch in-place-bytes
                --require-list-field patch_device_elf descriptor_resources
                    resource_fields_changed true
                --require-list-item-delta patch_device_elf descriptor_resources
                    patched_sgpr_count old_sgpr_count 4
                    sgpr_count_metadata_patch=in-place-bytes
                    resource_fields_changed=true
                --require-list-item-delta patch_device_elf descriptor_resources
                    patched_vgpr_count old_vgpr_count 8
                    resource_fields_changed=true
                --require-min-field device_edge_delta edge_slot_delta_count 1
        DEPENDS
                rocfuzz_example_rocblas_sgemm
                ${ROCFUZZ_AFL_PRELOAD_DEPENDS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running VOPD opaque fresh-register growth report smoke for rocBLAS SGEMM"
        VERBATIM)

    list(APPEND ROCFUZZ_EXAMPLE_SMOKE_POLICY_TARGETS
        rocfuzz_example_rocrand_uniform_report_check
        rocfuzz_example_rocsparse_spmv_report_check
        rocfuzz_example_rocsolver_getrf_report_check
        rocfuzz_example_miopen_activation_report_check
        rocfuzz_example_high_edge_report_smoke
        rocfuzz_example_rocblas_sgemm_default_hybrid_report_smoke
        rocfuzz_example_rocblas_sgemm_vopd_fresh_growth_report_smoke)
else()
    message(STATUS "rocfuzz example high edge-limit report smoke disabled (python3 not found)")
endif()

# Maintained example coverage gate: bounded smokes only, not AFL campaigns.
add_custom_target(rocfuzz_example_smoke_policy_gates
    DEPENDS
            ${ROCFUZZ_EXAMPLE_SMOKE_POLICY_TARGETS})

set(ROCFUZZ_EXAMPLE_COVERAGE_GATE_TARGETS
    rocfuzz_example_smoke_policy_gates)

if(ROCFUZZ_PYTHON3_EXECUTABLE)
    set(ROCFUZZ_EXAMPLE_COVERAGE_BASELINE_REPORT
        ${CMAKE_CURRENT_BINARY_DIR}/coverage-baseline.md)
    add_custom_target(rocfuzz_example_coverage_baseline_report
        COMMAND ${ROCFUZZ_PYTHON3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/scripts/summarize-coverage-baseline.py
                --examples-build ${CMAKE_CURRENT_BINARY_DIR}
                --output ${ROCFUZZ_EXAMPLE_COVERAGE_BASELINE_REPORT}
        DEPENDS
                rocfuzz_example_smoke_policy_gates
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating rocfuzz per-example coverage baseline report"
        VERBATIM)
    list(APPEND ROCFUZZ_EXAMPLE_COVERAGE_GATE_TARGETS
        rocfuzz_example_coverage_baseline_report)
endif()

add_custom_target(rocfuzz_example_coverage_gates
    DEPENDS
            ${ROCFUZZ_EXAMPLE_COVERAGE_GATE_TARGETS})
