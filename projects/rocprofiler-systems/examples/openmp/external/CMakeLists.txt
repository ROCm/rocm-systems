# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# ----------------------------------------------------------------------------------------#
#
# OMPVV (OpenMP Validation and Verification) submodule
#
# * Currently, GNU's libgomp cannot be used to capture traces. Only LLVM's
#   LIBOMP can be used for trace capture
# * As of implementation, amdflang uses LIBOMP 201611 (5.0), but not all
#   OpenMP 5.0 features are implemented. Only a subset of tests can be compiled.
# ----------------------------------------------------------------------------------------#

set(OMPVV_USE_OFFLOAD_TESTS FALSE)
if(ROCPROFSYS_DISABLE_EXAMPLES)
    if("openmp-vv" IN_LIST ROCPROFSYS_DISABLE_EXAMPLES)
        return()
    endif()
    if(NOT "openmp-vv-offload" IN_LIST ROCPROFSYS_DISABLE_EXAMPLES)
        set(OMPVV_USE_OFFLOAD_TESTS TRUE)
    endif()
else()
    set(OMPVV_USE_OFFLOAD_TESTS TRUE)
endif()

rocprofiler_systems_message(STATUS "Configuring OMPVV...")

# Master branch contains stable releases
rocprofiler_systems_checkout_git_submodule(
  RELATIVE_PATH
  external/ompvv
  WORKING_DIRECTORY
  ${PROJECT_SOURCE_DIR}
  TEST_FILE
  Makefile
  REPO_URL
  https://github.com/OpenMP-Validation-and-Verification/OpenMP_VV.git
  REPO_BRANCH
  "master"
)

set(ROCPROFSYS_OMPVV_SUBMODULE_DIR "${PROJECT_SOURCE_DIR}/external/ompvv")
set(ROCPROFSYS_OPENMP_EXTERNAL_DIR "${CMAKE_BINARY_DIR}/examples/openmp/external")
set(ROCPROFSYS_OMPVV_SOURCE_DIR "${ROCPROFSYS_OPENMP_EXTERNAL_DIR}/ompvv")
set(OMPVV_MAKEDEF_FILE "sys/make/make.def")
set(OMPVV_BIN_DEST "${CMAKE_BINARY_DIR}")

# ---------------------------------------------------------------------------------------#
# Helper Function
# ---------------------------------------------------------------------------------------#

function(configure_ompvv_tests TEST_TYPE TEST_LIST_VAR)
    # Precreate bin dir to avoid race condition
    if(NOT EXISTS "${ROCPROFSYS_OMPVV_SOURCE_DIR}/bin")
        execute_process(
            COMMAND
                ${CMAKE_COMMAND} -E make_directory "${ROCPROFSYS_OMPVV_SOURCE_DIR}/bin"
        )
    endif()
    if(TEST_TYPE STREQUAL "HOST")
        set(TARGET_PREFIX "openmp-vv-host")
        set(FOFFLOADING_FLAGS "-fopenmp --offload-host-only")
    elseif(TEST_TYPE STREQUAL "OFFLOAD")
        set(TARGET_PREFIX "openmp-vv-offload")
        set(FOFFLOADING_FLAGS "-fopenmp")
        foreach(arch IN LISTS DEFAULT_GPU_TARGETS)
            set(FOFFLOADING_FLAGS "${FOFFLOADING_FLAGS} --offload-arch=${arch}")
        endforeach()
    else()
        rocprofiler_systems_message(FATAL_ERROR "Unknown TEST_TYPE - Only HOST and OFFLOAD supported")
    endif()

    set(LOCAL_TEST_LIST "")

    foreach(TEST IN LISTS ${TEST_LIST_VAR})
        get_filename_component(TEST_NAME ${TEST} NAME_WE)
        string(REPLACE "_" "-" TARGET_NAME ${TEST_NAME})
        set(TARGET_NAME "${TARGET_PREFIX}-${TARGET_NAME}")

        set(CUSTOM_FFLAGS "-lm ${FOFFLOADING_FLAGS}")
        set(CUSTOM_FLINKFLAGS "-lm ${FOFFLOADING_FLAGS}")

        add_custom_command(
            OUTPUT "${OMPVV_BIN_DEST}/${TARGET_NAME}"
            COMMAND
                make FC=${OMPVV_FC} DEVICE_TYPE=amd OMP_VERSION=${OMPVV_OPENMP_VERSION}
                NUM_THREADS_HOST=${OMPVV_NUM_THREADS_HOST}
                NUM_THREADS_DEVICE=${OMPVV_NUM_THREADS_DEVICE}
                NUM_TEAMS_DEVICE=${OMPVV_NUM_TEAMS_DEVICE}
                "FOFFLOADING=${FOFFLOADING_FLAGS}" "FFLAGS=${CUSTOM_FFLAGS}"
                "FLINKFLAGS=${CUSTOM_FLINKFLAGS}" "SOURCES=${TEST}" compile >
                ${CMAKE_CURRENT_BINARY_DIR}/ompvv-compile-${TARGET_NAME}.log 2>&1
            COMMAND
                ${CMAKE_COMMAND} -E copy_if_different
                "${ROCPROFSYS_OMPVV_SOURCE_DIR}/bin/${TEST_NAME}.F90.o"
                "${OMPVV_BIN_DEST}/${TARGET_NAME}"
            COMMENT "Compiling OMPVV test: ${TEST_NAME}"
            WORKING_DIRECTORY ${ROCPROFSYS_OMPVV_SOURCE_DIR}
            VERBATIM
            DEPENDS
                ${ROCPROFSYS_OMPVV_SOURCE_DIR}/${TEST}
                ${ROCPROFSYS_OMPVV_SOURCE_DIR}/${OMPVV_MAKEDEF_FILE}
        )

        list(APPEND LOCAL_TEST_LIST ${TARGET_NAME})

        add_custom_target(
            "${TARGET_NAME}-build"
            ALL
            DEPENDS "${OMPVV_BIN_DEST}/${TARGET_NAME}"
        )
        add_executable("${TARGET_NAME}-exec" IMPORTED GLOBAL)
        set_target_properties(
            "${TARGET_NAME}-exec"
            PROPERTIES IMPORTED_LOCATION "${OMPVV_BIN_DEST}/${TARGET_NAME}"
        )
    endforeach()

    if(TEST_TYPE STREQUAL "HOST")
        set(ROCPROFSYS_OMPVV_HOST_TESTS "${LOCAL_TEST_LIST}" PARENT_SCOPE)
    else()
        set(ROCPROFSYS_OMPVV_OFFLOAD_TESTS "${LOCAL_TEST_LIST}" PARENT_SCOPE)
    endif()
endfunction()

# ---------------------------------------------------------------------------------------#
# Copy OMPVV to build folder
# ---------------------------------------------------------------------------------------#

if(NOT EXISTS "${ROCPROFSYS_OMPVV_SOURCE_DIR}")
    execute_process(
        COMMAND
            ${CMAKE_COMMAND} -E copy_directory ${ROCPROFSYS_OMPVV_SUBMODULE_DIR}
            ${ROCPROFSYS_OMPVV_SOURCE_DIR}
    )
endif()

# ---------------------------------------------------------------------------------------#
# Compiler setup
# ---------------------------------------------------------------------------------------#

# OMPVV requires amdclang++
find_program(
    amdclangpp_EXECUTABLE
    NAMES amdclang++
    HINTS ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    PATHS ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES bin llvm/bin
)
mark_as_advanced(amdclangpp_EXECUTABLE)

if(NOT amdclangpp_EXECUTABLE)
    rocprofiler_systems_message(
      FATAL_ERROR
      "Could not find amdclang++. This is required for the OMPVV tests."
    )
endif()

find_program(
    amdflang_EXECUTABLE
    NAMES amdflang
    HINTS ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    PATHS ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES bin llvm/bin
)
mark_as_advanced(amdflang_EXECUTABLE)

if(NOT amdflang_EXECUTABLE)
    rocprofiler_systems_message(
      FATAL_ERROR
      "Could not find amdflang. This is required for the OMPVV tests."
    )
endif()

set(OMPVV_COMPILER
    "${amdflang_EXECUTABLE}"
    CACHE FILEPATH
    "Fortran compiler used for OMPVV tests"
)
set(OMPVV_FC "amdflang")

execute_process(
    COMMAND ${amdflang_EXECUTABLE} --version
    OUTPUT_VARIABLE AMDFLANG_VERSION_CMD_OUTPUT
    RESULT_VARIABLE AMDFLANG_VERSION_CMD_RESULT
)

set(AMDFLANG_FULL_VERSION "")

if(AMDFLANG_VERSION_CMD_RESULT EQUAL 0)
    string(
        REGEX MATCH
        "version ([0-9]+)\\.([0-9]+)\\.([0-9]+)"
        VERSION_MATCH
        "${AMDFLANG_VERSION_CMD_OUTPUT}"
    )
    if(VERSION_MATCH)
        set(AMDFLANG_VERSION_MAJOR "${CMAKE_MATCH_1}")
        set(AMDFLANG_VERSION_MINOR "${CMAKE_MATCH_2}")
        set(AMDFLANG_VERSION_PATCH "${CMAKE_MATCH_3}")
        set(AMDFLANG_FULL_VERSION
            "${AMDFLANG_VERSION_MAJOR}.${AMDFLANG_VERSION_MINOR}.${AMDFLANG_VERSION_PATCH}"
        )
        rocprofiler_systems_message(STATUS "Detected amdflang version: ${AMDFLANG_FULL_VERSION}")
    endif()
endif()

if(NOT AMDFLANG_FULL_VERSION)
    rocprofiler_systems_message(FATAL_ERROR "Failed to detect amdflang version.")
endif()

# ---------------------------------------------------------------------------------------#
# Variables
# ---------------------------------------------------------------------------------------#

set(OMPVV_OPENMP_VERSION 5.0)
set(OMPVV_TDIR "tests/${OMPVV_OPENMP_VERSION}")
set(OMPVV_NUM_THREADS_HOST ${ROCPROFSYS_THREAD_COUNT})
set(OMPVV_NUM_TEAMS_DEVICE 8) # default used by ompvv
set(OMPVV_NUM_THREADS_DEVICE 8) # default used by ompvv

# Files listed below are OpenMP 5.0 tests that compile for
# amdflang 20.0, 19.0 and 18.0.
# Set of host only tests to be compiled
set(OMPVV_HOST_TESTS_TO_COMPILE
    "${OMPVV_TDIR}/teams/test_team_default_shared.F90"
    "${OMPVV_TDIR}/parallel_for_simd/test_parallel_for_simd_atomic.F90"
)

# Set of offloading tests to be compiled (excluding reduction and simd_atomic
# tests due to OMPVV test failure)
set(OMPVV_OFFLOAD_TESTS_TO_COMPILE "")
if(OMPVV_USE_OFFLOAD_TESTS)
    set(OMPVV_OFFLOAD_TESTS_TO_COMPILE
        "${OMPVV_TDIR}/target_simd/test_target_simd_if.F90"
        "${OMPVV_TDIR}/target_teams_distribute_parallel_for/test_target_teams_distribute_parallel_for_collapse.F90"
    )
endif()

# To be added once task_detach race condition resolved in SDK
# if(AMDFLANG_VERSION_MAJOR EQUAL 20)
#     list(APPEND OMPVV_HOST_TESTS_TO_COMPILE
#         "${OMPVV_TDIR}/task/test_task_detach.F90"
#     )
# endif()

# ---------------------------------------------------------------------------------------#
# Setup build time compilation
# ---------------------------------------------------------------------------------------#

set(ROCPROFSYS_OMPVV_HOST_TESTS "")
set(ROCPROFSYS_OMPVV_OFFLOAD_TESTS "")

configure_ompvv_tests("HOST" OMPVV_HOST_TESTS_TO_COMPILE)
if(OMPVV_USE_OFFLOAD_TESTS)
    configure_ompvv_tests("OFFLOAD" OMPVV_OFFLOAD_TESTS_TO_COMPILE)
endif()

set(ROCPROFSYS_OMPVV_HOST_TESTS
    "${ROCPROFSYS_OMPVV_HOST_TESTS}"
    CACHE STRING
    "Internal variable used to generate OMPVV Host CTests"
    FORCE
)

set(ROCPROFSYS_OMPVV_OFFLOAD_TESTS
    "${ROCPROFSYS_OMPVV_OFFLOAD_TESTS}"
    CACHE STRING
    "Internal variable used to generate OMPVV Offload CTests"
    FORCE
)

rocprofiler_systems_message(STATUS
                            "Successfully configured OMPVV"
)
