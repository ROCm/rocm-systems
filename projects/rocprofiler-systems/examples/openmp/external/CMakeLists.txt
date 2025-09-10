# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# ----------------------------------------------------------------------------------------#
#
# OMPVV (OpenMP Validation and Verification) submodule
#
# * Currently, GNU's libgomp cannot be used to capture traces. Only LLVM's
#   LIBOMP can be used for trace capture
# * amdflang uses LIBOMP 201611 (5.0), but not all OpenMP 5.0 features are
#   implemented. Only a subset of tests can be compiled.
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
set(ROCPROFSYS_OMPVV_SOURCE_DIR "${CMAKE_BINARY_DIR}/examples/openmp/external/ompvv")
set(OMPVV_MAKEDEF_FILE "sys/make/make.def")

# ---------------------------------------------------------------------------------------#
# Helper functions
# ---------------------------------------------------------------------------------------#

# Updates the make.def FOFFLOADING flag
function(update_make_def_foffloading FOFFLOADING_FLAGS)
    file(READ ${ROCPROFSYS_OMPVV_SOURCE_DIR}/${OMPVV_MAKEDEF_FILE} MAKE_DEF_CONTENT)

    string(
        REGEX REPLACE
        "FOFFLOADING[ \t]*=[ \t]*-fopenmp[^\n\r]*"
        "FOFFLOADING   =  ${FOFFLOADING_FLAGS}"
        MAKE_DEF_CONTENT
        "${MAKE_DEF_CONTENT}"
    )

    file(WRITE ${ROCPROFSYS_OMPVV_SOURCE_DIR}/${OMPVV_MAKEDEF_FILE} "${MAKE_DEF_CONTENT}")
endfunction()

# Individually compiles a given list of tests
function(compile_ompvv_tests TEST_LIST_VAR)
    # OMPVV doesn't support building subsets of test folders
    foreach(test IN LISTS ${TEST_LIST_VAR})
        rocprofiler_systems_message(STATUS "Compiling OMPVV test: ${test}")

        execute_process(
            COMMAND
                make FC=${OMPVV_FC} DEVICE_TYPE=amd OMP_VERSION=${OMPVV_OPENMP_VERSION}
                NUM_THREADS_HOST=${OMPVV_NUM_THREADS_HOST}
                NUM_THREADS_DEVICE=${OMPVV_NUM_THREADS_DEVICE}
                NUM_TEAMS_DEVICE=${OMPVV_NUM_TEAMS_DEVICE} "SOURCES=${test}" compile
            WORKING_DIRECTORY ${ROCPROFSYS_OMPVV_SOURCE_DIR}
            RESULT_VARIABLE OMPVV_BUILD_RESULT
            OUTPUT_VARIABLE OMPVV_BUILD_OUTPUT
            ERROR_VARIABLE OMPVV_BUILD_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
        )

        if(NOT OMPVV_BUILD_RESULT EQUAL 0)
            rocprofiler_systems_message(
              FATAL_ERROR "OMPVV build failed for ${test}: ${OMPVV_BUILD_ERROR}"
            )
        endif()
    endforeach()
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

# if(AMDFLANG_VERSION_MAJOR EQUAL 20)
#     list(APPEND OMPVV_HOST_TESTS_TO_COMPILE
#         "${OMPVV_TDIR}/task/test_task_detach.F90"
#     )
# endif()

# ---------------------------------------------------------------------------------------#
# Compile tests
# ---------------------------------------------------------------------------------------#

# Remove all -O3 flags to prevent function inlining
file(READ ${ROCPROFSYS_OMPVV_SOURCE_DIR}/${OMPVV_MAKEDEF_FILE} MAKE_DEF_CONTENT)
string(REGEX REPLACE "-O3[ \t]*" "" MAKE_DEF_CONTENT "${MAKE_DEF_CONTENT}")
file(WRITE ${ROCPROFSYS_OMPVV_SOURCE_DIR}/${OMPVV_MAKEDEF_FILE} "${MAKE_DEF_CONTENT}")

# OMPVV makefile option NO_OFFLOADING not supported for amdflang,
# must manually do it
set(OMPVV_HOST_FOFFLOADING_FLAGS "-fopenmp --offload-host-only")
update_make_def_foffloading("${OMPVV_HOST_FOFFLOADING_FLAGS}")
compile_ompvv_tests(OMPVV_HOST_TESTS_TO_COMPILE)

if(OMPVV_USE_OFFLOAD_TESTS)
    set(OMPVV_OFFLOAD_FOFFLOADING_FLAGS "-fopenmp")
    foreach(arch IN LISTS DEFAULT_GPU_TARGETS)
        set(OMPVV_OFFLOAD_FOFFLOADING_FLAGS
            "${OMPVV_OFFLOAD_FOFFLOADING_FLAGS} --offload-arch=${arch}"
        )
    endforeach()
    update_make_def_foffloading("${OMPVV_OFFLOAD_FOFFLOADING_FLAGS}")
    compile_ompvv_tests(OMPVV_OFFLOAD_TESTS_TO_COMPILE)
endif()

# ---------------------------------------------------------------------------------------#
# Copy generated binaries to top level of build folder
# ---------------------------------------------------------------------------------------#

set(OMPVV_BIN_SOURCE "${ROCPROFSYS_OMPVV_SOURCE_DIR}/bin")
set(OMPVV_BIN_DEST "${CMAKE_BINARY_DIR}")

set(ROCPROFSYS_OMPVV_HOST_TESTS "")
set(ROCPROFSYS_OMPVV_OFFLOAD_TESTS "")

file(GLOB OMPVV_BINARIES "${OMPVV_BIN_SOURCE}/*")

foreach(OMPVV_BIN ${OMPVV_BINARIES})
    get_filename_component(BIN_NAME ${OMPVV_BIN} NAME_WE)

    # Determine type of test
    set(IS_HOST_TEST FALSE)
    set(IS_OFFLOAD_TEST FALSE)
    foreach(host_test IN LISTS OMPVV_HOST_TESTS_TO_COMPILE)
        get_filename_component(HOST_TEST_NAME ${host_test} NAME_WE)
        if(BIN_NAME STREQUAL HOST_TEST_NAME)
            set(IS_HOST_TEST TRUE)
            break()
        endif()
    endforeach()

    if(NOT IS_HOST_TEST AND OMPVV_USE_OFFLOAD_TESTS)
        foreach(offload_test IN LISTS OMPVV_OFFLOAD_TESTS_TO_COMPILE)
            get_filename_component(OFFLOAD_TEST_NAME ${offload_test} NAME_WE)
            if(BIN_NAME STREQUAL OFFLOAD_TEST_NAME)
                set(IS_OFFLOAD_TEST TRUE)
                break()
            endif()
        endforeach()
    endif()

    if(NOT IS_HOST_TEST AND NOT IS_OFFLOAD_TEST)
        continue()
    endif()

    # For consistent naming with other tests
    string(REGEX REPLACE "\\.F90$" "" NEW_BIN_NAME ${BIN_NAME})
    string(REPLACE "_" "-" NEW_BIN_NAME ${NEW_BIN_NAME})

    file(COPY "${OMPVV_BIN}" DESTINATION "${OMPVV_BIN_DEST}")

    if(IS_HOST_TEST)
        set(TARGET_NAME "openmp-vv-host-${NEW_BIN_NAME}")
        list(APPEND ROCPROFSYS_OMPVV_HOST_TESTS ${TARGET_NAME})
    elseif(IS_OFFLOAD_TEST)
        set(TARGET_NAME "openmp-vv-offload-${NEW_BIN_NAME}")
        list(APPEND ROCPROFSYS_OMPVV_OFFLOAD_TESTS ${TARGET_NAME})
    endif()

    file(RENAME "${OMPVV_BIN_DEST}/${BIN_NAME}.F90.o" "${OMPVV_BIN_DEST}/${TARGET_NAME}")
    add_executable(${TARGET_NAME} IMPORTED GLOBAL)
    set_target_properties(
        ${TARGET_NAME}
        PROPERTIES IMPORTED_LOCATION "${OMPVV_BIN_DEST}/${TARGET_NAME}"
    )
endforeach()

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
                            "Successfully built and compiled OMPVV tests"
)
