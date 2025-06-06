# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#
# configuration and functions for testing
#
include_guard(DIRECTORY)

if(EXISTS /etc/os-release AND NOT IS_DIRECTORY /etc/os-release)
    file(READ /etc/os-release _OS_RELEASE_RAW)

    if(_OS_RELEASE_RAW)
        string(REPLACE "\"" "" _OS_RELEASE_RAW "${_OS_RELEASE_RAW}")
        string(REPLACE "-" " " _OS_RELEASE_RAW "${_OS_RELEASE_RAW}")
        string(REGEX REPLACE "NAME=.*\nVERSION=([0-9]+)\.([0-9]+).*\nID=([a-z]+).*"
                             "\\3-\\1.\\2" _OS_RELEASE "${_OS_RELEASE_RAW}")
    endif()
    unset(_OS_RELEASE_RAW)
endif()

rocprofiler_systems_message(STATUS "OS release: ${_OS_RELEASE}")

include(ProcessorCount)
if(NOT DEFINED NUM_PROCS_REAL)
    processorcount(NUM_PROCS_REAL)
endif()

if(NOT DEFINED NUM_PROCS)
    set(NUM_PROCS 2)
endif()

math(EXPR NUM_SAMPLING_PROCS "${NUM_PROCS_REAL}-1")
if(NUM_SAMPLING_PROCS GREATER 3)
    set(NUM_SAMPLING_PROCS 3)
endif()

math(EXPR NUM_THREADS "${NUM_PROCS_REAL} + (${NUM_PROCS_REAL} / 2)")
if(NUM_THREADS GREATER 12)
    set(NUM_THREADS 12)
endif()

math(EXPR MAX_CAUSAL_ITERATIONS "(${ROCPROFSYS_MAX_THREADS} - 1) / 2")
if(MAX_CAUSAL_ITERATIONS GREATER 100)
    set(MAX_CAUSAL_ITERATIONS 100)
endif()

set(_test_library_path
    "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:$ENV{LD_LIBRARY_PATH}")
set(_test_openmp_env "OMP_PROC_BIND=spread" "OMP_PLACES=threads" "OMP_NUM_THREADS=2")

set(_base_environment
    "ROCPROFSYS_TRACE=ON" "ROCPROFSYS_PROFILE=ON" "ROCPROFSYS_USE_SAMPLING=ON"
    "ROCPROFSYS_USE_PROCESS_SAMPLING=ON" "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_FILE_OUTPUT=ON" "${_test_openmp_env}" "${_test_library_path}")

set(_flat_environment
    "ROCPROFSYS_TRACE=ON"
    "ROCPROFSYS_PROFILE=ON"
    "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_COUT_OUTPUT=ON"
    "ROCPROFSYS_FLAT_PROFILE=ON"
    "ROCPROFSYS_TIMELINE_PROFILE=OFF"
    "ROCPROFSYS_COLLAPSE_PROCESSES=ON"
    "ROCPROFSYS_COLLAPSE_THREADS=ON"
    "ROCPROFSYS_SAMPLING_FREQ=50"
    "ROCPROFSYS_TIMEMORY_COMPONENTS=wall_clock,trip_count"
    "${_test_openmp_env}"
    "${_test_library_path}")

set(_lock_environment
    "ROCPROFSYS_USE_SAMPLING=ON"
    "ROCPROFSYS_USE_PROCESS_SAMPLING=OFF"
    "ROCPROFSYS_SAMPLING_FREQ=750"
    "ROCPROFSYS_COLLAPSE_THREADS=ON"
    "ROCPROFSYS_TRACE_THREAD_LOCKS=ON"
    "ROCPROFSYS_TRACE_THREAD_SPIN_LOCKS=ON"
    "ROCPROFSYS_TRACE_THREAD_RW_LOCKS=ON"
    "ROCPROFSYS_COUT_OUTPUT=ON"
    "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_TIMELINE_PROFILE=OFF"
    "ROCPROFSYS_VERBOSE=2"
    "${_test_library_path}")

set(_ompt_environment
    "ROCPROFSYS_TRACE=ON"
    "ROCPROFSYS_PROFILE=ON"
    "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_USE_OMPT=ON"
    "ROCPROFSYS_TIMEMORY_COMPONENTS=wall_clock,trip_count,peak_rss"
    "${_test_openmp_env}"
    "${_test_library_path}")

set(_perfetto_environment
    "ROCPROFSYS_TRACE=ON"
    "ROCPROFSYS_PROFILE=OFF"
    "ROCPROFSYS_USE_SAMPLING=ON"
    "ROCPROFSYS_USE_PROCESS_SAMPLING=ON"
    "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_PERFETTO_BACKEND=inprocess"
    "ROCPROFSYS_PERFETTO_FILL_POLICY=ring_buffer"
    "${_test_openmp_env}"
    "${_test_library_path}")

set(_timemory_environment
    "ROCPROFSYS_TRACE=OFF"
    "ROCPROFSYS_PROFILE=ON"
    "ROCPROFSYS_USE_SAMPLING=ON"
    "ROCPROFSYS_USE_PROCESS_SAMPLING=ON"
    "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_TIMEMORY_COMPONENTS=wall_clock,trip_count,peak_rss"
    "${_test_openmp_env}"
    "${_test_library_path}")

set(_test_environment ${_base_environment})

set(_causal_environment
    "${_test_openmp_env}" "${_test_library_path}" "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_FILE_OUTPUT=ON" "ROCPROFSYS_CAUSAL_RANDOM_SEED=1342342")

set(_python_environment
    "ROCPROFSYS_TRACE=ON"
    "ROCPROFSYS_PROFILE=ON"
    "ROCPROFSYS_USE_SAMPLING=OFF"
    "ROCPROFSYS_USE_PROCESS_SAMPLING=ON"
    "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_TREE_OUTPUT=OFF"
    "ROCPROFSYS_USE_PID=OFF"
    "ROCPROFSYS_TIMEMORY_COMPONENTS=wall_clock,trip_count"
    "${_test_library_path}"
    "PYTHONPATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_PYTHONDIR}")

set(_attach_environment
    "ROCPROFSYS_TRACE=ON"
    "ROCPROFSYS_PROFILE=ON"
    "ROCPROFSYS_USE_SAMPLING=OFF"
    "ROCPROFSYS_USE_PROCESS_SAMPLING=ON"
    "ROCPROFSYS_USE_OMPT=ON"
    "ROCPROFSYS_USE_KOKKOSP=ON"
    "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_USE_PID=OFF"
    "ROCPROFSYS_TIMEMORY_COMPONENTS=wall_clock,trip_count"
    "OMP_NUM_THREADS=${NUM_PROCS_REAL}"
    "${_test_library_path}")

set(_rccl_environment
    "ROCPROFSYS_TRACE=ON"
    "ROCPROFSYS_PROFILE=ON"
    "ROCPROFSYS_USE_SAMPLING=OFF"
    "ROCPROFSYS_USE_PROCESS_SAMPLING=ON"
    "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_USE_PID=OFF"
    "ROCPROFSYS_USE_RCCLP=ON"
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy"
    "${_test_openmp_env}"
    "${_test_library_path}")

set(_window_environment
    "ROCPROFSYS_TRACE=ON"
    "ROCPROFSYS_PROFILE=ON"
    "ROCPROFSYS_USE_SAMPLING=OFF"
    "ROCPROFSYS_USE_PROCESS_SAMPLING=OFF"
    "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_FILE_OUTPUT=ON"
    "ROCPROFSYS_VERBOSE=2"
    "${_test_openmp_env}"
    "${_test_library_path}")

# -------------------------------------------------------------------------------------- #

set(MPIEXEC_EXECUTABLE_ARGS)
option(
    ROCPROFSYS_CI_MPI_RUN_AS_ROOT
    "Enabled --allow-run-as-root in MPI tests with OpenMPI. Enable with care! Should only be in docker containers"
    OFF)
mark_as_advanced(ROCPROFSYS_CI_MPI_RUN_AS_ROOT)
if(ROCPROFSYS_CI_MPI_RUN_AS_ROOT)
    execute_process(
        COMMAND ${MPIEXEC_EXECUTABLE} --allow-run-as-root --help
        RESULT_VARIABLE _mpiexec_allow_run_as_root
        OUTPUT_QUIET ERROR_QUIET)
    if(_mpiexec_allow_run_as_root EQUAL 0)
        list(APPEND MPIEXEC_EXECUTABLE_ARGS --allow-run-as-root)
    endif()
endif()

execute_process(
    COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -n 1 ls
    RESULT_VARIABLE _mpiexec_oversubscribe
    OUTPUT_QUIET ERROR_QUIET)

set(rocprofiler_systems_perf_event_paranoid "4")
set(rocprofiler_systems_cap_sys_admin "1")
set(rocprofiler_systems_cap_perfmon "1")

if(EXISTS "/proc/sys/kernel/perf_event_paranoid")
    file(STRINGS "/proc/sys/kernel/perf_event_paranoid"
         rocprofiler_systems_perf_event_paranoid LIMIT_COUNT 1)
endif()

execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -O2 -g -std=c++17
            ${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-capchk.cpp -o rocprof-sys-capchk
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    RESULT_VARIABLE _capchk_compile
    OUTPUT_QUIET ERROR_QUIET)

if(_capchk_compile EQUAL 0)
    execute_process(
        COMMAND ${PROJECT_BINARY_DIR}/bin/rocprof-sys-capchk CAP_SYS_ADMIN effective
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        RESULT_VARIABLE rocprofiler_systems_cap_sys_admin
        OUTPUT_QUIET ERROR_QUIET)

    execute_process(
        COMMAND ${PROJECT_BINARY_DIR}/bin/rocprof-sys-capchk CAP_PERFMON effective
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        RESULT_VARIABLE rocprofiler_systems_cap_perfmon
        OUTPUT_QUIET ERROR_QUIET)
endif()

rocprofiler_systems_message(
    STATUS "perf_event_paranoid: ${rocprofiler_systems_perf_event_paranoid}")
rocprofiler_systems_message(STATUS "CAP_SYS_ADMIN: ${rocprofiler_systems_cap_sys_admin}")
rocprofiler_systems_message(STATUS "CAP_PERFMON: ${rocprofiler_systems_cap_perfmon}")

if(_mpiexec_oversubscribe EQUAL 0)
    list(APPEND MPIEXEC_EXECUTABLE_ARGS --oversubscribe)
endif()

# -------------------------------------------------------------------------------------- #

set(_VALID_GPU OFF)
if(ROCPROFSYS_USE_ROCM AND (NOT DEFINED ROCPROFSYS_CI_GPU OR ROCPROFSYS_CI_GPU))
    set(_VALID_GPU ON)
    find_program(
        ROCPROFSYS_AMD_SMI_EXE
        NAMES amd-smi
        HINTS ${ROCmVersion_DIR}
        PATHS ${ROCmVersion_DIR}
        PATH_SUFFIXES bin)
    if(ROCPROFSYS_AMD_SMI_EXE)
        execute_process(
            COMMAND ${ROCPROFSYS_AMD_SMI_EXE}
            OUTPUT_VARIABLE _AMDSMI_OUT
            ERROR_VARIABLE _AMDSMI_ERR
            RESULT_VARIABLE _AMDSMI_RET)
        if(_AMDSMI_RET EQUAL 0)
            if("${_AMDSMI_OUTPUT}" MATCHES "ERROR" OR "${_AMDSMI_ERR}" MATCHES "ERROR")
                set(_VALID_GPU OFF)
            endif()
        else()
            set(_VALID_GPU OFF)
        endif()
    endif()
    if(NOT _VALID_GPU)
        rocprofiler_systems_message(
            AUTHOR_WARNING "amd-smi did not successfully run. Disabling GPU tests...")
    endif()
endif()

set(LULESH_USE_GPU ${LULESH_USE_ROCM})
if(LULESH_USE_CUDA)
    set(LULESH_USE_GPU ON)
endif()

# -------------------------------------------------------------------------------------- #

macro(ROCPROFILER_SYSTEMS_CHECK_PASS_FAIL_REGEX NAME PASS FAIL)
    if(NOT "${${PASS}}" STREQUAL ""
       AND NOT "${${FAIL}}" STREQUAL ""
       AND NOT "${${FAIL}}" MATCHES "\\|ROCPROFSYS_ABORT_FAIL_REGEX"
       AND NOT "${${FAIL}}" MATCHES "${ROCPROFSYS_ABORT_FAIL_REGEX}")
        rocprofiler_systems_message(
            FATAL_ERROR
            "${NAME} has set pass and fail regexes but fail regex does not include '|ROCPROFSYS_ABORT_FAIL_REGEX'"
            )
    endif()

    if("${${FAIL}}" STREQUAL "")
        set(${FAIL} "(${ROCPROFSYS_ABORT_FAIL_REGEX})")
    else()
        string(REPLACE "|ROCPROFSYS_ABORT_FAIL_REGEX" "|${ROCPROFSYS_ABORT_FAIL_REGEX}"
                       ${FAIL} "${${FAIL}}")
    endif()
endmacro()

# -------------------------------------------------------------------------------------- #

function(ROCPROFILER_SYSTEMS_WRITE_TEST_CONFIG _FILE _ENV)
    set(_ENV_ONLY
        "ROCPROFSYS_(CI|CI_TIMEOUT|MODE|USE_MPIP|DEBUG_[A-Z_]+|FORCE_ROCPROFILER_INIT|DEFAULT_MIN_INSTRUCTIONS|MONOCHROME|VERBOSE)="
        )
    set(_FILE_CONTENTS)
    set(_ENV_CONTENTS)

    set(_DEBUG_SETTINGS ON)
    foreach(_VAL ${${_ENV}})
        if("${_VAL}" MATCHES "^ROCPROFSYS_DEBUG_SETTINGS=")
            set(_DEBUG_SETTINGS OFF)
        endif()
        if("${_VAL}" MATCHES "^ROCPROFSYS_" AND NOT "${_VAL}" MATCHES "${_ENV_ONLY}")
            set(_FILE_CONTENTS "${_FILE_CONTENTS}${_VAL}\n")
        else()
            list(APPEND _ENV_CONTENTS "${_VAL}")
        endif()
    endforeach()

    set(_CONFIG_FILE ${PROJECT_BINARY_DIR}/rocprof-sys-tests-config/${_FILE})
    file(
        WRITE ${_CONFIG_FILE}
        "# auto-generated by cmake

# default values
ROCPROFSYS_CI                     = ON
ROCPROFSYS_VERBOSE                = 1
ROCPROFSYS_DL_VERBOSE             = 1
ROCPROFSYS_SAMPLING_FREQ          = 300
ROCPROFSYS_SAMPLING_DELAY         = 0.05
ROCPROFSYS_SAMPLING_CPUS          = 0-${NUM_SAMPLING_PROCS}
ROCPROFSYS_SAMPLING_GPUS          = $env:HIP_VISIBLE_DEVICES

# test-specific values
${_FILE_CONTENTS}
")
    list(APPEND _ENV_CONTENTS "ROCPROFSYS_CONFIG_FILE=${_CONFIG_FILE}")
    if(_DEBUG_SETTINGS)
        list(APPEND _ENV_CONTENTS "ROCPROFSYS_DEBUG_SETTINGS=1")
    endif()
    set(${_ENV}
        "${_ENV_CONTENTS}"
        PARENT_SCOPE)
endfunction()

# -------------------------------------------------------------------------------------- #
# Check GPU architectures on the system. If a regex is provided, it will be used to filter
# the architectures. Otherwise, all architectures will be returned. Uses rocminfo to get
# the architectures.
function(ROCPROFILER_SYSTEMS_GET_GFX_ARCHS _VAR)
    cmake_parse_arguments(ARG "ECHO" "PREFIX;DELIM;GFX_MATCH" "" ${ARGN})

    if(NOT DEFINED ARG_DELIM)
        set(ARG_DELIM ", ")
    endif()

    if(NOT DEFINED ARG_PREFIX)
        set(ARG_PREFIX "[${PROJECT_NAME}] ")
    endif()

    find_program(
        rocminfo_EXECUTABLE
        NAMES rocminfo
        HINTS ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm
        PATHS ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm
        PATH_SUFFIXES bin)

    if(rocminfo_EXECUTABLE)
        execute_process(
            COMMAND ${rocminfo_EXECUTABLE}
            RESULT_VARIABLE rocminfo_RET
            OUTPUT_VARIABLE rocminfo_OUT
            ERROR_VARIABLE rocminfo_ERR
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)

        if(rocminfo_RET EQUAL 0)
            string(REGEX MATCHALL "gfx([0-9A-Fa-f]+)" rocminfo_GFXINFO "${rocminfo_OUT}")
            list(REMOVE_DUPLICATES rocminfo_GFXINFO)
            set(${_VAR}
                "${rocminfo_GFXINFO}"
                PARENT_SCOPE)

            if(ARG_ECHO)
                string(REPLACE ";" "${ARG_DELIM}" _GFXINFO_ECHO "${rocminfo_GFXINFO}")
                message(STATUS "${ARG_PREFIX}System architectures: ${_GFXINFO_ECHO}")
            endif()

            # Filter the architectures if a regex is provided
            if(ARG_GFX_MATCH)
                string(REGEX MATCH "${ARG_GFX_MATCH}" _GFX_MATCH "${rocminfo_GFXINFO}")
                list(REMOVE_DUPLICATES _GFX_MATCH)
                set(${_VAR}
                    "${_GFX_MATCH}"
                    PARENT_SCOPE)

                if(ARG_ECHO)
                    string(REPLACE ";" "${ARG_DELIM}" _GFXINFO_ECHO "${_GFX_MATCH}")
                    message(
                        STATUS
                            "${ARG_PREFIX}System architectures (filtered: ${ARG_GFX_MATCH}): ${_GFXINFO_ECHO}"
                        )
                endif()
            endif()
        else()
            message(
                AUTHOR_WARNING
                    "${rocminfo_EXECUTABLE} failed with error code ${rocminfo_RET}\nstderr:\n${rocminfo_ERR}\nstdout:\n${rocminfo_OUT}"
                )
        endif()
    endif()
endfunction()

# -------------------------------------------------------------------------------------- #
# extends the timeout when sanitizers are used due to slowdown
function(ROCPROFILER_SYSTEMS_ADJUST_TIMEOUT_FOR_SANITIZER _VAR)
    if(ROCPROFSYS_USE_SANITIZER)
        math(EXPR _timeout_v "2 * ${${_VAR}}")
        set(${_VAR}
            "${_timeout_v}"
            PARENT_SCOPE)
    endif()
endfunction()

# -------------------------------------------------------------------------------------- #
# extends the timeout when sanitizers are used due to slowdown
macro(ROCPROFILER_SYSTEMS_PATCH_SANITIZER_ENVIRONMENT _VAR)
    if(ROCPROFSYS_USE_SANITIZER)
        if(ROCPROFSYS_USE_SANITIZER)
            if(ROCPROFSYS_SANITIZER_TYPE MATCHES "address")
                if(NOT ASAN_LIBRARY)
                    rocprofiler_systems_message(
                        FATAL_ERROR
                        "Please define the realpath to the address sanitizer library in variable ASAN_LIBRARY"
                        )
                endif()
                list(APPEND ${_VAR} "LD_PRELOAD=${ASAN_LIBRARY}")
            elseif(ROCPROFSYS_SANITIZER_TYPE MATCHES "thread")
                if(NOT TSAN_LIBRARY)
                    rocprofiler_systems_message(
                        FATAL_ERROR
                        "Please define the realpath to the thread sanitizer library in variable TSAN_LIBRARY"
                        )
                endif()
                list(APPEND ${_VAR} "LD_PRELOAD=${TSAN_LIBRARY}")
            endif()
        endif()
    endif()
endmacro()

# -------------------------------------------------------------------------------------- #

function(ROCPROFILER_SYSTEMS_ADD_TEST)
    foreach(_PREFIX SAMPLING RUNTIME REWRITE REWRITE_RUN BASELINE)
        foreach(_TYPE PASS FAIL SKIP)
            list(APPEND _REGEX_OPTS "${_PREFIX}_${_TYPE}_REGEX")
        endforeach()
    endforeach()
    set(_KWARGS REWRITE_ARGS RUNTIME_ARGS SAMPLING_ARGS RUN_ARGS ENVIRONMENT LABELS
                PROPERTIES ${_REGEX_OPTS})

    cmake_parse_arguments(
        TEST "SKIP_BASELINE;SKIP_SAMPLING;SKIP_REWRITE;SKIP_RUNTIME"
        "NAME;TARGET;MPI;GPU;NUM_PROCS;SAMPLING_TIMEOUT;REWRITE_TIMEOUT;RUNTIME_TIMEOUT"
        "${_KWARGS}" ${ARGN})

    foreach(_PREFIX SAMPLING RUNTIME REWRITE REWRITE_RUN BASELINE)
        if("${${_PREFIX}_FAIL_REGEX}" STREQUAL "")
            set(${_PREFIX}_FAIL_REGEX "(${ROCPROFSYS_ABORT_FAIL_REGEX})")
        endif()
    endforeach()

    if(TEST_GPU AND NOT _VALID_GPU)
        rocprofiler_systems_message(
            STATUS "${TEST_NAME} requires a GPU and no valid GPUs were found")
        return()
    endif()

    if("${TEST_MPI}" STREQUAL "")
        set(TEST_MPI OFF)
    endif()

    list(INSERT TEST_REWRITE_ARGS 0 --print-instrumented functions)
    list(INSERT TEST_RUNTIME_ARGS 0 --print-instrumented functions)

    if(NOT DEFINED TEST_NUM_PROCS)
        set(TEST_NUM_PROCS ${NUM_PROCS})
    endif()

    if(NUM_PROCS EQUAL 0)
        set(TEST_NUM_PROCS 0)
    endif()

    if(NOT TEST_REWRITE_TIMEOUT)
        set(TEST_REWRITE_TIMEOUT 120)
    endif()

    if(NOT TEST_RUNTIME_TIMEOUT)
        set(TEST_RUNTIME_TIMEOUT 300)
    endif()

    if(NOT TEST_SAMPLING_TIMEOUT)
        set(TEST_SAMPLING_TIMEOUT 120)
    endif()

    if(NOT DEFINED TEST_ENVIRONMENT OR "${TEST_ENVIRONMENT}" STREQUAL "")
        set(TEST_ENVIRONMENT "${_test_environment}")
    endif()

    list(APPEND TEST_ENVIRONMENT "ROCPROFSYS_CI=ON")

    if(TEST_GPU)
        list(APPEND TEST_LABELS "gpu")

        if(NOT "ROCPROFSYS_USE_ROCM=OFF" IN_LIST TEST_ENVIRONMENT)
            list(APPEND TEST_LABELS "rocm")
        endif()

        if(NOT "ROCPROFSYS_USE_ROCM=OFF" IN_LIST TEST_ENVIRONMENT)
            list(APPEND TEST_LABELS "amd-smi")
        endif()
    endif()

    if("ROCPROFSYS_USE_ROCM=ON" IN_LIST TEST_ENVIRONMENT AND NOT "rocm" IN_LIST
                                                             TEST_ENVIRONMENT)
        list(APPEND TEST_LABELS "rocm")
    endif()

    if("ROCPROFSYS_USE_AMD_SMI=ON" IN_LIST TEST_ENVIRONMENT AND NOT "amd-smi" IN_LIST
                                                                TEST_ENVIRONMENT)
        list(APPEND TEST_LABELS "amd-smi")
    endif()

    if(TARGET ${TEST_TARGET})
        if(DEFINED TEST_MPI
           AND ${TEST_MPI}
           AND TEST_NUM_PROCS GREATER 0)
            if(NOT TEST_NUM_PROCS GREATER NUM_PROCS_REAL)
                set(COMMAND_PREFIX ${MPIEXEC_EXECUTABLE} ${MPIEXEC_EXECUTABLE_ARGS}
                                   ${MPIEXEC_NUMPROC_FLAG} ${TEST_NUM_PROCS})
                list(APPEND TEST_LABELS mpi parallel-${TEST_NUM_PROCS})
                list(APPEND TEST_PROPERTIES PARALLEL_LEVEL ${TEST_NUM_PROCS})
            else()
                set(COMMAND_PREFIX ${MPIEXEC_EXECUTABLE} ${MPIEXEC_EXECUTABLE_ARGS}
                                   ${MPIEXEC_NUMPROC_FLAG} 1)
            endif()
        else()
            list(APPEND TEST_ENVIRONMENT "ROCPROFSYS_USE_PID=OFF")
        endif()

        if(NOT TEST_SKIP_BASELINE AND NOT ROCPROFSYS_USE_SANITIZER)
            add_test(
                NAME ${TEST_NAME}-baseline
                COMMAND ${COMMAND_PREFIX} $<TARGET_FILE:${TEST_TARGET}> ${TEST_RUN_ARGS}
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
        endif()

        if(NOT TEST_SKIP_SAMPLING)
            add_test(
                NAME ${TEST_NAME}-sampling
                COMMAND
                    ${COMMAND_PREFIX} $<TARGET_FILE:rocprofiler-systems-sample>
                    ${TEST_SAMPLE_ARGS} -- $<TARGET_FILE:${TEST_TARGET}> ${TEST_RUN_ARGS}
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
        endif()

        if(NOT TEST_SKIP_REWRITE)
            add_test(
                NAME ${TEST_NAME}-binary-rewrite
                COMMAND
                    $<TARGET_FILE:rocprofiler-systems-instrument> -o
                    $<TARGET_FILE_DIR:${TEST_TARGET}>/${TEST_NAME}.inst
                    ${TEST_REWRITE_ARGS} -- $<TARGET_FILE:${TEST_TARGET}>
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

            add_test(
                NAME ${TEST_NAME}-binary-rewrite-run
                COMMAND
                    ${COMMAND_PREFIX} $<TARGET_FILE:rocprofiler-systems-run> --
                    $<TARGET_FILE_DIR:${TEST_TARGET}>/${TEST_NAME}.inst ${TEST_RUN_ARGS}
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
        endif()

        if(NOT TEST_SKIP_RUNTIME AND NOT ROCPROFSYS_USE_SANITIZER)
            add_test(
                NAME ${TEST_NAME}-runtime-instrument
                COMMAND $<TARGET_FILE:rocprofiler-systems-instrument> ${TEST_RUNTIME_ARGS}
                        -- $<TARGET_FILE:${TEST_TARGET}> ${TEST_RUN_ARGS}
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
        endif()

        if(TEST ${TEST_NAME}-binary-rewrite-run)
            set_tests_properties(${TEST_NAME}-binary-rewrite-run
                                 PROPERTIES DEPENDS ${TEST_NAME}-binary-rewrite)
        endif()

        foreach(_TEST baseline sampling binary-rewrite binary-rewrite-run
                      runtime-instrument)
            string(REGEX REPLACE "-run(-|/)" "\\1" _prefix "${TEST_NAME}-${_TEST}/")
            set(_labels "${_TEST}")
            string(REPLACE "-run" "" _labels "${_TEST}")
            if(TEST_TARGET)
                list(APPEND _labels "${TEST_TARGET}")
            endif()
            if(TEST_LABELS)
                list(APPEND _labels "${TEST_LABELS}")
            endif()

            set(_environ
                "ROCPROFSYS_DEFAULT_MIN_INSTRUCTIONS=64" "${TEST_ENVIRONMENT}"
                "ROCPROFSYS_OUTPUT_PATH=${PROJECT_BINARY_DIR}/rocprof-sys-tests-output"
                "ROCPROFSYS_OUTPUT_PREFIX=${_prefix}")

            set(_timeout ${TEST_REWRITE_TIMEOUT})
            if("${_TEST}" MATCHES "sampling")
                set(_timeout ${TEST_SAMPLING_TIMEOUT})
            elseif("${_TEST}" MATCHES "runtime-instrument")
                set(_timeout ${TEST_RUNTIME_TIMEOUT})
            endif()

            set(_props)
            if("${_TEST}" MATCHES "run|sampling|baseline")
                set(_props ${TEST_PROPERTIES})
                if(NOT "RUN_SERIAL" IN_LIST _props)
                    list(APPEND _props RUN_SERIAL ON)
                endif()
            endif()

            if("${_TEST}" MATCHES "binary-rewrite-run")
                set(_REGEX_VAR REWRITE_RUN)
            elseif("${_TEST}" MATCHES "runtime-instrument")
                set(_REGEX_VAR RUNTIME)
            elseif("${_TEST}" MATCHES "binary-rewrite")
                set(_REGEX_VAR REWRITE)
            elseif("${_TEST}" MATCHES "baseline")
                set(_REGEX_VAR BASELINE)
            elseif("${_TEST}" MATCHES "sampling")
                set(_REGEX_VAR SAMPLING)
            else()
                set(_REGEX_VAR)
            endif()

            if("${_TEST}" MATCHES "binary-rewrite-run|runtime-instrument|sampling")
                rocprofiler_systems_patch_sanitizer_environment(_environ)
            endif()

            rocprofiler_systems_adjust_timeout_for_sanitizer(_timeout)

            foreach(_TYPE PASS FAIL SKIP)
                if(_REGEX_VAR)
                    set(_${_TYPE}_REGEX TEST_${_REGEX_VAR}_${_TYPE}_REGEX)
                else()
                    set(_${_TYPE}_REGEX)
                endif()
            endforeach()

            list(APPEND _environ "ROCPROFSYS_CI_TIMEOUT=${_timeout}")

            rocprofiler_systems_check_pass_fail_regex("${TEST_NAME}-${_TEST}"
                                                      "${_PASS_REGEX}" "${_FAIL_REGEX}")
            if(TEST ${TEST_NAME}-${_TEST})
                rocprofiler_systems_write_test_config(${TEST_NAME}-${_TEST}.cfg _environ)
                set_tests_properties(
                    ${TEST_NAME}-${_TEST}
                    PROPERTIES ENVIRONMENT
                               "${_environ}"
                               TIMEOUT
                               ${_timeout}
                               LABELS
                               "${_labels}"
                               PASS_REGULAR_EXPRESSION
                               "${${_PASS_REGEX}}"
                               FAIL_REGULAR_EXPRESSION
                               "${${_FAIL_REGEX}}"
                               SKIP_REGULAR_EXPRESSION
                               "${${_SKIP_REGEX}}"
                               ${_props})
            endif()
        endforeach()
    endif()
endfunction()

# -------------------------------------------------------------------------------------- #

function(ROCPROFILER_SYSTEMS_ADD_CAUSAL_TEST)
    foreach(_PREFIX CAUSAL CAUSAL_VALIDATE)
        foreach(_TYPE PASS FAIL SKIP)
            list(APPEND _REGEX_OPTS "${_PREFIX}_${_TYPE}_REGEX")
        endforeach()
    endforeach()

    set(_KWARGS CAUSAL_ARGS CAUSAL_VALIDATE_ARGS RUN_ARGS ENVIRONMENT LABELS PROPERTIES
                ${_REGEX_OPTS})

    cmake_parse_arguments(
        TEST "SKIP_BASELINE"
        "NAME;TARGET;CAUSAL_MODE;CAUSAL_TIMEOUT;CAUSAL_VALIDATE_TIMEOUT" "${_KWARGS}"
        ${ARGN})

    if(NOT DEFINED TEST_CAUSAL_MODE)
        rocprofiler_systems_message(FATAL_ERROR
                                    "${TEST_NAME} :: CAUSAL_MODE must be defined")
    endif()

    if(NOT TEST_CAUSAL_TIMEOUT)
        set(TEST_CAUSAL_TIMEOUT 600)
    endif()

    if(NOT TEST_CAUSAL_VALIDATE_TIMEOUT)
        set(TEST_CAUSAL_VALIDATE_TIMEOUT 60)
    endif()

    if("${TEST_CAUSAL_FAIL_REGEX}" STREQUAL "")
        set(TEST_CAUSAL_FAIL_REGEX "(${ROCPROFSYS_ABORT_FAIL_REGEX})")
    endif()

    if(TARGET ${TEST_TARGET})
        set(COMMAND_PREFIX $<TARGET_FILE:rocprofiler-systems-causal> --reset -m
                           ${TEST_CAUSAL_MODE} ${TEST_CAUSAL_ARGS} --)

        if(NOT TEST_SKIP_BASELINE)
            add_test(
                NAME ${TEST_NAME}-baseline
                COMMAND $<TARGET_FILE:${TEST_TARGET}> ${TEST_RUN_ARGS}
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
        endif()

        add_test(
            NAME causal-${TEST_NAME}
            COMMAND ${COMMAND_PREFIX} $<TARGET_FILE:${TEST_TARGET}> ${TEST_RUN_ARGS}
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

        if(NOT "${TEST_CAUSAL_VALIDATE_ARGS}" STREQUAL "")
            if("$ENV{ROCPROFSYS_CI}" MATCHES "ON|on|1|true|TRUE"
               OR "$ENV{CI}" MATCHES "true"
               OR NOT "$ENV{GITHUB_RUN_ID}" STREQUAL "")
                set(_VALIDATE_EXTRA "--ci")
            else()
                set(_VALIDATE_EXTRA "")
            endif()

            add_test(
                NAME validate-causal-${TEST_NAME}
                COMMAND ${CMAKE_CURRENT_LIST_DIR}/validate-causal-json.py
                        ${_VALIDATE_EXTRA} ${TEST_CAUSAL_VALIDATE_ARGS}
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
        endif()

        if(TEST validate-causal-${TEST_NAME})
            set_tests_properties(validate-causal-${TEST_NAME}
                                 PROPERTIES DEPENDS causal-${TEST_NAME})
        endif()

        foreach(_TEST baseline causal validate-causal)

            if(NOT TEST ${_TEST}-${TEST_NAME})
                if(NOT TEST ${TEST_NAME}-${_TEST})
                    continue()
                else()
                    set(_NAME "${TEST_NAME}-${_TEST}")
                endif()
            else()
                set(_NAME "${_TEST}-${TEST_NAME}")
            endif()

            set(_prefix "${_NAME}/")
            set(_labels "${_TEST}" "causal-profiling")

            if(TEST_TARGET)
                list(APPEND _labels "${TEST_TARGET}")
            endif()

            if(TEST_LABELS)
                list(APPEND _labels "${TEST_LABELS}")
            endif()

            set(_environ
                "${_causal_environment}"
                "ROCPROFSYS_OUTPUT_PATH=${PROJECT_BINARY_DIR}/rocprof-sys-tests-output"
                "ROCPROFSYS_OUTPUT_PREFIX=${_prefix}"
                "ROCPROFSYS_CI=ON"
                "ROCPROFSYS_USE_PID=OFF"
                "ROCPROFSYS_THREAD_POOL_SIZE=0"
                "ROCPROFSYS_VERBOSE=1"
                "ROCPROFSYS_DL_VERBOSE=0"
                "ROCPROFSYS_DEBUG_SETTINGS=0"
                "${TEST_ENVIRONMENT}")

            set(_timeout ${TEST_CAUSAL_TIMEOUT})

            rocprofiler_systems_adjust_timeout_for_sanitizer(_timeout)

            if("${_TEST}" MATCHES "validate-causal")
                set(_timeout ${TEST_CAUSAL_VALIDATE_TIMEOUT})
            endif()

            set(_props ${TEST_PROPERTIES})

            if("${_TEST}" STREQUAL "validate-causal")
                set(_REGEX_VAR CAUSAL_VALIDATE)
            elseif("${_TEST}" STREQUAL "causal")
                set(_REGEX_VAR CAUSAL)
            else()
                set(_REGEX_VAR)
            endif()

            foreach(_TYPE PASS FAIL SKIP)
                if(_REGEX_VAR)
                    set(_${_TYPE}_REGEX TEST_${_REGEX_VAR}_${_TYPE}_REGEX)
                else()
                    set(_${_TYPE}_REGEX)
                endif()
            endforeach()

            list(APPEND _environ "ROCPROFSYS_CI_TIMEOUT=${_timeout}")
            rocprofiler_systems_write_test_config(${_NAME}.cfg _environ)
            rocprofiler_systems_check_pass_fail_regex("${_NAME}" "${_PASS_REGEX}"
                                                      "${_FAIL_REGEX}")
            set_tests_properties(
                ${_NAME}
                PROPERTIES ENVIRONMENT
                           "${_environ}"
                           TIMEOUT
                           ${_timeout}
                           LABELS
                           "${_labels}"
                           PASS_REGULAR_EXPRESSION
                           "${${_PASS_REGEX}}"
                           FAIL_REGULAR_EXPRESSION
                           "${${_FAIL_REGEX}}"
                           SKIP_REGULAR_EXPRESSION
                           "${${_SKIP_REGEX}}"
                           ${_props})
        endforeach()
    endif()
endfunction()

# -------------------------------------------------------------------------------------- #

function(ROCPROFILER_SYSTEMS_ADD_PYTHON_TEST)
    if(NOT ROCPROFSYS_USE_PYTHON)
        return()
    endif()

    cmake_parse_arguments(
        TEST
        "STANDALONE" # options
        "NAME;FILE;TIMEOUT;PYTHON_EXECUTABLE;PYTHON_VERSION" # single value args
        "PROFILE_ARGS;RUN_ARGS;ENVIRONMENT;LABELS;PROPERTIES;PASS_REGEX;FAIL_REGEX;SKIP_REGEX;DEPENDS;COMMAND" # multiple
        # value args
        ${ARGN})

    if(NOT TEST_TIMEOUT)
        set(TEST_TIMEOUT 120)
    endif()

    rocprofiler_systems_adjust_timeout_for_sanitizer(TEST_TIMEOUT)

    set(PYTHON_EXECUTABLE "${TEST_PYTHON_EXECUTABLE}")

    if(NOT DEFINED TEST_ENVIRONMENT OR "${TEST_ENVIRONMENT}" STREQUAL "")
        set(TEST_ENVIRONMENT "${_python_environment}")
    endif()

    list(APPEND TEST_LABELS "python" "python-${TEST_PYTHON_VERSION}")

    if(NOT TEST_COMMAND)
        list(APPEND TEST_ENVIRONMENT "ROCPROFSYS_CI=ON"
             "ROCPROFSYS_OUTPUT_PATH=${PROJECT_BINARY_DIR}/rocprof-sys-tests-output")
        get_filename_component(_TEST_FILE "${TEST_FILE}" NAME)
        set(_TEST_FILE
            ${PROJECT_BINARY_DIR}/python/tests/${TEST_PYTHON_VERSION}/${_TEST_FILE})
        configure_file(${TEST_FILE} ${_TEST_FILE} @ONLY)
        if(TEST_STANDALONE)
            add_test(
                NAME ${TEST_NAME}-${TEST_PYTHON_VERSION}
                COMMAND ${TEST_PYTHON_EXECUTABLE} ${_TEST_FILE} ${TEST_RUN_ARGS}
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
        else()
            add_test(
                NAME ${TEST_NAME}-${TEST_PYTHON_VERSION}
                COMMAND ${TEST_PYTHON_EXECUTABLE} -m rocprofsys ${TEST_PROFILE_ARGS} --
                        ${_TEST_FILE} ${TEST_RUN_ARGS}
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
            add_test(
                NAME ${TEST_NAME}-${TEST_PYTHON_VERSION}-annotated
                COMMAND ${TEST_PYTHON_EXECUTABLE} -m rocprofsys ${TEST_PROFILE_ARGS}
                        --annotate-trace -- ${_TEST_FILE} ${TEST_RUN_ARGS}
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
        endif()
    else()
        list(APPEND TEST_LABELS "python-check" "python-${TEST_PYTHON_VERSION}-check")
        add_test(
            NAME ${TEST_NAME}-${TEST_PYTHON_VERSION}
            COMMAND ${TEST_COMMAND} ${TEST_FILE}
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    endif()

    foreach(_TEST ${TEST_NAME}-${TEST_PYTHON_VERSION}
                  ${TEST_NAME}-${TEST_PYTHON_VERSION}-annotated)

        if(NOT TEST "${_TEST}")
            continue()
        endif()

        string(REPLACE "${TEST_NAME}-${TEST_PYTHON_VERSION}" "${TEST_NAME}" _TEST_DIR
                       "${_TEST}")
        set(_TEST_ENV
            "${TEST_ENVIRONMENT}"
            "ROCPROFSYS_OUTPUT_PREFIX=${_TEST_DIR}/${TEST_PYTHON_VERSION}/"
            "ROCPROFSYS_CI_TIMEOUT=${TEST_TIMEOUT}")

        set(_TEST_PROPERTIES "${TEST_PROPERTIES}")
        # assign pass variable to pass regex
        set(_PASS_REGEX TEST_PASS_REGEX)
        # assign fail variable to fail regex
        set(_FAIL_REGEX TEST_FAIL_REGEX)

        rocprofiler_systems_check_pass_fail_regex("${_TEST}" "${_PASS_REGEX}"
                                                  "${_FAIL_REGEX}")
        set_tests_properties(
            ${_TEST}
            PROPERTIES ENVIRONMENT
                       "${_TEST_ENV}"
                       TIMEOUT
                       ${TEST_TIMEOUT}
                       LABELS
                       "${TEST_LABELS}"
                       DEPENDS
                       "${TEST_DEPENDS}"
                       PASS_REGULAR_EXPRESSION
                       "${${_PASS_REGEX}}"
                       FAIL_REGULAR_EXPRESSION
                       "${${_FAIL_REGEX}}"
                       SKIP_REGULAR_EXPRESSION
                       "${TEST_SKIP_REGEX}"
                       REQUIRED_FILES
                       "${TEST_FILE}"
                       ${_TEST_PROPERTIES})
    endforeach()
endfunction()

# -------------------------------------------------------------------------------------- #
#
# Find Python3 interpreter for output validation
#
# -------------------------------------------------------------------------------------- #

if(NOT ROCPROFSYS_USE_PYTHON)
    find_package(Python3 QUIET COMPONENTS Interpreter)

    if(Python3_FOUND)
        set(ROCPROFSYS_VALIDATION_PYTHON ${Python3_EXECUTABLE})
        execute_process(COMMAND ${Python3_EXECUTABLE} -c "import perfetto"
                        RESULT_VARIABLE ROCPROFSYS_VALIDATION_PYTHON_PERFETTO)

        if(NOT ROCPROFSYS_VALIDATION_PYTHON_PERFETTO EQUAL 0)
            rocprofiler_systems_message(AUTHOR_WARNING
                                        "Python3 found but perfetto support is disabled")
        endif()
    endif()
else()
    set(_INDEX 0)
    foreach(_VERSION ${ROCPROFSYS_PYTHON_VERSIONS})
        if(NOT ROCPROFSYS_USE_PYTHON)
            continue()
        endif()

        list(GET ROCPROFSYS_PYTHON_ROOT_DIRS ${_INDEX} _PYTHON_ROOT_DIR)

        rocprofiler_systems_find_python(
            _PYTHON
            ROOT_DIR "${_PYTHON_ROOT_DIR}"
            COMPONENTS Interpreter)

        if(_PYTHON_EXECUTABLE)
            set(ROCPROFSYS_VALIDATION_PYTHON ${_PYTHON_EXECUTABLE})
            execute_process(COMMAND ${_PYTHON_EXECUTABLE} -c "import perfetto"
                            RESULT_VARIABLE ROCPROFSYS_VALIDATION_PYTHON_PERFETTO)

            # prefer Python3 with perfetto support
            if(ROCPROFSYS_VALIDATION_PYTHON_PERFETTO EQUAL 0)
                break()
            else()
                rocprofiler_systems_message(
                    AUTHOR_WARNING
                    "${_PYTHON_EXECUTABLE} found but perfetto support is disabled")
            endif()
        endif()

        math(EXPR _INDEX "${_INDEX} + 1")
    endforeach()
endif()

if(NOT ROCPROFSYS_VALIDATION_PYTHON)
    rocprofiler_systems_message(
        AUTHOR_WARNING "Python3 interpreter not found. Validation tests will be disabled")
endif()

# -------------------------------------------------------------------------------------- #
#
# Output validation test function
#
# -------------------------------------------------------------------------------------- #

function(ROCPROFILER_SYSTEMS_ADD_VALIDATION_TEST)

    if(NOT ROCPROFSYS_VALIDATION_PYTHON)
        return()
    endif()

    cmake_parse_arguments(
        TEST
        ""
        "NAME;TIMEOUT;TIMEMORY_METRIC;TIMEMORY_FILE;PERFETTO_METRIC;PERFETTO_FILE"
        "ENVIRONMENT;LABELS;PROPERTIES;PASS_REGEX;FAIL_REGEX;SKIP_REGEX;DEPENDS;EXIST_FILES;ARGS"
        ${ARGN})

    if(NOT TEST "${TEST_NAME}")
        rocprofiler_systems_message(
            AUTHOR_WARNING
            "No validation test(s) for ${TEST_NAME} because test does not exist")
        return()
    endif()

    if(NOT TEST_TIMEOUT)
        set(TEST_TIMEOUT 30)
    endif()

    rocprofiler_systems_adjust_timeout_for_sanitizer(TEST_TIMEOUT)

    set(PYTHON_EXECUTABLE "${ROCPROFSYS_VALIDATION_PYTHON}")

    list(APPEND TEST_LABELS "validate")
    foreach(_DEP ${TEST_DEPENDS})
        list(APPEND TEST_LABELS "validate-${_DEP}")
    endforeach()

    list(APPEND TEST_DEPENDS "${TEST_NAME}")
    if("${TEST_NAME}" MATCHES "-binary-rewrite")
        list(APPEND TEST_DEPENDS "${TEST_NAME}-run")
    endif()

    if(NOT TEST_PASS_REGEX)
        set(TEST_PASS_REGEX
            "rocprof-sys-tests-output/${TEST_NAME}/(${TEST_TIMEMORY_FILE}|${TEST_PERFETTO_FILE}) validated"
            )
    endif()

    foreach(_FILE ${TEST_EXIST_FILES})
        add_test(
            NAME validate-${TEST_NAME}-${_FILE}-exists
            COMMAND test -e
                    ${PROJECT_BINARY_DIR}/rocprof-sys-tests-output/${TEST_NAME}/${_FILE}
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    endforeach()

    if(TEST_TIMEMORY_FILE)
        add_test(
            NAME validate-${TEST_NAME}-timemory
            COMMAND
                ${ROCPROFSYS_VALIDATION_PYTHON}
                ${CMAKE_CURRENT_LIST_DIR}/validate-timemory-json.py -m
                "${TEST_TIMEMORY_METRIC}" ${TEST_ARGS} -i
                ${PROJECT_BINARY_DIR}/rocprof-sys-tests-output/${TEST_NAME}/${TEST_TIMEMORY_FILE}
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    endif()

    if(ROCPROFSYS_VALIDATION_PYTHON_PERFETTO EQUAL 0 AND TEST_PERFETTO_FILE)
        add_test(
            NAME validate-${TEST_NAME}-perfetto
            COMMAND
                ${ROCPROFSYS_VALIDATION_PYTHON}
                ${CMAKE_CURRENT_LIST_DIR}/validate-perfetto-proto.py -m
                "${TEST_PERFETTO_METRIC}" ${TEST_ARGS} -i
                ${PROJECT_BINARY_DIR}/rocprof-sys-tests-output/${TEST_NAME}/${TEST_PERFETTO_FILE}
                -t /opt/trace_processor/bin/trace_processor_shell
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    endif()

    list(APPEND TEST_ENVIRONMENT "ROCPROFSYS_CI_TIMEOUT=${TEST_TIMEOUT}")

    foreach(_TEST validate-${TEST_NAME}-timemory validate-${TEST_NAME}-perfetto)

        if(NOT TEST "${_TEST}")
            continue()
        endif()

        rocprofiler_systems_check_pass_fail_regex("${_TEST}" "TEST_PASS_REGEX"
                                                  "TEST_FAIL_REGEX")
        set_tests_properties(
            ${_TEST}
            PROPERTIES ENVIRONMENT
                       "${TEST_ENVIRONMENT}"
                       TIMEOUT
                       ${TEST_TIMEOUT}
                       LABELS
                       "${TEST_LABELS}"
                       DEPENDS
                       "${TEST_DEPENDS};${TEST_NAME}"
                       PASS_REGULAR_EXPRESSION
                       "${TEST_PASS_REGEX}"
                       FAIL_REGULAR_EXPRESSION
                       "${TEST_FAIL_REGEX}"
                       SKIP_REGULAR_EXPRESSION
                       "${TEST_SKIP_REGEX}"
                       REQUIRED_FILES
                       "${TEST_FILE}"
                       ${TEST_PROPERTIES})
    endforeach()
endfunction()
