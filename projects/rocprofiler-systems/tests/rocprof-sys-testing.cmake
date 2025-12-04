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

if(ROCPROFSYS_INSTALL_TESTS)
    # Paths for testing against installed binaries
    # Tests will run against the installed location, not the build directory

    set(ROCPROFSYS_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" 
        CACHE STRING "Path where rocprofiler-systems is installed" FORCE)
    set(ROCPROFSYS_SAMPLES_INSTALL_PREFIX 
        "${ROCPROFSYS_INSTALL_PREFIX}/share/rocprofiler-systems/samples" 
        CACHE STRING "Path where samples are installed" FORCE)
    set(ROCPROFSYS_BIN_INSTALL_PREFIX 
        "${ROCPROFSYS_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}" 
        CACHE STRING "Path where rocprof-sys binaries are installed" FORCE)
    set(ROCPROFSYS_TEST_INSTALL_PREFIX
        "${ROCPROFSYS_INSTALL_PREFIX}/share/rocprofiler-systems/tests"
        CACHE STRING "Path where test configuration files are installed" FORCE)
    set(ROCPROFSYS_BIN_INSTALL_PREFIX
        "${ROCPROFSYS_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}"
        CACHE STRING "Path where rocprof-sys binaries are installed" FORCE)

    # Instrument binary is sometimes used outside of test functions, so we set it here
    set(_INSTRUMENT_BINARY "${ROCPROFSYS_BIN_INSTALL_PREFIX}/rocprof-sys-instrument")

    # Other paths that need to be changed for install tests
    set(_ROCPROFSYS_ROCPD_RULE_PATH_PREFIX "${ROCPROFSYS_INSTALL_PREFIX}/share/rocprofiler-systems/tests")
    
    set(_RUN_PID_SCRIPT_PATH "${ROCPROFSYS_TEST_INSTALL_PREFIX}/run-rocprof-sys-pid.sh")
    set(_OUTPUT_PREFIX "@ROCPROFSYS_TEST_OUTPUT@")
    set(_OUTPUT_PATH "${_OUTPUT_PREFIX}/rocprof-sys-tests-output")
    set(_PYTHON_TEST_FILE_PREFIX "${ROCPROFSYS_SAMPLES_INSTALL_PREFIX}/python")
    set(_PYTHON_CODE_COVERAGE_FILE_PREFIX "${ROCPROFSYS_SAMPLES_INSTALL_PREFIX}/code-coverage")
    
else()
    set(_ROCPROFSYS_ROCPD_RULE_PATH_PREFIX "${CMAKE_CURRENT_LIST_DIR}")
    set(_INSTRUMENT_BINARY "$<TARGET_FILE:rocprofiler-systems-instrument>")

    set(_RUN_PID_SCRIPT_PATH "${CMAKE_CURRENT_LIST_DIR}/run-rocprof-sys-pid.sh")
    set(_OUTPUT_PREFIX "${PROJECT_BINARY_DIR}")
    set(_OUTPUT_PATH "${_OUTPUT_PREFIX}/rocprof-sys-tests-output")
    set(_PYTHON_TEST_FILE_PREFIX ${CMAKE_SOURCE_DIR}/examples/python)
    set(_PYTHON_CODE_COVERAGE_FILE_PREFIX ${CMAKE_SOURCE_DIR}/examples/code-coverage)
endif()

# -------------------------------------------------------------------------------------- #
# Helper function to map target names to their installed sample paths
# -------------------------------------------------------------------------------------- #

function(ROCPROFILER_SYSTEMS_GET_INSTALLED_SAMPLE_PATH OUTPUT_VAR TARGET_NAME)
    # Strip -exec suffix if present (for ompvv)
    if("${TARGET_NAME}" MATCHES "-exec$")
        string(REGEX REPLACE "-exec$" "" TARGET_NAME "${TARGET_NAME}")
    endif()
    
    # Map target names to their sample subdirectories based on naming conventions
    if("${TARGET_NAME}" MATCHES "^openmp-")
        set(${OUTPUT_VAR} "${ROCPROFSYS_SAMPLES_INSTALL_PREFIX}/openmp/${TARGET_NAME}" PARENT_SCOPE)
    elseif("${TARGET_NAME}" MATCHES "^parallel-overhead")
        set(${OUTPUT_VAR} "${ROCPROFSYS_SAMPLES_INSTALL_PREFIX}/parallel-overhead/${TARGET_NAME}" PARENT_SCOPE)
    elseif("${TARGET_NAME}" MATCHES "^mpi-")
        set(${OUTPUT_VAR} "${ROCPROFSYS_SAMPLES_INSTALL_PREFIX}/mpi/${TARGET_NAME}" PARENT_SCOPE)
    elseif("${TARGET_NAME}" MATCHES "^lulesh") # Lulesh bins are in causal dir
        set(${OUTPUT_VAR} "${ROCPROFSYS_SAMPLES_INSTALL_PREFIX}/causal/${TARGET_NAME}" PARENT_SCOPE)
    elseif("${TARGET_NAME}" MATCHES "^transpose")
        set(${OUTPUT_VAR} "${ROCPROFSYS_SAMPLES_INSTALL_PREFIX}/transpose/${TARGET_NAME}" PARENT_SCOPE)
    elseif("${TARGET_NAME}" MATCHES "^user-api")
        set(${OUTPUT_VAR} "${ROCPROFSYS_SAMPLES_INSTALL_PREFIX}/user-api/${TARGET_NAME}" PARENT_SCOPE)
    elseif("${TARGET_NAME}" MATCHES "^rccl-")
        set(${OUTPUT_VAR} "${ROCPROFSYS_SAMPLES_INSTALL_PREFIX}/rccl/${TARGET_NAME}" PARENT_SCOPE)
    elseif("${TARGET_NAME}" MATCHES "^code-coverage")
        set(${OUTPUT_VAR} "${ROCPROFSYS_SAMPLES_INSTALL_PREFIX}/code-coverage/${TARGET_NAME}" PARENT_SCOPE)
    elseif("${TARGET_NAME}" MATCHES "^causal-")
        set(${OUTPUT_VAR} "${ROCPROFSYS_SAMPLES_INSTALL_PREFIX}/causal/${TARGET_NAME}" PARENT_SCOPE)
    elseif("${TARGET_NAME}" MATCHES "^fork-" OR "${TARGET_NAME}" MATCHES "^hipMallocConcurrency")
        set(${OUTPUT_VAR} "${ROCPROFSYS_SAMPLES_INSTALL_PREFIX}/fork/${TARGET_NAME}" PARENT_SCOPE)
    elseif("${TARGET_NAME}" MATCHES "^python-")
        set(${OUTPUT_VAR} "${ROCPROFSYS_SAMPLES_INSTALL_PREFIX}/python/${TARGET_NAME}" PARENT_SCOPE)
    else()
        # Fallback - use target name as subdirectory
        rocprofiler_systems_message(WARNING 
            "Unknown target pattern for ${TARGET_NAME}, using fallback path mapping")
        set(${OUTPUT_VAR} "${ROCPROFSYS_SAMPLES_INSTALL_PREFIX}/${TARGET_NAME}/${TARGET_NAME}" PARENT_SCOPE)
    endif()
endfunction()

# -------------------------------------------------------------------------------------- #
# Helper function to collect all lib subdirectories paths from installed samples
# -------------------------------------------------------------------------------------- #

function(ROCPROFILER_SYSTEMS_GET_SAMPLE_LIB_PATHS OUTPUT_VAR)
    set(_lib_paths "")
    
    if(ROCPROFSYS_INSTALL_TESTS AND EXISTS "${ROCPROFSYS_SAMPLES_INSTALL_PREFIX}")
        # Get all subdirectories in the samples directory
        file(GLOB _sample_dirs 
            LIST_DIRECTORIES true 
            "${ROCPROFSYS_SAMPLES_INSTALL_PREFIX}/*")
        
        # Check each subdirectory for a lib folder
        foreach(_dir ${_sample_dirs})
            if(IS_DIRECTORY "${_dir}")
                set(_lib_dir "${_dir}/lib")
                if(IS_DIRECTORY "${_lib_dir}")
                    list(APPEND _lib_paths "${_lib_dir}")
                endif()
            endif()
        endforeach()
    endif()
    
    # Return the list of library paths found
    set(${OUTPUT_VAR} "${_lib_paths}" PARENT_SCOPE)
endfunction()

set(ROCPROFSYS_ABORT_FAIL_REGEX
    "### ERROR ###|unknown-hash=|address of faulting memory reference|exiting with non-zero exit code|terminate called after throwing an instance|calling abort.. in |Exit code: [1-9]"
    CACHE INTERNAL
    "Regex to catch abnormal exits when a PASS_REGULAR_EXPRESSION is set"
    FORCE
)

if(EXISTS /etc/os-release AND NOT IS_DIRECTORY /etc/os-release)
    file(READ /etc/os-release _OS_RELEASE_RAW)

    if(_OS_RELEASE_RAW)
        string(REPLACE "\"" "" _OS_RELEASE_RAW "${_OS_RELEASE_RAW}")
        string(REPLACE "-" " " _OS_RELEASE_RAW "${_OS_RELEASE_RAW}")
        string(
            REGEX REPLACE
            "NAME=.*\nVERSION=([0-9]+)\.([0-9]+).*\nID=([a-z]+).*"
            "\\3-\\1.\\2"
            _OS_RELEASE
            "${_OS_RELEASE_RAW}"
        )
    endif()
    unset(_OS_RELEASE_RAW)
endif()

rocprofiler_systems_message(STATUS "OS release: ${_OS_RELEASE}")

include(ProcessorCount)
if(NOT DEFINED NUM_PROCS_REAL)
    ProcessorCount(NUM_PROCS_REAL)
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

if(
    DEFINED ROCmVersion_FULL_VERSION
    AND ROCmVersion_FULL_VERSION VERSION_GREATER_EQUAL "7.0"
)
    set(ENABLE_ROCPD_TEST YES)
else()
    set(ENABLE_ROCPD_TEST NO)
endif()

rocprofiler_systems_message(
    STATUS
    "ROCm ${ROCmVersion_FULL_VERSION} - Including ROCPD Test: ${ENABLE_ROCPD_TEST}"
)

if(DEFINED ROCM_PATH)
    set(ROCM_LLVM_LIB_PATH "${ROCM_PATH}/lib/llvm/lib")
    set(_test_library_path
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${ROCM_LLVM_LIB_PATH}/:$ENV{LD_LIBRARY_PATH}"
    )
else()
    set(_test_library_path
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:$ENV{LD_LIBRARY_PATH}"
    )
endif()

if(ROCPROFSYS_INSTALL_TESTS)
    set(INSTALLED_LIBRARY_PATH "${ROCPROFSYS_INSTALL_PREFIX}/lib")
    
    # Get all sample lib directories
    rocprofiler_systems_get_sample_lib_paths(_SAMPLE_LIB_PATHS)
    
    # Append the main library path
    string(APPEND _test_library_path ":${INSTALLED_LIBRARY_PATH}")
    
    # Append each sample lib path found
    foreach(_lib_path ${_SAMPLE_LIB_PATHS})
        string(APPEND _test_library_path ":${_lib_path}")
    endforeach()
    
    if(_SAMPLE_LIB_PATHS)
        rocprofiler_systems_message(STATUS 
            "Found ${CMAKE_MATCH_COUNT} sample lib directories for LD_LIBRARY_PATH")
    endif()
endif()

set(_test_openmp_env "OMP_PROC_BIND=spread" "OMP_PLACES=threads" "OMP_NUM_THREADS=2")

set(_base_environment
    "ROCPROFSYS_TRACE=ON"
    "ROCPROFSYS_PROFILE=ON"
    "ROCPROFSYS_USE_SAMPLING=ON"
    "ROCPROFSYS_USE_PROCESS_SAMPLING=ON"
    "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_FILE_OUTPUT=ON"
    "${_test_openmp_env}"
    "${_test_library_path}"
)

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
    "${_test_library_path}"
)

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
    "${_test_library_path}"
)

set(_perfetto_environment
    "ROCPROFSYS_TRACE=ON"
    "ROCPROFSYS_PROFILE=OFF"
    "ROCPROFSYS_USE_SAMPLING=ON"
    "ROCPROFSYS_USE_PROCESS_SAMPLING=ON"
    "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_PERFETTO_BACKEND=inprocess"
    "ROCPROFSYS_PERFETTO_FILL_POLICY=ring_buffer"
    "${_test_openmp_env}"
    "${_test_library_path}"
)

set(_timemory_environment
    "ROCPROFSYS_TRACE=OFF"
    "ROCPROFSYS_PROFILE=ON"
    "ROCPROFSYS_USE_SAMPLING=ON"
    "ROCPROFSYS_USE_PROCESS_SAMPLING=ON"
    "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_TIMEMORY_COMPONENTS=wall_clock,trip_count,peak_rss"
    "${_test_openmp_env}"
    "${_test_library_path}"
)

set(_test_environment ${_base_environment})

set(_causal_environment
    "${_test_openmp_env}"
    "${_test_library_path}"
    "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_FILE_OUTPUT=ON"
    "ROCPROFSYS_CAUSAL_RANDOM_SEED=1342342"
)

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
    "PYTHONPATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_PYTHONDIR}"
)

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
    "${_test_library_path}"
)

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
    "${_test_library_path}"
)

set(_window_environment
    "ROCPROFSYS_TRACE=ON"
    "ROCPROFSYS_PROFILE=ON"
    "ROCPROFSYS_USE_SAMPLING=OFF"
    "ROCPROFSYS_USE_PROCESS_SAMPLING=OFF"
    "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_FILE_OUTPUT=ON"
    "ROCPROFSYS_VERBOSE=2"
    "${_test_openmp_env}"
    "${_test_library_path}"
)

# -------------------------------------------------------------------------------------- #

set(MPIEXEC_EXECUTABLE_ARGS)
option(
    ROCPROFSYS_CI_MPI_RUN_AS_ROOT
    "Enabled --allow-run-as-root in MPI tests with OpenMPI. Enable with care! Should only be in docker containers"
    OFF
)
mark_as_advanced(ROCPROFSYS_CI_MPI_RUN_AS_ROOT)
if(ROCPROFSYS_CI_MPI_RUN_AS_ROOT)
    execute_process(
        COMMAND ${MPIEXEC_EXECUTABLE} --allow-run-as-root --help
        RESULT_VARIABLE _mpiexec_allow_run_as_root
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(_mpiexec_allow_run_as_root EQUAL 0)
        list(APPEND MPIEXEC_EXECUTABLE_ARGS --allow-run-as-root)
    endif()
endif()

execute_process(
    COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -n 1 ls
    RESULT_VARIABLE _mpiexec_oversubscribe
    OUTPUT_QUIET
    ERROR_QUIET
)

set(rocprofiler_systems_perf_event_paranoid "4")
set(rocprofiler_systems_cap_sys_admin "1")
set(rocprofiler_systems_cap_perfmon "1")

if(EXISTS "/proc/sys/kernel/perf_event_paranoid")
    file(
        STRINGS "/proc/sys/kernel/perf_event_paranoid"
        rocprofiler_systems_perf_event_paranoid
        LIMIT_COUNT 1
    )
endif()

# TODO: This will need to be ported to runtime at some point
execute_process(
    COMMAND
        ${CMAKE_CXX_COMPILER} -O2 -g -std=c++17
        ${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-capchk.cpp -o rocprof-sys-capchk
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    RESULT_VARIABLE _capchk_compile
    OUTPUT_QUIET
    ERROR_QUIET
)

if(_capchk_compile EQUAL 0)
    execute_process(
        COMMAND ${PROJECT_BINARY_DIR}/bin/rocprof-sys-capchk CAP_SYS_ADMIN effective
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        RESULT_VARIABLE rocprofiler_systems_cap_sys_admin
        OUTPUT_QUIET
        ERROR_QUIET
    )

    execute_process(
        COMMAND ${PROJECT_BINARY_DIR}/bin/rocprof-sys-capchk CAP_PERFMON effective
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        RESULT_VARIABLE rocprofiler_systems_cap_perfmon
        OUTPUT_QUIET
        ERROR_QUIET
    )
endif()

rocprofiler_systems_message(
    STATUS "perf_event_paranoid: ${rocprofiler_systems_perf_event_paranoid}"
)
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
        PATH_SUFFIXES bin
    )
    if(ROCPROFSYS_AMD_SMI_EXE)
        execute_process(
            COMMAND ${ROCPROFSYS_AMD_SMI_EXE}
            OUTPUT_VARIABLE _AMDSMI_OUT
            ERROR_VARIABLE _AMDSMI_ERR
            RESULT_VARIABLE _AMDSMI_RET
        )
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
            AUTHOR_WARNING "amd-smi did not successfully run. Disabling GPU tests..."
        )
    endif()
endif()

set(LULESH_USE_GPU ${LULESH_USE_ROCM})
if(LULESH_USE_CUDA)
    set(LULESH_USE_GPU ON)
endif()

# -------------------------------------------------------------------------------------- #

macro(ROCPROFILER_SYSTEMS_CHECK_PASS_FAIL_REGEX NAME PASS FAIL)
    if(
        NOT "${${PASS}}" STREQUAL ""
        AND NOT "${${FAIL}}" STREQUAL ""
        AND NOT "${${FAIL}}" MATCHES "\\|ROCPROFSYS_ABORT_FAIL_REGEX"
        AND NOT "${${FAIL}}" MATCHES "${ROCPROFSYS_ABORT_FAIL_REGEX}"
    )
        rocprofiler_systems_message(
            FATAL_ERROR
            "${NAME} has set pass and fail regexes but fail regex does not include '|ROCPROFSYS_ABORT_FAIL_REGEX'"
        )
    endif()

    if("${${FAIL}}" STREQUAL "")
        set(${FAIL} "(${ROCPROFSYS_ABORT_FAIL_REGEX})")
    else()
        string(
            REPLACE
            "|ROCPROFSYS_ABORT_FAIL_REGEX"
            "|${ROCPROFSYS_ABORT_FAIL_REGEX}"
            ${FAIL}
            "${${FAIL}}"
        )
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
"
    )

    if(ROCPROFSYS_INSTALL_TESTS)
        # Point to the installed config file
        set(_CONFIG_FILE "${ROCPROFSYS_TEST_INSTALL_PREFIX}/rocprof-sys-tests-config/${_FILE}")
    endif()
    list(APPEND _ENV_CONTENTS "ROCPROFSYS_CONFIG_FILE=${_CONFIG_FILE}")
    if(_DEBUG_SETTINGS)
        list(APPEND _ENV_CONTENTS "ROCPROFSYS_DEBUG_SETTINGS=1")
    endif()
    set(${_ENV} "${_ENV_CONTENTS}" PARENT_SCOPE)
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
        PATH_SUFFIXES bin
    )

    if(rocminfo_EXECUTABLE)
        execute_process(
            COMMAND ${rocminfo_EXECUTABLE}
            RESULT_VARIABLE rocminfo_RET
            OUTPUT_VARIABLE rocminfo_OUT
            ERROR_VARIABLE rocminfo_ERR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
        )

        if(rocminfo_RET EQUAL 0)
            string(REGEX MATCHALL "gfx([0-9A-Fa-f]+)" rocminfo_GFXINFO "${rocminfo_OUT}")
            list(REMOVE_DUPLICATES rocminfo_GFXINFO)
            set(${_VAR} "${rocminfo_GFXINFO}" PARENT_SCOPE)

            if(ARG_ECHO)
                string(REPLACE ";" "${ARG_DELIM}" _GFXINFO_ECHO "${rocminfo_GFXINFO}")
                message(STATUS "${ARG_PREFIX}System architectures: ${_GFXINFO_ECHO}")
            endif()

            # Filter the architectures if a regex is provided
            if(ARG_GFX_MATCH)
                string(REGEX MATCH "${ARG_GFX_MATCH}" _GFX_MATCH "${rocminfo_GFXINFO}")
                list(REMOVE_DUPLICATES _GFX_MATCH)
                set(${_VAR} "${_GFX_MATCH}" PARENT_SCOPE)

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
        set(${_VAR} "${_timeout_v}" PARENT_SCOPE)
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
    foreach(
        _PREFIX
        SAMPLING
        RUNTIME
        REWRITE
        REWRITE_RUN
        BASELINE
        SYS_RUN
    )
        foreach(_TYPE PASS FAIL SKIP)
            list(APPEND _REGEX_OPTS "${_PREFIX}_${_TYPE}_REGEX")
        endforeach()
    endforeach()
    set(_KWARGS
        REWRITE_ARGS
        RUNTIME_ARGS
        SAMPLING_ARGS
        RUN_ARGS
        ENVIRONMENT
        LABELS
        PROPERTIES
        ${_REGEX_OPTS}
    )

    cmake_parse_arguments(
        TEST
        "SKIP_BASELINE;SKIP_SAMPLING;SKIP_REWRITE;SKIP_RUNTIME;SKIP_SYS_RUN"
        "NAME;TARGET;MPI;GPU;NUM_PROCS;SAMPLING_TIMEOUT;REWRITE_TIMEOUT;RUNTIME_TIMEOUT;SYS_RUN_TIMEOUT;WILL_FAIL;DISABLED"
        "${_KWARGS}"
        ${ARGN}
    )

    foreach(
        _PREFIX
        SAMPLING
        RUNTIME
        REWRITE
        REWRITE_RUN
        BASELINE
        SYS_RUN
    )
        if("${${_PREFIX}_FAIL_REGEX}" STREQUAL "")
            set(${_PREFIX}_FAIL_REGEX "(${ROCPROFSYS_ABORT_FAIL_REGEX})")
        endif()
    endforeach()

    if(TEST_GPU AND NOT _VALID_GPU)
        rocprofiler_systems_message(
            STATUS "${TEST_NAME} requires a GPU and no valid GPUs were found"
        )
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

    if(NOT TEST_SYS_RUN_TIMEOUT)
        set(TEST_SYS_RUN_TIMEOUT 300)
    endif()

    if(NOT TEST_DISABLED)
        set(TEST_DISABLED OFF)
    endif()

    if(NOT TEST_WILL_FAIL)
        set(TEST_WILL_FAIL OFF)
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

    if(
        "ROCPROFSYS_USE_ROCM=ON" IN_LIST TEST_ENVIRONMENT
        AND NOT "rocm" IN_LIST TEST_ENVIRONMENT
    )
        list(APPEND TEST_LABELS "rocm")
    endif()

    if(
        "ROCPROFSYS_USE_AMD_SMI=ON" IN_LIST TEST_ENVIRONMENT
        AND NOT "amd-smi" IN_LIST TEST_ENVIRONMENT
    )
        list(APPEND TEST_LABELS "amd-smi")
    endif()

    # -------------------------------------------------------------------------------------- #
    # Determine whether to create tests and which binary paths to use
    # -------------------------------------------------------------------------------------- #
    
    set(_USE_INSTALLED_PATHS FALSE)
    
    if(ROCPROFSYS_INSTALL_TESTS)
        # Only tests with "theRock" label are processed
        if("theRock" IN_LIST TEST_LABELS)
            set(_USE_INSTALLED_PATHS TRUE)
            rocprofiler_systems_message(STATUS 
                "[Install Test] Creating placeholder test for: ${TEST_NAME}")
        else()
            return()
        endif()
    elseif(NOT TARGET ${TEST_TARGET})
        return()
    endif()
    
    # -------------------------------------------------------------------------------------- #
    # Set up binary paths based on mode (build vs install)
    # -------------------------------------------------------------------------------------- #
    
    if(_USE_INSTALLED_PATHS)
        # Use installed paths (hardcoded, no generator expressions)
        rocprofiler_systems_get_installed_sample_path(_TEST_BINARY "${TEST_TARGET}")
        set(_INSTRUMENT_BINARY "${ROCPROFSYS_BIN_INSTALL_PREFIX}/rocprof-sys-instrument")
        set(_SAMPLE_BINARY "${ROCPROFSYS_BIN_INSTALL_PREFIX}/rocprof-sys-sample")
        set(_RUN_BINARY "${ROCPROFSYS_BIN_INSTALL_PREFIX}/rocprof-sys-run")
        set(_CAUSAL_BINARY "${ROCPROFSYS_BIN_INSTALL_PREFIX}/rocprof-sys-causal")
        
        # A script will replace @ROCPROFSYS_TEST_TMPDIR@ with actual path
        set(_REWRITE_OUTPUT "@ROCPROFSYS_TEST_TMPDIR@/${TEST_NAME}.inst")
        
        rocprofiler_systems_message(STATUS 
            "  Sample binary: ${_TEST_BINARY}")
        
    else()
        set(_TEST_BINARY "$<TARGET_FILE:${TEST_TARGET}>")
        set(_INSTRUMENT_BINARY "$<TARGET_FILE:rocprofiler-systems-instrument>")
        set(_SAMPLE_BINARY "$<TARGET_FILE:rocprofiler-systems-sample>")
        set(_RUN_BINARY "$<TARGET_FILE:rocprofiler-systems-run>")
        set(_CAUSAL_BINARY "$<TARGET_FILE:rocprofiler-systems-causal>")
        set(_REWRITE_OUTPUT "$<TARGET_FILE_DIR:${TEST_TARGET}>/${TEST_NAME}.inst")
    endif()
    
    set(_working_directory "${_OUTPUT_PREFIX}")
    # -------------------------------------------------------------------------------------- #
    # Now create tests using the determined paths
    # -------------------------------------------------------------------------------------- #
    
    if(DEFINED TEST_MPI AND ${TEST_MPI} AND TEST_NUM_PROCS GREATER 0)
        if(NOT TEST_NUM_PROCS GREATER NUM_PROCS_REAL)
            set(COMMAND_PREFIX
                ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_EXECUTABLE_ARGS}
                ${MPIEXEC_NUMPROC_FLAG}
                ${TEST_NUM_PROCS}
            )
            list(APPEND TEST_LABELS mpi parallel-${TEST_NUM_PROCS})
            list(APPEND TEST_PROPERTIES PARALLEL_LEVEL ${TEST_NUM_PROCS})
        else()
            set(COMMAND_PREFIX
                ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_EXECUTABLE_ARGS}
                ${MPIEXEC_NUMPROC_FLAG}
                1
            )
        endif()
    else()
        list(APPEND TEST_ENVIRONMENT "ROCPROFSYS_USE_PID=OFF")
    endif()

    if(NOT TEST_SKIP_BASELINE AND NOT ROCPROFSYS_USE_SANITIZER)
        add_test(
            NAME ${TEST_NAME}-baseline
            COMMAND ${COMMAND_PREFIX} ${_TEST_BINARY} ${TEST_RUN_ARGS}
            WORKING_DIRECTORY ${_working_directory}
        )
    endif()

    if(NOT TEST_SKIP_SAMPLING)
        add_test(
            NAME ${TEST_NAME}-sampling
            COMMAND
                ${COMMAND_PREFIX} ${_SAMPLE_BINARY}
                ${TEST_SAMPLING_ARGS} -- ${_TEST_BINARY} ${TEST_RUN_ARGS}
            WORKING_DIRECTORY ${_working_directory}
        )
    endif()

    if(NOT TEST_SKIP_REWRITE)
        add_test(
            NAME ${TEST_NAME}-binary-rewrite
            COMMAND
                ${_INSTRUMENT_BINARY} -o
                ${_REWRITE_OUTPUT}
                ${TEST_REWRITE_ARGS} -- ${_TEST_BINARY}
            WORKING_DIRECTORY ${_working_directory}
        )

        add_test(
            NAME ${TEST_NAME}-binary-rewrite-run
            COMMAND
                ${COMMAND_PREFIX} ${_RUN_BINARY} --
                ${_REWRITE_OUTPUT} ${TEST_RUN_ARGS}
            WORKING_DIRECTORY ${_working_directory}
        )

        add_test(
            NAME ${TEST_NAME}-binary-rewrite-cleanup
            COMMAND
                ${CMAKE_COMMAND} -E rm -rf
                ${_REWRITE_OUTPUT}
                ${PROJECT_BINARY_DIR}/rocprof-sys-tests-output/${TEST_NAME}-binary-rewrite
            WORKING_DIRECTORY ${_working_directory}
        )
    endif()

    if(NOT TEST_SKIP_RUNTIME AND NOT ROCPROFSYS_USE_SANITIZER)
        add_test(
            NAME ${TEST_NAME}-runtime-instrument
            COMMAND
                ${_INSTRUMENT_BINARY} ${TEST_RUNTIME_ARGS} --
                ${_TEST_BINARY} ${TEST_RUN_ARGS}
            WORKING_DIRECTORY ${_working_directory}
        )

        add_test(
            NAME ${TEST_NAME}-runtime-instrument-cleanup
            COMMAND
                ${CMAKE_COMMAND} -E rm -rf
                ${PROJECT_BINARY_DIR}/rocprof-sys-tests-output/${TEST_NAME}-runtime-instrument
            WORKING_DIRECTORY ${_working_directory}
        )
    endif()

    if(NOT TEST_SKIP_SYS_RUN)
        add_test(
            NAME ${TEST_NAME}-sys-run
            COMMAND
                ${COMMAND_PREFIX} ${_RUN_BINARY} --
                ${_TEST_BINARY} ${TEST_RUN_ARGS}
            WORKING_DIRECTORY ${_working_directory}
        )
    endif()

    if(TEST ${TEST_NAME}-binary-rewrite-run)
        set_tests_properties(
            ${TEST_NAME}-binary-rewrite-run
            PROPERTIES DEPENDS ${TEST_NAME}-binary-rewrite
        )
    endif()

    foreach(
        _TEST
        baseline
        sampling
        binary-rewrite
        binary-rewrite-run
        binary-rewrite-cleanup
        runtime-instrument
        runtime-instrument-cleanup
        sys-run
    )
        string(
            REGEX REPLACE
            "rewrite-run(-|/)"
            "rewrite\\1"
            _prefix
            "${TEST_NAME}-${_TEST}/"
        )
        set(_labels "${_TEST}")
        string(REPLACE "rewrite-run" "rewrite" _labels "${_TEST}")
        if(TEST_TARGET)
            list(APPEND _labels "${TEST_TARGET}")
        endif()
        if(TEST_LABELS)
            list(APPEND _labels "${TEST_LABELS}")
        endif()

        set(_output_path "${_OUTPUT_PATH}")

        set(_environ
            "ROCPROFSYS_DEFAULT_MIN_INSTRUCTIONS=64"
            "${TEST_ENVIRONMENT}"
            "ROCPROFSYS_OUTPUT_PATH=${_output_path}"
            "ROCPROFSYS_OUTPUT_PREFIX=${_prefix}"
        )

        set(_timeout ${TEST_REWRITE_TIMEOUT})
        if("${_TEST}" MATCHES "sampling")
            set(_timeout ${TEST_SAMPLING_TIMEOUT})
        elseif("${_TEST}" MATCHES "runtime-instrument")
            set(_timeout ${TEST_RUNTIME_TIMEOUT})
        elseif("${_TEST}" MATCHES "sys-run")
            set(_timeout ${TEST_SYS_RUN_TIMEOUT})
        endif()

        set(_props)
        if("${_TEST}" MATCHES "sys-run|sampling|baseline")
            set(_props ${TEST_PROPERTIES})
            if(NOT "RUN_SERIAL" IN_LIST _props)
                list(APPEND _props RUN_SERIAL ON)
            endif()
        endif()

        if("${_TEST}" MATCHES "-cleanup$")
            set(_REGEX_VAR)
        elseif("${_TEST}" MATCHES "binary-rewrite-run")
            set(_REGEX_VAR REWRITE_RUN)
        elseif("${_TEST}" MATCHES "runtime-instrument")
            set(_REGEX_VAR RUNTIME)
        elseif("${_TEST}" MATCHES "binary-rewrite")
            set(_REGEX_VAR REWRITE)
        elseif("${_TEST}" MATCHES "baseline")
            set(_REGEX_VAR BASELINE)
        elseif("${_TEST}" MATCHES "sampling")
            set(_REGEX_VAR SAMPLING)
        elseif("${_TEST}" MATCHES "sys-run")
            set(_REGEX_VAR SYS_RUN)
        else()
            set(_REGEX_VAR)
        endif()

        if(
            "${_TEST}"
                MATCHES
                "binary-rewrite-run|runtime-instrument|sampling|sys-run"
        )
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
            "${_PASS_REGEX}" "${_FAIL_REGEX}"
        )
        if(TEST ${TEST_NAME}-${_TEST})
            rocprofiler_systems_write_test_config(${TEST_NAME}-${_TEST}.cfg _environ)
            set_tests_properties(
                ${TEST_NAME}-${_TEST}
                PROPERTIES
                    ENVIRONMENT "${_environ}"
                    TIMEOUT ${_timeout}
                    LABELS "${_labels}"
                    PASS_REGULAR_EXPRESSION "${${_PASS_REGEX}}"
                    FAIL_REGULAR_EXPRESSION "${${_FAIL_REGEX}}"
                    SKIP_REGULAR_EXPRESSION "${${_SKIP_REGEX}}"
                    WILL_FAIL ${TEST_WILL_FAIL}
                    DISABLED ${TEST_DISABLED}
                    FIXTURES_REQUIRED rocprofsys-global-tmp-files
                    ${_props}
            )

            if("${_TEST}" STREQUAL "binary-rewrite")
                set_tests_properties(
                    ${TEST_NAME}-${_TEST}
                    PROPERTIES FIXTURES_SETUP ${TEST_NAME}-binary-rewrite-fixture
                )
            elseif("${_TEST}" STREQUAL "binary-rewrite-run")
                set_tests_properties(
                    ${TEST_NAME}-${_TEST}
                    PROPERTIES
                        FIXTURES_REQUIRED
                            "rocprofsys-global-tmp-files;${TEST_NAME}-binary-rewrite-fixture"
                )
            elseif("${_TEST}" STREQUAL "binary-rewrite-cleanup")
                set_tests_properties(
                    ${TEST_NAME}-${_TEST}
                    PROPERTIES FIXTURES_CLEANUP ${TEST_NAME}-binary-rewrite-fixture
                )
            elseif("${_TEST}" STREQUAL "runtime-instrument")
                set_tests_properties(
                    ${TEST_NAME}-${_TEST}
                    PROPERTIES FIXTURES_SETUP ${TEST_NAME}-runtime-instrument-fixture
                )
            elseif("${_TEST}" STREQUAL "runtime-instrument-cleanup")
                set_tests_properties(
                    ${TEST_NAME}-${_TEST}
                    PROPERTIES
                        FIXTURES_CLEANUP ${TEST_NAME}-runtime-instrument-fixture
                )
            endif()
        endif()
    endforeach()
endfunction()

# -------------------------------------------------------------------------------------- #

function(ROCPROFILER_SYSTEMS_ADD_CAUSAL_TEST)
    foreach(_PREFIX CAUSAL CAUSAL_VALIDATE)
        foreach(_TYPE PASS FAIL SKIP)
            list(APPEND _REGEX_OPTS "${_PREFIX}_${_TYPE}_REGEX")
        endforeach()
    endforeach()

    set(_KWARGS
        CAUSAL_ARGS
        CAUSAL_VALIDATE_ARGS
        RUN_ARGS
        ENVIRONMENT
        LABELS
        PROPERTIES
        ${_REGEX_OPTS}
    )

    cmake_parse_arguments(
        TEST
        "SKIP_BASELINE"
        "NAME;TARGET;CAUSAL_MODE;CAUSAL_TIMEOUT;CAUSAL_VALIDATE_TIMEOUT"
        "${_KWARGS}"
        ${ARGN}
    )

    if(NOT DEFINED TEST_CAUSAL_MODE)
        rocprofiler_systems_message(FATAL_ERROR
            "${TEST_NAME} :: CAUSAL_MODE must be defined"
        )
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

    # -------------------------------------------------------------------------------------- #
    # Determine whether to create tests and which binary paths to use
    # -------------------------------------------------------------------------------------- #
    
    set(_USE_INSTALLED_PATHS FALSE)
    
    if(ROCPROFSYS_INSTALL_TESTS)
        # Only tests with "theRock" label are processed
        if("theRock" IN_LIST TEST_LABELS)
            set(_USE_INSTALLED_PATHS TRUE)
            rocprofiler_systems_message(STATUS 
                "[Install Test] Creating placeholder causal test for: ${TEST_NAME}")
        else()
            return()
        endif()
    endif()

    # -------------------------------------------------------------------------------------- #
    # Set up binary paths based on mode (build vs install)
    # -------------------------------------------------------------------------------------- #
    
    if(_USE_INSTALLED_PATHS)
        # Use installed paths (hardcoded, no generator expressions)
        rocprofiler_systems_get_installed_sample_path(_TEST_BINARY "${TEST_TARGET}")
        set(_CAUSAL_BINARY "${ROCPROFSYS_BIN_INSTALL_PREFIX}/rocprof-sys-causal")
        
        rocprofiler_systems_message(STATUS 
            "  Causal test sample binary: ${_TEST_BINARY}")
    elseif(TARGET ${TEST_TARGET})
        set(_TEST_BINARY "$<TARGET_FILE:${TEST_TARGET}>")
        set(_CAUSAL_BINARY "$<TARGET_FILE:rocprofiler-systems-causal>")
    else()
        return()
    endif()

    set(_working_directory "${_OUTPUT_PREFIX}")

    # -------------------------------------------------------------------------------------- #
    # Now create tests using the determined paths
    # -------------------------------------------------------------------------------------- #

    set(COMMAND_PREFIX
        ${_CAUSAL_BINARY}
        --reset
        -m
        ${TEST_CAUSAL_MODE}
        ${TEST_CAUSAL_ARGS}
        --
    )

    if(NOT TEST_SKIP_BASELINE)
        add_test(
            NAME ${TEST_NAME}-baseline
            COMMAND ${_TEST_BINARY} ${TEST_RUN_ARGS}
            WORKING_DIRECTORY ${_working_directory}
        )
    endif()

    add_test(
        NAME causal-${TEST_NAME}
        COMMAND ${COMMAND_PREFIX} ${_TEST_BINARY} ${TEST_RUN_ARGS}
        WORKING_DIRECTORY ${_working_directory}
    )

        if(NOT "${TEST_CAUSAL_VALIDATE_ARGS}" STREQUAL "")
            if(
                "$ENV{ROCPROFSYS_CI}" MATCHES "ON|on|1|true|TRUE"
                OR "$ENV{CI}" MATCHES "true"
                OR NOT "$ENV{GITHUB_RUN_ID}" STREQUAL ""
            )
                set(_VALIDATE_EXTRA "--ci")
            else()
                set(_VALIDATE_EXTRA "")
            endif()

            add_test(
                NAME validate-causal-${TEST_NAME}
                COMMAND
                    ${CMAKE_CURRENT_LIST_DIR}/validate-causal-json.py ${_VALIDATE_EXTRA}
                    ${TEST_CAUSAL_VALIDATE_ARGS}
                WORKING_DIRECTORY ${_working_directory}
            )
        endif()

        if(TEST validate-causal-${TEST_NAME})
            set_tests_properties(
                validate-causal-${TEST_NAME}
                PROPERTIES DEPENDS causal-${TEST_NAME}
            )
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

            set(_output_path "${_OUTPUT_PATH}")
            set(_environ
                "${_causal_environment}"
                "ROCPROFSYS_OUTPUT_PATH=${_output_path}"
                "ROCPROFSYS_OUTPUT_PREFIX=${_prefix}"
                "ROCPROFSYS_CI=ON"
                "ROCPROFSYS_USE_PID=OFF"
                "ROCPROFSYS_THREAD_POOL_SIZE=0"
                "ROCPROFSYS_VERBOSE=1"
                "ROCPROFSYS_DL_VERBOSE=0"
                "ROCPROFSYS_DEBUG_SETTINGS=0"
                "${TEST_ENVIRONMENT}"
            )

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
                "${_FAIL_REGEX}"
            )
            set_tests_properties(
                ${_NAME}
                PROPERTIES
                    ENVIRONMENT "${_environ}"
                    TIMEOUT ${_timeout}
                    LABELS "${_labels}"
                    PASS_REGULAR_EXPRESSION "${${_PASS_REGEX}}"
                    FAIL_REGULAR_EXPRESSION "${${_FAIL_REGEX}}"
                    SKIP_REGULAR_EXPRESSION "${${_SKIP_REGEX}}"
                    FIXTURES_REQUIRED rocprofsys-global-tmp-files
                    ${_props}
            )
        endforeach()
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
        "PROFILE_ARGS;RUN_ARGS;ENVIRONMENT;LABELS;PROPERTIES;PASS_REGEX;FAIL_REGEX;SKIP_REGEX;DEPENDS;COMMAND;FIXTURES_REQUIRED" # multiple
        # value args
        ${ARGN}
    )

    set(_USE_INSTALLED_PATHS FALSE)
    if(ROCPROFSYS_INSTALL_TESTS)
        # Only tests with "theRock" label are processed
        if("theRock" IN_LIST TEST_LABELS)
            set(_USE_INSTALLED_PATHS TRUE)
            rocprofiler_systems_message(STATUS 
                "[Install Test] Creating placeholder python test for: ${TEST_NAME}")
        else()
            return()
        endif()
    endif()

    # -------------------------------------------------------------------------------------- #
    # Set up output paths based on mode (build vs install)
    # -------------------------------------------------------------------------------------- #
    
    if(_USE_INSTALLED_PATHS)
        set(_output_path "${_OUTPUT_PATH}")
        set(_working_directory "${_OUTPUT_PREFIX}")
    else()
        set(_output_path "${PROJECT_BINARY_DIR}/rocprof-sys-tests-output")
        set(_working_directory "${PROJECT_BINARY_DIR}")
    endif()

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
        list(
            APPEND TEST_ENVIRONMENT
            "ROCPROFSYS_CI=ON"
            "ROCPROFSYS_OUTPUT_PATH=${_output_path}"
        )
        get_filename_component(_TEST_FILE "${TEST_FILE}" NAME)
        set(_TEST_FILE
            ${PROJECT_BINARY_DIR}/python/tests/${TEST_PYTHON_VERSION}/${_TEST_FILE}
        )

        if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU})
            list(APPEND TEST_LABELS "rocpd")
            list(APPEND TEST_ENVIRONMENT "ROCPROFSYS_USE_ROCPD=ON")
        endif()

        configure_file(${TEST_FILE} ${_TEST_FILE} @ONLY)

        if(TEST_STANDALONE)
            add_test(
                NAME ${TEST_NAME}-${TEST_PYTHON_VERSION}
                COMMAND ${TEST_PYTHON_EXECUTABLE} ${_TEST_FILE} ${TEST_RUN_ARGS}
                WORKING_DIRECTORY ${_working_directory}
            )
        else()
            add_test(
                NAME ${TEST_NAME}-${TEST_PYTHON_VERSION}
                COMMAND
                    ${TEST_PYTHON_EXECUTABLE} -m rocprofsys ${TEST_PROFILE_ARGS} --
                    ${_TEST_FILE} ${TEST_RUN_ARGS}
                WORKING_DIRECTORY ${_working_directory}
            )
            add_test(
                NAME ${TEST_NAME}-${TEST_PYTHON_VERSION}-annotated
                COMMAND
                    ${TEST_PYTHON_EXECUTABLE} -m rocprofsys ${TEST_PROFILE_ARGS}
                    --annotate-trace -- ${_TEST_FILE} ${TEST_RUN_ARGS}
                WORKING_DIRECTORY ${_working_directory}
            )
        endif()
    else()
        list(APPEND TEST_LABELS "python-check" "python-${TEST_PYTHON_VERSION}-check")
        
        # For install tests, substitute hardcoded relative paths with absolute paths
        if(ROCPROFSYS_INSTALL_TESTS)
            set(_substituted_command "")
            foreach(_arg IN LISTS TEST_COMMAND)
                # Replace "rocprof-sys-tests-output" with the install test output path
                string(REPLACE "rocprof-sys-tests-output" "${_output_path}" _new_arg "${_arg}")
                list(APPEND _substituted_command "${_new_arg}")
            endforeach()
            
            set(_substituted_file "${TEST_FILE}")
            string(REPLACE "rocprof-sys-tests-output" "${_output_path}" _substituted_file "${_substituted_file}")
        else()
            set(_substituted_command "${TEST_COMMAND}")
            set(_substituted_file "${TEST_FILE}")
        endif()
        
        add_test(
            NAME ${TEST_NAME}-${TEST_PYTHON_VERSION}
            COMMAND ${_substituted_command} ${_substituted_file}
            WORKING_DIRECTORY ${_working_directory}
        )
    endif()

    foreach(
        _TEST
        ${TEST_NAME}-${TEST_PYTHON_VERSION}
        ${TEST_NAME}-${TEST_PYTHON_VERSION}-annotated
    )
        if(NOT TEST "${_TEST}")
            continue()
        endif()

        string(
            REPLACE
            "${TEST_NAME}-${TEST_PYTHON_VERSION}"
            "${TEST_NAME}"
            _TEST_DIR
            "${_TEST}"
        )
        set(_TEST_ENV
            "${TEST_ENVIRONMENT}"
            "ROCPROFSYS_OUTPUT_PREFIX=${_TEST_DIR}/${TEST_PYTHON_VERSION}/"
            "ROCPROFSYS_CI_TIMEOUT=${TEST_TIMEOUT}"
        )

        set(_TEST_PROPERTIES "${TEST_PROPERTIES}")
        # assign pass variable to pass regex
        set(_PASS_REGEX TEST_PASS_REGEX)
        # assign fail variable to fail regex
        set(_FAIL_REGEX TEST_FAIL_REGEX)

        rocprofiler_systems_check_pass_fail_regex("${_TEST}" "${_PASS_REGEX}"
            "${_FAIL_REGEX}"
        )

        set(_PYTHON_FIXTURES "rocprofsys-global-tmp-files")
        if(TEST_FIXTURES_REQUIRED)
            list(APPEND _PYTHON_FIXTURES "${TEST_FIXTURES_REQUIRED}")
        endif()

        set_tests_properties(
            ${_TEST}
            PROPERTIES
                ENVIRONMENT "${_TEST_ENV}"
                TIMEOUT ${TEST_TIMEOUT}
                LABELS "${TEST_LABELS}"
                DEPENDS "${TEST_DEPENDS}"
                PASS_REGULAR_EXPRESSION "${${_PASS_REGEX}}"
                FAIL_REGULAR_EXPRESSION "${${_FAIL_REGEX}}"
                SKIP_REGULAR_EXPRESSION "${TEST_SKIP_REGEX}"
                REQUIRED_FILES "${TEST_FILE}"
                FIXTURES_REQUIRED "${_PYTHON_FIXTURES}"
                ${_TEST_PROPERTIES}
        )
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
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import perfetto"
            RESULT_VARIABLE ROCPROFSYS_VALIDATION_PYTHON_PERFETTO
        )

        if(NOT ROCPROFSYS_VALIDATION_PYTHON_PERFETTO EQUAL 0)
            rocprofiler_systems_message(AUTHOR_WARNING
                "Python3 found but perfetto support is disabled"
            )
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
            COMPONENTS Interpreter
        )

        if(_PYTHON_EXECUTABLE)
            set(ROCPROFSYS_VALIDATION_PYTHON ${_PYTHON_EXECUTABLE})
            execute_process(
                COMMAND ${_PYTHON_EXECUTABLE} -c "import perfetto"
                RESULT_VARIABLE ROCPROFSYS_VALIDATION_PYTHON_PERFETTO
            )

            # prefer Python3 with perfetto support
            if(ROCPROFSYS_VALIDATION_PYTHON_PERFETTO EQUAL 0)
                break()
            else()
                rocprofiler_systems_message(
                    AUTHOR_WARNING
                    "${_PYTHON_EXECUTABLE} found but perfetto support is disabled"
                )
            endif()
        endif()

        math(EXPR _INDEX "${_INDEX} + 1")
    endforeach()
endif()

if(NOT ROCPROFSYS_VALIDATION_PYTHON)
    rocprofiler_systems_message(
        AUTHOR_WARNING "Python3 interpreter not found. Validation tests will be disabled"
    )
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
        "NAME;TIMEOUT;TIMEMORY_METRIC;TIMEMORY_FILE;PERFETTO_METRIC;PERFETTO_FILE;ROCPD_FILE"
        "ENVIRONMENT;LABELS;PROPERTIES;PASS_REGEX;FAIL_REGEX;SKIP_REGEX;DEPENDS;EXIST_FILES;ARGS"
        ${ARGN}
    )

    if(NOT TEST "${TEST_NAME}")
        rocprofiler_systems_message(
            AUTHOR_WARNING
            "No validation test(s) for ${TEST_NAME} because test does not exist"
        )
        return()
    endif()
    
    # Skip if parent test doesn't have theRock label in install mode
    if(ROCPROFSYS_INSTALL_TESTS)
        get_test_property(${TEST_NAME} LABELS PARENT_TEST_LABELS)
        if(NOT "theRock" IN_LIST PARENT_TEST_LABELS)
            return()
        endif()
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

    # Add labels from the parent test to the validation test
    get_test_property(${TEST_NAME} LABELS PARENT_TEST_LABELS)
    if(PARENT_TEST_LABELS)
        list(APPEND TEST_LABELS "${PARENT_TEST_LABELS}")
        list(REMOVE_DUPLICATES TEST_LABELS)
        list(SORT TEST_LABELS)
    endif()

    if(NOT TEST_PASS_REGEX)
        set(TEST_PASS_REGEX
            "rocprof-sys-tests-output/${TEST_NAME}/(${TEST_TIMEMORY_FILE}|${TEST_PERFETTO_FILE}|${TEST_ROCPD_FILE}) validated"
        )
    endif()
    
    # Determine output path based on mode
    set(_output_path "${_OUTPUT_PATH}/${TEST_NAME}")
    set(_working_directory "${_OUTPUT_PREFIX}")
    if(ROCPROFSYS_INSTALL_TESTS)
        # For install tests, use placeholder that will be replaced
        set(_VALIDATION_SCRIPT_PATH "${ROCPROFSYS_TEST_INSTALL_PREFIX}")
    else()
        set(_VALIDATION_SCRIPT_PATH "${CMAKE_CURRENT_LIST_DIR}")
    endif()

    foreach(_FILE ${TEST_EXIST_FILES})
        add_test(
            NAME validate-${TEST_NAME}-${_FILE}-exists
            COMMAND
                test -e
                ${_output_path}/${_FILE}
            WORKING_DIRECTORY ${_working_directory}
        )
    endforeach()

    if(TEST_TIMEMORY_FILE AND NOT ROCPROFSYS_INSTALL_TESTS)
        add_test(
            NAME validate-${TEST_NAME}-timemory
            COMMAND
                ${ROCPROFSYS_VALIDATION_PYTHON}
                ${_VALIDATION_SCRIPT_PATH}/validate-timemory-json.py -m
                "${TEST_TIMEMORY_METRIC}" ${TEST_ARGS} -i
                ${_output_path}/${TEST_TIMEMORY_FILE}
            WORKING_DIRECTORY ${_working_directory}
        )
    endif()

    if(ROCPROFSYS_VALIDATION_PYTHON_PERFETTO EQUAL 0 AND TEST_PERFETTO_FILE)
        add_test(
            NAME validate-${TEST_NAME}-perfetto
            COMMAND
                ${ROCPROFSYS_VALIDATION_PYTHON}
                ${_VALIDATION_SCRIPT_PATH}/validate-perfetto-proto.py -m
                "${TEST_PERFETTO_METRIC}" ${TEST_ARGS} -i
                ${_output_path}/${TEST_PERFETTO_FILE}
                -t /opt/trace_processor/bin/trace_processor_shell
            WORKING_DIRECTORY ${_working_directory}
        )
    endif()

    if(TEST_ROCPD_FILE)
        add_test(
            NAME validate-${TEST_NAME}-rocpd
            COMMAND
                ${ROCPROFSYS_VALIDATION_PYTHON}
                ${_VALIDATION_SCRIPT_PATH}/validate-rocpd.py -db
                ${_output_path}/${TEST_ROCPD_FILE}
                ${TEST_ARGS}
            WORKING_DIRECTORY ${_working_directory}
        )
    endif()

    list(APPEND TEST_ENVIRONMENT "ROCPROFSYS_CI_TIMEOUT=${TEST_TIMEOUT}")

    # Determine the cleanup fixture this validation test should require
    set(_VALIDATION_FIXTURES "rocprofsys-global-tmp-files")
    if("${TEST_NAME}" MATCHES "-binary-rewrite(-run)?$")
        string(REGEX REPLACE "-binary-rewrite(-run)?$" "" _BASE_TEST_NAME "${TEST_NAME}")
        list(APPEND _VALIDATION_FIXTURES "${_BASE_TEST_NAME}-binary-rewrite-fixture")
    elseif("${TEST_NAME}" MATCHES "-runtime-instrument$")
        string(REGEX REPLACE "-runtime-instrument$" "" _BASE_TEST_NAME "${TEST_NAME}")
        list(APPEND _VALIDATION_FIXTURES "${_BASE_TEST_NAME}-runtime-instrument-fixture")
    endif()

    foreach(
        _TEST
        validate-${TEST_NAME}-timemory
        validate-${TEST_NAME}-perfetto
        validate-${TEST_NAME}-rocpd
    )
        # Skip tests that don't exist
        if(NOT TEST "${_TEST}")
            continue()
        endif()

        # Skip timemory validation if no timemory file is specified
        if("${_TEST}" MATCHES "-timemory" AND NOT TEST_TIMEMORY_FILE)
            continue()
        endif()

        # Skip perfetto validation if no perfetto file is specified
        if("${_TEST}" MATCHES "-perfetto" AND NOT TEST_PERFETTO_FILE)
            continue()
        endif()

        # Skip rocpd validation if no rocpd file is specified
        if("${_TEST}" MATCHES "-rocpd" AND NOT TEST_ROCPD_FILE)
            continue()
        endif()

        rocprofiler_systems_check_pass_fail_regex("${_TEST}" "TEST_PASS_REGEX"
            "TEST_FAIL_REGEX"
        )
        set_tests_properties(
            ${_TEST}
            PROPERTIES
                ENVIRONMENT "${TEST_ENVIRONMENT}"
                TIMEOUT ${TEST_TIMEOUT}
                LABELS "${TEST_LABELS}"
                DEPENDS "${TEST_DEPENDS};${TEST_NAME}"
                PASS_REGULAR_EXPRESSION "${TEST_PASS_REGEX}"
                FAIL_REGULAR_EXPRESSION "${TEST_FAIL_REGEX}"
                SKIP_REGULAR_EXPRESSION "${TEST_SKIP_REGEX}"
                REQUIRED_FILES "${TEST_FILE}"
                FIXTURES_REQUIRED "${_VALIDATION_FIXTURES}"
                ${TEST_PROPERTIES}
        )
    endforeach()
endfunction()

# -------------------------------------------------------------------------------------- #
#
# Adds a ctest for executables
#
# -------------------------------------------------------------------------------------- #

function(ROCPROFILER_SYSTEMS_ADD_BIN_TEST)
    cmake_parse_arguments(
        TEST
        "" # options
        "NAME;TARGET;TIMEOUT;WORKING_DIRECTORY" # single value args
        "ARGS;ENVIRONMENT;LABELS;PROPERTIES;PASS_REGEX;FAIL_REGEX;SKIP_REGEX;DEPENDS;COMMAND" # multiple
        # value args
        ${ARGN}
    )

    # -------------------------------------------------------------------------------------- #
    # Determine whether to create tests and which binary paths to use
    # -------------------------------------------------------------------------------------- #
    
    set(_USE_INSTALLED_PATHS FALSE)
    
    if(ROCPROFSYS_INSTALL_TESTS)
        # Only tests with "theRock" label are processed
        if("theRock" IN_LIST TEST_LABELS)
            set(_USE_INSTALLED_PATHS TRUE)
            rocprofiler_systems_message(STATUS 
                "[Install Test] Creating placeholder bin test for: ${TEST_NAME}")
        else()
            return()
        endif()
    elseif(NOT TEST_COMMAND AND NOT TARGET ${TEST_TARGET})
        if(ROCPROFSYS_BUILD_TESTING)
            message(FATAL_ERROR "Error! ${TEST_TARGET} does not exist")
        else()
            return()
        endif()
    endif()

    if(NOT TEST_WORKING_DIRECTORY)
        set(TEST_WORKING_DIRECTORY ${_OUTPUT_PREFIX})
    endif()

    if(NOT TEST_ENVIRONMENT)
        set(TEST_ENVIRONMENT
            "ROCPROFSYS_TRACE=ON"
            "ROCPROFSYS_PROFILE=ON"
            "ROCPROFSYS_USE_SAMPLING=ON"
            "ROCPROFSYS_TIME_OUTPUT=OFF"
            "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:$ENV{LD_LIBRARY_PATH}" # TODO: fix the path here
        )
    endif()

    # -------------------------------------------------------------------------------------- #
    # Set up binary paths based on mode (build vs install)
    # -------------------------------------------------------------------------------------- #
    set(_output_path "${_OUTPUT_PATH}")
    if(_USE_INSTALLED_PATHS)
        # Use installed paths for tool binaries (not samples)
        if(TEST_TARGET)
            # Map tool target names to their installed binary paths
            if("${TEST_TARGET}" STREQUAL "rocprofiler-systems-instrument")
                set(_TEST_BINARY "${ROCPROFSYS_BIN_INSTALL_PREFIX}/rocprof-sys-instrument")
            elseif("${TEST_TARGET}" STREQUAL "rocprofiler-systems-avail")
                set(_TEST_BINARY "${ROCPROFSYS_BIN_INSTALL_PREFIX}/rocprof-sys-avail")
            elseif("${TEST_TARGET}" STREQUAL "rocprofiler-systems-run")
                set(_TEST_BINARY "${ROCPROFSYS_BIN_INSTALL_PREFIX}/rocprof-sys-run")
            elseif("${TEST_TARGET}" STREQUAL "rocprofiler-systems-sample")
                set(_TEST_BINARY "${ROCPROFSYS_BIN_INSTALL_PREFIX}/rocprof-sys-sample")
            elseif("${TEST_TARGET}" STREQUAL "rocprofiler-systems-causal")
                set(_TEST_BINARY "${ROCPROFSYS_BIN_INSTALL_PREFIX}/rocprof-sys-causal")
            else()
                # For other targets, try sample path mapping
                rocprofiler_systems_get_installed_sample_path(_TEST_BINARY "${TEST_TARGET}")
            endif()
            
            rocprofiler_systems_message(STATUS 
                "  Bin test binary: ${_TEST_BINARY}")
        endif()
    else()
        # Use build paths (with generator expressions)
        if(TEST_TARGET)
            set(_TEST_BINARY "$<TARGET_FILE:${TEST_TARGET}>")
        endif()
    endif()

    # common
    list(
        APPEND TEST_ENVIRONMENT
        "ROCPROFSYS_CI=ON"
        "ROCPROFSYS_CI_TIMEOUT=${TEST_TIMEOUT}"
        "ROCPROFSYS_CONFIG_FILE="
        "ROCPROFSYS_OUTPUT_PATH=${_output_path}"
        "TWD=${TEST_WORKING_DIRECTORY}"
    )
    # copy for inverse
    set(TEST_ENVIRONMENT_INV "${TEST_ENVIRONMENT}")

    # different for regular test and inverse test
    list(APPEND TEST_ENVIRONMENT "ROCPROFSYS_OUTPUT_PREFIX=${TEST_NAME}/")
    list(APPEND TEST_ENVIRONMENT_INV "ROCPROFSYS_OUTPUT_PREFIX=${TEST_NAME}-inverse/")

    if(
        NOT "${TEST_PASS_REGEX}" STREQUAL ""
        AND NOT "${TEST_FAIL_REGEX}" STREQUAL ""
        AND NOT "${TEST_FAIL_REGEX}" MATCHES "\\|ROCPROFSYS_ABORT_FAIL_REGEX"
    )
        rocprofiler_systems_message(
            FATAL_ERROR
            "${TEST_NAME} has set pass and fail regexes but fail regex does not include '|ROCPROFSYS_ABORT_FAIL_REGEX'"
        )
    endif()

    if("${TEST_FAIL_REGEX}" STREQUAL "")
        set(TEST_FAIL_REGEX "(${ROCPROFSYS_ABORT_FAIL_REGEX})")
    else()
        string(
            REPLACE
            "|ROCPROFSYS_ABORT_FAIL_REGEX"
            "|${ROCPROFSYS_ABORT_FAIL_REGEX}"
            TEST_FAIL_REGEX
            "${TEST_FAIL_REGEX}"
        )
    endif()

    # -------------------------------------------------------------------------------------- #
    # Create the test using determined paths
    # -------------------------------------------------------------------------------------- #

    if(TEST_COMMAND)
        # For install tests, substitute hardcoded relative paths with absolute paths
        if(ROCPROFSYS_INSTALL_TESTS)
            # Substitute in both COMMAND and ARGS
            set(_substituted_command "")
            foreach(_arg IN LISTS TEST_COMMAND)
                # Replace "rocprof-sys-tests-output" with the install test output path
                string(REPLACE "rocprof-sys-tests-output" "${_output_path}" _new_arg "${_arg}")
                list(APPEND _substituted_command "${_new_arg}")
            endforeach()
            
            set(_substituted_args "")
            foreach(_arg IN LISTS TEST_ARGS)
                # Replace "rocprof-sys-tests-output" with the install test output path
                string(REPLACE "rocprof-sys-tests-output" "${_output_path}" _new_arg "${_arg}")
                list(APPEND _substituted_args "${_new_arg}")
            endforeach()
        else()
            # For build tests, use args as-is (relative paths are correct)
            set(_substituted_command "${TEST_COMMAND}")
            set(_substituted_args "${TEST_ARGS}")
        endif()
        
        add_test(
            NAME ${TEST_NAME}
            COMMAND ${_substituted_command} ${_substituted_args}
            WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY}
        )

        set_tests_properties(
            ${TEST_NAME}
            PROPERTIES
                ENVIRONMENT "${TEST_ENVIRONMENT}"
                TIMEOUT ${TEST_TIMEOUT}
                DEPENDS "${TEST_DEPENDS}"
                LABELS "rocprofiler-systems-bin;${TEST_LABELS}"
                PASS_REGULAR_EXPRESSION "${TEST_PASS_REGEX}"
                FAIL_REGULAR_EXPRESSION "${TEST_FAIL_REGEX}"
                SKIP_REGULAR_EXPRESSION "${TEST_SKIP_REGEX}"
                FIXTURES_REQUIRED rocprofsys-global-tmp-files
                ${TEST_PROPERTIES}
        )
    elseif(_TEST_BINARY)
        add_test(
            NAME ${TEST_NAME}
            COMMAND ${_TEST_BINARY} ${TEST_ARGS}
            WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY}
        )

        set_tests_properties(
            ${TEST_NAME}
            PROPERTIES
                ENVIRONMENT "${TEST_ENVIRONMENT}"
                TIMEOUT ${TEST_TIMEOUT}
                DEPENDS "${TEST_DEPENDS}"
                LABELS "rocprofiler-systems-bin;${TEST_LABELS}"
                PASS_REGULAR_EXPRESSION "${TEST_PASS_REGEX}"
                FAIL_REGULAR_EXPRESSION "${TEST_FAIL_REGEX}"
                SKIP_REGULAR_EXPRESSION "${TEST_SKIP_REGEX}"
                FIXTURES_REQUIRED rocprofsys-global-tmp-files
                ${TEST_PROPERTIES}
        )
    endif()
endfunction()
