# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
get_filename_component(_prefix "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)

# Resolve existing symlink ancestors before checking a not-yet-created result
# directory. Creating it first would already mutate an immutable prefix.
function(resolve_output_path output path)
    set(_ancestor "${path}")
    set(_suffix "")
    while(NOT EXISTS "${_ancestor}")
        get_filename_component(_name "${_ancestor}" NAME)
        get_filename_component(_parent "${_ancestor}" DIRECTORY)
        if(_parent STREQUAL _ancestor OR IS_SYMLINK "${_ancestor}")
            message(FATAL_ERROR "Unresolvable results path: ${path}")
        endif()
        set(_suffix "/${_name}${_suffix}")
        set(_ancestor "${_parent}")
    endwhile()
    if(NOT IS_DIRECTORY "${_ancestor}")
        message(FATAL_ERROR "Results ancestor is not a directory: ${_ancestor}")
    endif()
    file(REAL_PATH "${_ancestor}" _resolved)
    set(_resolved "${_resolved}${_suffix}")
    cmake_path(NORMAL_PATH _resolved)
    set(${output} "${_resolved}" PARENT_SCOPE)
endfunction()

if(NOT DEFINED RESULTS_DIRECTORY OR NOT IS_ABSOLUTE "${RESULTS_DIRECTORY}")
    message(
        FATAL_ERROR
        "RESULTS_DIRECTORY must be an absolute path to a fresh results directory"
    )
endif()
resolve_output_path(_results "${RESULTS_DIRECTORY}")
cmake_path(IS_PREFIX _prefix "${_results}" NORMALIZE _inside_prefix)
if(_inside_prefix)
    message(
        FATAL_ERROR
        "RESULTS_DIRECTORY must be outside the completed test prefix"
    )
endif()
if(DEFINED RUNTIME_PREFIX AND IS_DIRECTORY "${RUNTIME_PREFIX}")
    file(REAL_PATH "${RUNTIME_PREFIX}" _runtime)
    cmake_path(IS_PREFIX _runtime "${_results}" NORMALIZE _inside_runtime)
    if(_inside_runtime)
        message(
            FATAL_ERROR
            "RESULTS_DIRECTORY must be outside the completed runtime prefix"
        )
    endif()
endif()
if(EXISTS "${_results}")
    file(GLOB _contents "${_results}/*" "${_results}/.*")
    if(_contents OR NOT IS_DIRECTORY "${_results}")
        message(
            FATAL_ERROR
            "RESULTS_DIRECTORY must be fresh or empty: ${_results}"
        )
    endif()
endif()
file(MAKE_DIRECTORY "${_results}")
file(
    WRITE "${_results}/run.txt"
    "test_prefix=${_prefix}\nsuite=${TEST_SUITE}\nresults=${_results}\n"
)

function(fail_run reason)
    file(APPEND "${_results}/run.txt" "status=failed\nreason=${reason}\n")
    message(FATAL_ERROR "${reason}; records: ${_results}")
endfunction()

function(quote_cmake output value)
    string(REPLACE "\\" "\\\\" value "${value}")
    string(REPLACE "\"" "\\\"" value "${value}")
    string(REPLACE "$" "\\$" value "${value}")
    set(${output} "\"${value}\"" PARENT_SCOPE)
endfunction()

if(NOT TEST_SUITE MATCHES "^[a-z][a-z0-9-]*$")
    fail_run("TEST_SUITE must name a packaged suite")
endif()
if(DEFINED TEST_CASE)
    fail_run("TEST_CASE filtering is not supported by the one-test slice")
endif()
set(_suite_file "${_prefix}/cmake/suites/${TEST_SUITE}.cmake")
if(NOT EXISTS "${_suite_file}")
    fail_run("Unknown or missing suite: ${TEST_SUITE}")
endif()
include("${_suite_file}")
foreach(_field suite_family suite_executable suite_test suite_requires_runtime)
    if(NOT DEFINED ${_field} OR "${${_field}}" STREQUAL "")
        fail_run("Incomplete suite registration: ${_field}")
    endif()
endforeach()
if(
    NOT suite_family MATCHES "^(runtime-api|ddi|kfd)$"
    OR NOT suite_executable MATCHES "^bin/${suite_family}/[a-z0-9-]+$"
    OR NOT suite_test MATCHES "^[a-z][a-z0-9.-]+$"
    OR NOT suite_requires_runtime MATCHES "^(TRUE|FALSE)$"
)
    fail_run("Invalid suite registration")
endif()
set(_executable "${_prefix}/${suite_executable}")
if(NOT EXISTS "${_executable}" OR IS_DIRECTORY "${_executable}")
    fail_run("Missing required executable: ${_executable}")
endif()
set(_record "${_prefix}/share/rocm-runtime-tests/${suite_family}.txt")
if(NOT EXISTS "${_record}")
    fail_run("Missing build identity: ${_record}")
endif()
file(SHA256 "${_executable}" _executable_hash)
# Provenance is an opaque retained build log, not another manifest protocol.
# The executed hash is recorded separately for comparison with producer evidence.
file(COPY_FILE "${_record}" "${_results}/build.txt")
quote_cmake(_command "${_executable}")
if(suite_requires_runtime)
    if(NOT DEFINED RUNTIME_PREFIX OR NOT IS_ABSOLUTE "${RUNTIME_PREFIX}")
        fail_run("This suite requires an absolute RUNTIME_PREFIX")
    endif()
    file(REAL_PATH "${RUNTIME_PREFIX}" _runtime)
    cmake_path(IS_PREFIX _runtime "${_results}" NORMALIZE _inside_runtime)
    if(_inside_runtime)
        fail_run("RESULTS_DIRECTORY must be outside the completed runtime prefix")
    endif()
    set(_library "${_runtime}/lib/libamdf.so")
    if(NOT EXISTS "${_library}" OR IS_DIRECTORY "${_library}")
        fail_run("Missing selected AMDF image: ${_library}")
    endif()
    file(SHA256 "${_library}" _runtime_hash)
    quote_cmake(_argument "${_library}")
    string(APPEND _command " ${_argument}")
    # Retain intentional model/profiler interposition and other prepared inputs.
    if("$ENV{LD_LIBRARY_PATH}" STREQUAL "")
        set(ENV{LD_LIBRARY_PATH} "${_runtime}/lib")
    else()
        set(ENV{LD_LIBRARY_PATH} "${_runtime}/lib:$ENV{LD_LIBRARY_PATH}")
    endif()
    file(
        APPEND "${_results}/run.txt"
        "runtime_prefix=${_runtime}\nruntime_sha256=${_runtime_hash}\n"
    )
elseif(DEFINED RUNTIME_PREFIX)
    fail_run("${TEST_SUITE} does not consume RUNTIME_PREFIX")
endif()
file(
    APPEND "${_results}/run.txt"
    "required_test=${suite_test}\nexecutable_sha256=${_executable_hash}\ncommand=${_command}\nLD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}\nLD_PRELOAD=$ENV{LD_PRELOAD}\n"
)
quote_cmake(_test "${suite_test}")
file(WRITE "${_results}/CTestTestfile.cmake" "add_test(${_test} ${_command})\n")
set(CTEST_SOURCE_DIRECTORY "${_results}")
set(CTEST_BINARY_DIRECTORY "${_results}")
set(CTEST_SITE "runtime-tests")
set(CTEST_BUILD_NAME "${TEST_SUITE}")
ctest_start(Experimental)
ctest_test(
    RETURN_VALUE _result
    CAPTURE_CMAKE_ERROR _error
    OUTPUT_JUNIT
    "${_results}/junit.xml"
)
file(STRINGS "${_results}/Testing/TAG" _tag LIMIT_COUNT 1)
set(_xml "${_results}/Testing/${_tag}/Test.xml")
if(NOT EXISTS "${_xml}")
    fail_run("CTest did not retain executed inventory")
endif()
file(READ "${_xml}" _execution)
string(REGEX MATCHALL "<Test Status=\"[^\"]+\">" _executed "${_execution}")
string(REGEX MATCHALL "<Test Status=\"passed\">" _passed "${_execution}")
list(LENGTH _executed _executed_count)
list(LENGTH _passed _passed_count)
file(
    APPEND "${_results}/run.txt"
    "executed=${_executed_count}\npassed=${_passed_count}\n"
)
if(
    NOT _result EQUAL 0
    OR NOT _error EQUAL 0
    OR NOT _executed_count EQUAL 1
    OR NOT _passed_count EQUAL 1
)
    fail_run("Required test failed, was skipped or did not execute")
endif()
file(APPEND "${_results}/run.txt" "status=passed\n")
message(STATUS "Required test passed: ${suite_test}; records: ${_results}")
