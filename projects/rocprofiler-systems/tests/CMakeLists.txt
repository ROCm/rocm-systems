# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

#
# rocprofiler-systems tests
#
include_guard(GLOBAL)

set(_capchk_dir ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests)
set(_capchk_exe ${_capchk_dir}/rocprof-sys-capchk)
file(MAKE_DIRECTORY ${_capchk_dir})
execute_process(
    COMMAND
        ${CMAKE_CXX_COMPILER} -O2 -g -std=c++17
        ${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-capchk.cpp -o rocprof-sys-capchk
    WORKING_DIRECTORY ${_capchk_dir}
    OUTPUT_QUIET
    ERROR_QUIET
    RESULT_VARIABLE _capchk_result
)
if(NOT _capchk_result EQUAL 0)
    message(
        WARNING
        "rocprof-sys-capchk build failed (result ${_capchk_result}); install of capability checker will be skipped."
    )
endif()

# TODO: Remove this once TheRock is switched to CTests
if(ROCPROFSYS_BUILD_FOR_THEROCK)
    # Required to maintain compatibility with TheRock until it can be switched to CTests
    include(${CMAKE_CURRENT_LIST_DIR}/pytest/CMakeLists.txt)
else()
    include(${CMAKE_CURRENT_LIST_DIR}/pytest/CMakeLists.txt)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-pytest.cmake)
endif()

# ------------------------------------------------------------------------------#
#
# Move test files to build directory
#
# ------------------------------------------------------------------------------#

set(ROCPROFSYS_PYTHON_VALIDATION_FILES
    ${CMAKE_CURRENT_LIST_DIR}/validate-causal-json.py
    ${CMAKE_CURRENT_LIST_DIR}/validate-perfetto-proto.py
    ${CMAKE_CURRENT_LIST_DIR}/validate-rocpd.py
    ${CMAKE_CURRENT_LIST_DIR}/validate-timemory-json.py
)

set(ROCPROFSYS_TEST_SCRIPTS
    ${CMAKE_CURRENT_LIST_DIR}/run-rocprof-sys-pid.sh
    ${CMAKE_CURRENT_LIST_DIR}/get_default_nic.sh
    ${CMAKE_CURRENT_LIST_DIR}/generate_papi_nic_events.sh
    ${CMAKE_CURRENT_LIST_DIR}/amd_smi_data_parse.py
)

add_custom_target(
    copy-test-files
    ALL
    COMMAND
        ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${ROCPROFSYS_PYTHON_VALIDATION_FILES}
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests
    # copy_directory for directories (copy_if_different only works for files)
    COMMAND
        ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/rocpd-validation-rules
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${ROCPROFSYS_TEST_SCRIPTS}
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests
)

# ------------------------------------------------------------------------------#
#
# Pytests install
#
# ------------------------------------------------------------------------------#

if(ROCPROFSYS_INSTALL_TESTING)
    # Python Validation scripts
    install(
        PROGRAMS ${ROCPROFSYS_PYTHON_VALIDATION_FILES}
        DESTINATION share/rocprofiler-systems/tests
        COMPONENT rocprofiler-systems-tests
    )
    install(
        DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules
        DESTINATION share/rocprofiler-systems/tests
        COMPONENT rocprofiler-systems-tests
    )
    # Scripts
    install(
        PROGRAMS ${ROCPROFSYS_TEST_SCRIPTS}
        DESTINATION share/rocprofiler-systems/tests
        COMPONENT rocprofiler-systems-tests
    )
    # Capability check binary
    if(ROCPROFSYS_USE_PYTESTS AND EXISTS "${_capchk_exe}")
        install(
            PROGRAMS ${_capchk_exe}
            DESTINATION share/rocprofiler-systems/tests
            COMPONENT rocprofiler-systems-tests
        )
    endif()
endif()
